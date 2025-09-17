# audio_processor.py — CrewAI-powered agentic audio cleaner
# Public API (unchanged):
#   process_audio_clip_from_numpy(audio_array, sample_rate) -> dict | None

from __future__ import annotations
import os, io, json, tempfile, warnings
import numpy as np
import soundfile as sf
import librosa
from scipy.signal import butter, filtfilt

# ---- CrewAI / LangChain (optional; we fall back if unavailable/keys missing)
_CREW_AVAILABLE = True
try:
    from crewai import Agent, Task, Crew
    from langchain.tools import tool
    from langchain_anthropic import ChatAnthropic
except Exception:
    _CREW_AVAILABLE = False

# ===================== Low-level DSP helpers (deterministic) =====================
NOISE_REDUCTION_THRESHOLD_DB = -40.0
FINAL_SAMPLE_RATE = 16000
TARGET_RMS_DBFS = -20.0

def _to_float32(x: np.ndarray) -> np.ndarray:
    if x is None:
        return np.zeros(0, dtype=np.float32)
    x = x.astype(np.float32, copy=False)
    if np.max(np.abs(x)) > 1.5:  # looks like raw PCM ints
        x = x / np.max(np.abs(x))
    return x

def _mono(y: np.ndarray) -> np.ndarray:
    return y if y.ndim == 1 else np.mean(y, axis=1)

def _resample(y: np.ndarray, sr: int, target: int = FINAL_SAMPLE_RATE):
    if sr == target:
        return y, sr
    y = librosa.resample(y=y, orig_sr=sr, target_sr=target, res_type="kaiser_best")
    return y.astype(np.float32, copy=False), target

def _highpass(y: np.ndarray, sr: int, cutoff=50.0, order=4) -> np.ndarray:
    b, a = butter(order, cutoff / (sr / 2.0), btype="high")
    return filtfilt(b, a, y).astype(np.float32, copy=False)

def _spectral_gate(y: np.ndarray, sr: int, floor_db: float) -> np.ndarray:
    S = librosa.stft(y, n_fft=1024, hop_length=256, window="hann")
    mag, phase = np.abs(S), np.angle(S)
    noise = np.quantile(mag, 0.10, axis=1, keepdims=True)
    thresh = noise * (10 ** (abs(floor_db) / 20.0))
    mask = (mag >= thresh).astype(np.float32)
    S_clean = (mag * mask) * np.exp(1j * phase)
    y_hat = librosa.istft(S_clean, hop_length=256, window="hann", length=len(y))
    return y_hat.astype(np.float32, copy=False)

def _trim_silence(y: np.ndarray, top_db=30.0):
    yt, idx = librosa.effects.trim(y, top_db=top_db)
    trimmed = idx is not None and (idx[0] > 0 or idx[1] < len(y))
    return (yt if yt.size else y), bool(trimmed)

def _safe_normalize(y: np.ndarray, target_dbfs: float = TARGET_RMS_DBFS):
    eps = 1e-9
    rms = float(np.sqrt(np.mean(y * y) + eps))
    if rms < eps:
        return y, 0.0
    target_rms = 10 ** (target_dbfs / 20.0)
    gain = target_rms / rms
    peak = np.max(np.abs(y)) * gain
    if peak > 0.99:
        gain *= 0.99 / peak
    return (y * gain).astype(np.float32, copy=False), 20.0 * np.log10(max(gain, eps))

def _features(y: np.ndarray, sr: int) -> dict:
    eps = 1e-9
    rms = float(np.sqrt(np.mean(y * y) + eps))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y, frame_length=1024, hop_length=256)))
    sc = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return {
        "rms": rms,
        "zcr": zcr,
        "spectral_centroid": sc,
        "mfcc_mean": np.mean(mfcc, axis=1).tolist(),
        "mfcc_std": np.std(mfcc, axis=1).tolist(),
    }

# ===================== CrewAI Tool wrapper =====================
if _CREW_AVAILABLE:
    @tool("audio_clean_tool", return_direct=True)
    def audio_clean_tool(input_json: str) -> str:
        """
        Tool: cleans an audio file (wav) and writes cleaned wav back.
        input_json: {"wav_path": "...", "sample_rate_hint": 44100}
        Returns JSON: {"cleaned_path": "...", "features": {...}, "agent": {...}}
        """
        try:
            args = json.loads(input_json)
        except Exception:
            return json.dumps({"error": "invalid_json"})

        wav_path = args.get("wav_path")
        sr_hint = int(args.get("sample_rate_hint", 44100))
        if not wav_path or not os.path.exists(wav_path):
            return json.dumps({"error": "missing_wav"})

        y, sr = sf.read(wav_path, dtype="float32", always_2d=False)
        if y is None or len(y) == 0:
            return json.dumps({"error": "empty_input"})

        agent = {"name":"AudioCleaningAgent(CrewAI)", "version":"1.0", "actions":[], "reasons":[], "metrics":{}}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            y = _to_float32(y)
            agent["metrics"]["input_peak"] = float(np.max(np.abs(y)))
            agent["metrics"]["input_rms"] = float(np.sqrt(np.mean(y * y) + 1e-9))

            y = _mono(y)
            y, sr = _resample(y, sr, FINAL_SAMPLE_RATE); agent["actions"].append(f"resample_{sr}hz")
            y = y - float(np.mean(y));                                agent["actions"].append("remove_dc")
            y = _highpass(y, sr, cutoff=50.0);                       agent["actions"].append("highpass_50hz")
            y = _spectral_gate(y, sr, floor_db=NOISE_REDUCTION_THRESHOLD_DB); agent["actions"].append("spectral_gate")
            y, trimmed = _trim_silence(y, top_db=30.0)
            if trimmed: agent["actions"].append("trim_silence")

            if np.sqrt(np.mean(y * y)) < 1e-4:
                agent["reasons"].append("silent_after_cleaning")
                return json.dumps({"rejected": True, "agent": agent})

            y, gain_db = _safe_normalize(y, TARGET_RMS_DBFS)
            agent["actions"].append("normalize")
            agent["metrics"]["gain_db"] = round(float(gain_db), 2)

            feats = _features(y, sr)

            cleaned_path = wav_path.replace(".wav", "_clean.wav")
            sf.write(cleaned_path, y, sr)
            return json.dumps({"cleaned_path": cleaned_path, "features": feats, "agent": agent})

# ===================== Public API (uses Crew if available) =====================
def process_audio_clip_from_numpy(audio_array: np.ndarray, sample_rate: int):
    """
    Returns:
      None (rejected) OR
      {
        "processed_audio_array": np.ndarray,
        "sample_rate": 16000,
        "features": {...},
        "agent": {...}
      }
    """
    # Write input to a temp wav, run the Crew tool (or local pipeline), read back
    if audio_array is None or len(audio_array) == 0:
        return None

    with tempfile.TemporaryDirectory() as td:
        in_wav = os.path.join(td, "in.wav")
        sf.write(in_wav, audio_array, sample_rate)

        # If Crew available & we have an Anthropic key, use a tiny single-agent Crew.
        use_crew = _CREW_AVAILABLE and os.getenv("ANTHROPIC_API_KEY")

        if use_crew:
            try:
                llm = ChatAnthropic(
                    model=os.getenv("ANTHROPIC_MODEL","claude-3-5-haiku-20241022"),
                    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
                )
                agent = Agent(
                    role="Audio Cleaner",
                    goal="Clean the input wav using the tool and return JSON only.",
                    backstory="A precise DSP technician.",
                    llm=llm,
                    allow_delegation=False,
                    tools=[audio_clean_tool],
                )
                task = Task(
                    description=f'Call audio_clean_tool with JSON: {{"wav_path":"{in_wav}","sample_rate_hint":{sample_rate}}}. Return ONLY JSON.',
                    expected_output="JSON with cleaned_path/features/agent or rejected.",
                    agent=agent,
                )
                crew = Crew(agents=[agent], tasks=[task])
                out_text = str(crew.kickoff())
                # Try parse JSON
                start = out_text.find("{")
                end = out_text.rfind("}")
                if start != -1 and end != -1:
                    out = json.loads(out_text[start:end+1])
                else:
                    # last resort: call tool directly
                    out = json.loads(audio_clean_tool.run(json.dumps({"wav_path": in_wav, "sample_rate_hint": sample_rate})))
            except Exception:
                # Fallback: call tool directly
                out = json.loads(audio_clean_tool.run(json.dumps({"wav_path": in_wav, "sample_rate_hint": sample_rate})))
        else:
            # No Crew/LLM: run local deterministic pipeline equivalent
            y, sr = sf.read(in_wav, dtype="float32", always_2d=False)
            if y is None or len(y) == 0:
                return None
            agent = {"name":"AudioCleaningAgent(Local)", "version":"1.0", "actions":[], "reasons":[], "metrics":{}}
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                y = _to_float32(y)
                agent["metrics"]["input_peak"] = float(np.max(np.abs(y)))
                agent["metrics"]["input_rms"] = float(np.sqrt(np.mean(y * y) + 1e-9))
                y = _mono(y)
                y, sr = _resample(y, sample_rate, FINAL_SAMPLE_RATE); agent["actions"].append(f"resample_{sr}hz")
                y = y - float(np.mean(y));                                agent["actions"].append("remove_dc")
                y = _highpass(y, sr, cutoff=50.0);                       agent["actions"].append("highpass_50hz")
                y = _spectral_gate(y, sr, floor_db=NOISE_REDUCTION_THRESHOLD_DB); agent["actions"].append("spectral_gate")
                y, trimmed = _trim_silence(y, top_db=30.0)
                if trimmed: agent["actions"].append("trim_silence")
                if np.sqrt(np.mean(y * y)) < 1e-4:
                    agent["reasons"].append("silent_after_cleaning")
                    return None
                y, gain_db = _safe_normalize(y, TARGET_RMS_DBFS)
                agent["actions"].append("normalize")
                agent["metrics"]["gain_db"] = round(float(gain_db), 2)
                feats = _features(y, sr)
                return {"processed_audio_array": y, "sample_rate": sr, "features": feats, "agent": agent}

        if out.get("rejected"):
            return None
        if "cleaned_path" not in out:
            return None

        y2, sr2 = sf.read(out["cleaned_path"], dtype="float32", always_2d=False)
        return {"processed_audio_array": y2, "sample_rate": sr2, "features": out.get("features", {}), "agent": out.get("agent", {})}
