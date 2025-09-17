# image_processor.py — CrewAI-powered agentic frame cleaner
# Public API (unchanged):
#   process_image_frame_from_memory(frame_bgr, require_face: bool=False) -> dict | None
# Also exposes: is_blurry(gray), is_low_contrast(gray)

from __future__ import annotations
import os, json, tempfile
import numpy as np
import cv2
from PIL import Image

# ---- CrewAI / LangChain (optional; we fall back if unavailable/keys missing)
_CREW_AVAILABLE = True
try:
    from crewai import Agent, Task, Crew
    from langchain.tools import tool
    from langchain_anthropic import ChatAnthropic
except Exception:
    _CREW_AVAILABLE = False

try:
    import imagehash
    _HAS_IHASH = True
except Exception:
    _HAS_IHASH = False

# ===================== CV helpers =====================
BLUR_THRESHOLD = 100.0
CONTRAST_THRESHOLD = 20.0
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)
_FACE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def is_blurry(gray: np.ndarray, threshold=BLUR_THRESHOLD) -> bool:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var()) < threshold

def is_low_contrast(gray: np.ndarray, threshold=CONTRAST_THRESHOLD) -> bool:
    return float(gray.std()) < threshold

def _clahe(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE)
    return clahe.apply(gray)

def _unsharp(gray: np.ndarray, amount: float = 0.7, ksize: int = 5) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (ksize, ksize), 0)
    return cv2.addWeighted(gray, 1 + amount, blur, -amount, 0)

def _phash(bgr: np.ndarray) -> str:
    if _HAS_IHASH:
        im = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        return str(imagehash.phash(im))
    small = cv2.resize(bgr, (8, 8))
    g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return "".join("1" if v else "0" for v in (g > g.mean()).astype(np.uint8).flatten())

# ===================== CrewAI Tool wrapper =====================
if _CREW_AVAILABLE:
    @tool("image_clean_tool", return_direct=True)
    def image_clean_tool(input_json: str) -> str:
        """
        Tool: cleans an image file and writes cleaned image back.
        input_json: {"img_path": "...", "require_face": false}
        Returns JSON: {"cleaned_path": "...", "features": {...}, "agent": {...}} or {"rejected": true, "agent": {...}}
        """
        try:
            args = json.loads(input_json)
        except Exception:
            return json.dumps({"error": "invalid_json"})
        img_path = args.get("img_path")
        require_face = bool(args.get("require_face", False))
        if not img_path or not os.path.exists(img_path):
            return json.dumps({"error": "missing_img"})

        bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if bgr is None or bgr.size == 0:
            return json.dumps({"error": "empty_frame"})

        agent = {"name":"ImageCleaningAgent(CrewAI)", "version":"1.0", "actions":[], "reasons":[], "metrics":{}}

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        contrast = float(gray.std())
        agent["metrics"].update({"laplacian_var": lap_var, "contrast_std": contrast})

        if is_blurry(gray):
            agent["reasons"].append("blurry")
            return json.dumps({"rejected": True, "agent": agent})
        if is_low_contrast(gray):
            agent["reasons"].append("low_contrast")
            return json.dumps({"rejected": True, "agent": agent})
        if require_face:
            faces = _FACE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(64, 64))
            agent["metrics"]["faces_detected"] = int(len(faces))
            if len(faces) == 0:
                agent["reasons"].append("no_face")
                return json.dumps({"rejected": True, "agent": agent})

        eq = _clahe(gray);                 agent["actions"].append("clahe")
        sharp = _unsharp(eq, amount=0.7);  agent["actions"].append("unsharp")
        proc_bgr = cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)

        p_hash = _phash(proc_bgr)
        h, w = proc_bgr.shape[:2]
        features = {
            "dimensions": {"width": w, "height": h},
            "perceptual_hash": p_hash,
            "quality": {"laplacian_var": lap_var, "contrast_std": contrast}
        }

        out_path = img_path.replace(".jpg", "_clean.jpg").replace(".png", "_clean.png")
        cv2.imwrite(out_path, proc_bgr)
        return json.dumps({"cleaned_path": out_path, "features": features, "agent": agent})

# ===================== Public API (uses Crew if available) =====================
def process_image_frame_from_memory(image_frame: np.ndarray, require_face: bool = False) -> dict | None:
    """
    Returns:
      None (rejected) OR
      {
        "processed_image": np.ndarray (BGR),
        "features": { "perceptual_hash": str, "quality": {...}, "dimensions": {...} },
        "agent": {...}
      }
    """
    if image_frame is None or image_frame.size == 0:
        return None

    # Persist frame to a temp image so Crew Tool can operate on a path
    with tempfile.TemporaryDirectory() as td:
        in_img = os.path.join(td, "in.jpg")
        cv2.imwrite(in_img, image_frame)

        use_crew = _CREW_AVAILABLE and os.getenv("ANTHROPIC_API_KEY")
        if use_crew:
            try:
                llm = ChatAnthropic(
                    model=os.getenv("ANTHROPIC_MODEL","claude-3-5-haiku-20241022"),
                    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
                )
                agent = Agent(
                    role="Image Cleaner",
                    goal="Clean the input image using the tool and return JSON only.",
                    backstory="A careful computer-vision technician.",
                    llm=llm,
                    allow_delegation=False,
                    tools=[image_clean_tool],
                )
                task = Task(
                    description=f'Call image_clean_tool with JSON: {{"img_path":"{in_img}","require_face":{str(bool(require_face)).lower()}}}. Return ONLY JSON.',
                    expected_output="JSON with cleaned_path/features/agent or rejected.",
                    agent=agent,
                )
                crew = Crew(agents=[agent], tasks=[task])
                out_text = str(crew.kickoff())
                start = out_text.find("{"); end = out_text.rfind("}")
                if start != -1 and end != -1:
                    out = json.loads(out_text[start:end+1])
                else:
                    out = json.loads(image_clean_tool.run(json.dumps({"img_path": in_img, "require_face": require_face})))
            except Exception:
                out = json.loads(image_clean_tool.run(json.dumps({"img_path": in_img, "require_face": require_face})))
        else:
            # Local deterministic path (no Crew/LLM)
            gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)
            lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            contrast = float(gray.std())
            if is_blurry(gray) or is_low_contrast(gray):
                return None
            if require_face:
                faces = _FACE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(64, 64))
                if len(faces) == 0:
                    return None
            eq = _clahe(gray)
            sharp = _unsharp(eq, amount=0.7)
            proc_bgr = cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)
            p_hash = _phash(proc_bgr)
            h, w = proc_bgr.shape[:2]
            features = {
                "dimensions": {"width": w, "height": h},
                "perceptual_hash": p_hash,
                "quality": {"laplacian_var": lap_var, "contrast_std": contrast}
            }
            agent = {"name":"ImageCleaningAgent(Local)","version":"1.0","actions":["clahe","unsharp"],"reasons":[],"metrics":{"laplacian_var":lap_var,"contrast_std":contrast}}
            return {"processed_image": proc_bgr, "features": features, "agent": agent}

        if out.get("rejected"):
            return None
        if "cleaned_path" not in out:
            return None

        proc = cv2.imread(out["cleaned_path"], cv2.IMREAD_COLOR)
        return {"processed_image": proc, "features": out.get("features", {}), "agent": out.get("agent", {})}
