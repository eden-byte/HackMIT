import librosa
import numpy as np
from scipy.signal import butter, lfilter

# --- Configuration for Audio Processing ---
NOISE_REDUCTION_THRESHOLD_DB = -40 # Audio below this dB level is considered noise
FINAL_SAMPLE_RATE = 16000
NORMALIZATION_PEAK_DB = -1.0

def db_to_linear(db):
    """Converts decibels to a linear amplitude scale."""
    return 10.0**(db / 20.0)

def reduce_noise_by_gating(y, sr):
    """
    Reduces noise using a simple power-based noise gate.
    Quieter sections are attenuated.
    """
    # Calculate the Short-Time Fourier Transform (STFT)
    S = librosa.stft(y)

    # Convert amplitude to dB
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)

    # Create a mask: 1 for signal, 0 for noise
    mask = np.where(S_db > NOISE_REDUCTION_THRESHOLD_DB, 1, 0)

    # Apply the mask to the original spectrogram (with phase)
    S_denoised = S * mask

    # Inverse STFT to get back to the time domain
    y_denoised = librosa.istft(S_denoised)
    return y_denoised


def process_audio_clip_from_numpy(audio_array, sample_rate):
    """
    Cleans a raw NumPy audio array with a simplified and robust pipeline.
    """
    try:
        # --- FIX: Rock-solid check for silence or invalid input ---
        if audio_array is None or not np.any(audio_array):
            print("Skipping silent or empty audio clip.")
            return None

        # 1. Reduce background noise using a noise gate
        denoised_samples = reduce_noise_by_gating(audio_array, sample_rate)

        # 2. Peak Normalization
        peak_value = np.max(np.abs(denoised_samples))
        if peak_value > 0:
            target_amplitude = db_to_linear(NORMALIZATION_PEAK_DB)
            normalized_samples = denoised_samples / peak_value * target_amplitude
        else:
            normalized_samples = denoised_samples # It's silent

        # 3. Resample to the final standard rate
        resampled_samples = librosa.resample(y=normalized_samples, orig_sr=sample_rate, target_sr=FINAL_SAMPLE_RATE)

        # Another safety check after all processing
        if not np.any(resampled_samples):
            print("Skipping feature extraction for silent audio clip post-processing.")
            return None

        # 4. Extract features from the final, clean audio
        mfccs = librosa.feature.mfcc(y=resampled_samples, sr=FINAL_SAMPLE_RATE, n_mfcc=13)

        features = {
            'mfcc_mean': np.mean(mfccs, axis=1).tolist(),
            'sample_rate': FINAL_SAMPLE_RATE
        }

        return {
            "features": features,
            "processed_audio_array": resampled_samples,
            "sample_rate": FINAL_SAMPLE_RATE
        }
    except Exception as e:
        # The error was happening here, so this provides more specific context.
        print(f"Error during audio feature extraction (MFCC): {e}")
        return None
