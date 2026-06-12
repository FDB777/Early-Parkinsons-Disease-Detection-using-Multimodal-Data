import os
import json
import nolds
import neurokit2 as nk
import numpy as np
import pandas as pd
import parselmouth
import re
from pyrpde import rpde
import joblib

# -----------------------------
# SETTINGS
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = os.path.join(BASE_DIR, "voice_recordings")
MODEL_PATH = os.path.join(BASE_DIR, "voice_knn_og2.pkl")
DATASET_PATH = r"C:\Users\fbash\OneDrive\Desktop\github\voice_models\voice_normalized.csv"

MIN_PITCH = 75
MAX_PITCH = 500

# Praat params
FROM_TIME = 0
TO_TIME = 0
PERIOD_FLOOR = 0.0001
PERIOD_CEILING = 0.02
MAX_PERIOD_FACTOR = 1.3
MAX_AMPLITUDE_FACTOR = 1.6

_NORMALIZATION_CACHE = {}


def safe_mean(x, default=0.0):
    return float(np.mean(x)) if len(x) else default


def safe_min(x, default=0.0):
    return float(np.min(x)) if len(x) else default


def safe_max(x, default=0.0):
    return float(np.max(x)) if len(x) else default


def compute_jitter_ddp(periods):
    if len(periods) < 3:
        return 0.0

    diff1 = np.diff(periods)
    diff2 = np.diff(diff1)
    denom = np.mean(periods) + 1e-9

    return float(np.mean(np.abs(diff2)) / denom)


def compute_ppe(periods, bins=10):
    if len(periods) < 2 or bins < 2:
        return 0.0

    log_periods = np.log10(periods + 1e-12)
    hist, _ = np.histogram(log_periods, bins=bins, density=True)

    total = np.sum(hist)
    if total <= 0:
        return 0.0

    p = hist / total
    p = p[p > 0]

    ent = -np.sum(p * np.log(p))
    norm = np.log(bins)

    if norm <= 0:
        return 0.0

    return float(ent / norm)


def compute_nhr_from_hnr(hnr):
    return float(1.0 / (hnr + 1e-9)) if np.isfinite(hnr) else 0.0


def _extract_rqa_value(rqa_obj, key, default=0.0):
    try:
        if isinstance(rqa_obj, dict):
            val = rqa_obj.get(key, default)

            if isinstance(val, (list, tuple, np.ndarray)):
                return float(val[0]) if len(val) else default

            return float(val)

        if hasattr(rqa_obj, "__getitem__"):
            val = rqa_obj[key]

            if hasattr(val, "iloc"):
                return float(val.iloc[0])

            if isinstance(val, (list, tuple, np.ndarray)):
                return float(val[0]) if len(val) else default

            return float(val)

    except Exception:
        pass

    return float(default)


def compute_rqa_measures(signal):
    if len(signal) < 64:
        return 0.0, 0.0, 0.0

    try:
        rqa = nk.complexity_rqa(signal, dimension=3, delay=1)

        spread1 = _extract_rqa_value(rqa, "Lmax", 0.0)
        spread2 = _extract_rqa_value(rqa, "LAM", 0.0)
        d2 = _extract_rqa_value(rqa, "DET", 0.0)

        return spread1, spread2, d2

    except Exception:
        return 0.0, 0.0, 0.0


def safe_praat_call(obj, command, *args, default=0.0):
    try:
        value = parselmouth.praat.call(obj, command, *args)

        if value is None or not np.isfinite(value):
            return float(default)

        return float(value)

    except Exception:
        return float(default)


def extract_features(file_path, filename):

    sound = parselmouth.Sound(file_path)

    samples = sound.values.T.flatten().astype(np.float64)

    max_abs = np.max(np.abs(samples)) if samples.size else 0.0

    if max_abs > 0:
        samples = samples / max_abs

    if samples.size >= 2:
        samples = np.diff(samples)

    pitch = sound.to_pitch(pitch_floor=MIN_PITCH, pitch_ceiling=MAX_PITCH)

    f0 = pitch.selected_array["frequency"]
    f0 = f0[f0 > 0]

    if len(f0) == 0:
        raise ValueError("No voiced pitch frames detected")

    f0_mean = safe_mean(f0)
    f0_min = safe_min(f0)
    f0_max = safe_max(f0)

    periods = 1.0 / (f0 + 1e-9)

    pp = parselmouth.praat.call(
        sound,
        "To PointProcess (periodic, cc)",
        MIN_PITCH,
        MAX_PITCH,
    )

    jitter_local = safe_praat_call(
        pp,
        "Get jitter (local)",
        FROM_TIME,
        TO_TIME,
        PERIOD_FLOOR,
        PERIOD_CEILING,
        MAX_PERIOD_FACTOR,
    )

    jitter_rap = safe_praat_call(
        pp,
        "Get jitter (rap)",
        FROM_TIME,
        TO_TIME,
        PERIOD_FLOOR,
        PERIOD_CEILING,
        MAX_PERIOD_FACTOR,
    )

    jitter_ppq = safe_praat_call(
        pp,
        "Get jitter (ppq5)",
        FROM_TIME,
        TO_TIME,
        PERIOD_FLOOR,
        PERIOD_CEILING,
        MAX_PERIOD_FACTOR,
    )

    jitter_ddp = compute_jitter_ddp(periods)

    jitter_abs = safe_praat_call(
        pp,
        "Get jitter (local, absolute)",
        FROM_TIME,
        TO_TIME,
        PERIOD_FLOOR,
        PERIOD_CEILING,
        MAX_PERIOD_FACTOR,
    )

    shimmer_local = safe_praat_call(
        [sound, pp],
        "Get shimmer (local)",
        FROM_TIME,
        TO_TIME,
        PERIOD_FLOOR,
        PERIOD_CEILING,
        MAX_PERIOD_FACTOR,
        MAX_AMPLITUDE_FACTOR,
    )

    shimmer_db = safe_praat_call(
        [sound, pp],
        "Get shimmer (local_dB)",
        FROM_TIME,
        TO_TIME,
        PERIOD_FLOOR,
        PERIOD_CEILING,
        MAX_PERIOD_FACTOR,
        MAX_AMPLITUDE_FACTOR,
    )

    shimmer_apq3 = safe_praat_call(
        [sound, pp],
        "Get shimmer (apq3)",
        FROM_TIME,
        TO_TIME,
        PERIOD_FLOOR,
        PERIOD_CEILING,
        MAX_PERIOD_FACTOR,
        MAX_AMPLITUDE_FACTOR,
    )

    shimmer_apq5 = safe_praat_call(
        [sound, pp],
        "Get shimmer (apq5)",
        FROM_TIME,
        TO_TIME,
        PERIOD_FLOOR,
        PERIOD_CEILING,
        MAX_PERIOD_FACTOR,
        MAX_AMPLITUDE_FACTOR,
    )

    shimmer_apq = safe_praat_call(
        [sound, pp],
        "Get shimmer (apq11)",
        FROM_TIME,
        TO_TIME,
        PERIOD_FLOOR,
        PERIOD_CEILING,
        MAX_PERIOD_FACTOR,
        MAX_AMPLITUDE_FACTOR,
    )

    shimmer_dda = safe_praat_call(
        [sound, pp],
        "Get shimmer (dda)",
        FROM_TIME,
        TO_TIME,
        PERIOD_FLOOR,
        PERIOD_CEILING,
        MAX_PERIOD_FACTOR,
        MAX_AMPLITUDE_FACTOR,
    )

    harmonicity = sound.to_harmonicity_cc()

    hnr = safe_praat_call(
        harmonicity,
        "Get mean",
        0,
        0,
    )

    nhr = compute_nhr_from_hnr(hnr)

    try:
        rpde_val, _ = rpde(
            samples,
            tau=30,
            dim=4,
            epsilon=0.01,
        )

        rpde_val = float(rpde_val)

    except Exception:
        rpde_val = 0.0

    try:
        f0_diff = np.diff(f0) if len(f0) > 1 else f0
        dfa_val = float(nolds.dfa(f0_diff)) if len(f0_diff) >= 64 else 0.0

    except Exception:
        dfa_val = 0.0

    spread1, spread2, d2 = compute_rqa_measures(samples)

    ppe_val = compute_ppe(periods)

    return {
        "name": filename,
        "MDVP:Fo(Hz)": f0_mean,
        "MDVP:Fhi(Hz)": f0_max,
        "MDVP:Flo(Hz)": f0_min,
        "MDVP:Jitter(%)": jitter_local,
        "MDVP:Jitter(Abs)": jitter_abs,
        "MDVP:RAP": jitter_rap,
        "MDVP:PPQ": jitter_ppq,
        "Jitter:DDP": jitter_ddp,
        "MDVP:Shimmer": shimmer_local,
        "MDVP:Shimmer(dB)": shimmer_db,
        "Shimmer:APQ3": shimmer_apq3,
        "Shimmer:APQ5": shimmer_apq5,
        "MDVP:APQ": shimmer_apq,
        "Shimmer:DDA": shimmer_dda,
        "HNR": hnr,
        "NHR": nhr,
        "RPDE": rpde_val,
        "DFA": dfa_val,
        "spread1": spread1,
        "spread2": spread2,
        "D2": d2,
        "PPE": ppe_val,
        "status": "unknown",
    }


def run_voice_predictions(feature_rows):

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)

    df = pd.DataFrame(feature_rows)

    def _sanitize_cols(cols):
        return [re.sub(r"[^A-Za-z0-9_]+", "_", str(c)) for c in cols]

    if hasattr(model, "feature_names_in_"):
        feature_cols = list(model.feature_names_in_)
    else:
        feature_cols = [
            col for col in df.columns if col not in ["name", "status"]
        ]

    missing = [
        col for col in feature_cols
        if col not in df.columns
    ]

    used_sanitized = False
    if missing:
        df_sanitized = df.copy()
        df_sanitized.columns = _sanitize_cols(df_sanitized.columns)
        missing_sanitized = [
            col for col in feature_cols
            if col not in df_sanitized.columns
        ]
        if not missing_sanitized:
            df = df_sanitized
            used_sanitized = True
            missing = []

    if missing:
        raise ValueError(
            f"Missing required feature columns for model: {missing}"
        )

    x_data = df[feature_cols].copy()

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Normalization dataset not found at {DATASET_PATH}"
        )

    cache_key = (tuple(feature_cols), used_sanitized)
    stats = _NORMALIZATION_CACHE.get(cache_key)

    if stats is None:
        ds = pd.read_csv(DATASET_PATH)
        if used_sanitized:
            ds.columns = _sanitize_cols(ds.columns)

        missing_ds = [col for col in feature_cols if col not in ds.columns]

        if missing_ds:
            raise ValueError(
                f"Dataset missing required columns: {missing_ds}"
            )

        ds_numeric = ds[feature_cols].apply(
            pd.to_numeric,
            errors="coerce",
        )

        means = ds_numeric.mean()
        stds = ds_numeric.std(ddof=0).replace(0, 1.0)

        stats = {"means": means, "stds": stds}
        _NORMALIZATION_CACHE[cache_key] = stats

    x_data = (x_data - stats["means"]) / stats["stds"]

    model_to_use = model
    if hasattr(model, "steps"):
        scaler = getattr(model, "named_steps", {}).get("scaler")
        if (
            scaler is not None
            and hasattr(scaler, "mean_")
            and hasattr(scaler, "scale_")
        ):
            if (
                np.max(np.abs(scaler.mean_)) < 0.1
                and np.max(np.abs(scaler.scale_ - 1.0)) < 0.1
            ):
                model_to_use = (
                    model.named_steps.get("knn") or model
                )

    x_input = x_data
    if not hasattr(model_to_use, "feature_names_in_"):
        x_input = x_data.to_numpy()

    probs = model_to_use.predict_proba(x_input)[:, 1]

    labels = np.where(
        probs >= 0.5,
        "PD",
        "NON_PD",
    )

    results = []

    for name, prob, label in zip(
        df["name"].tolist(),
        probs.tolist(),
        labels.tolist(),
    ):

        results.append(
            {
                "name": name,
                "voice_probability": round(float(prob), 4),
                "voice_prediction": label,
            }
        )

    return results


def main():

    features_list = []

    if not os.path.isdir(AUDIO_DIR):
        raise FileNotFoundError(
            f"Audio directory not found: {AUDIO_DIR}"
        )

    for filename in os.listdir(AUDIO_DIR):

        if not filename.lower().endswith(
            (".wav", ".ogg")
        ):
            continue

        file_path = os.path.join(
            AUDIO_DIR,
            filename,
        )

        try:
            features = extract_features(
                file_path,
                filename,
            )

            features_list.append(features)

            print(f"Processed {filename}")

        except Exception as exc:

            print(
                f"Skipped {filename} due to error: {exc}"
            )

    if not features_list:
        print("No valid WAV files processed.")
        return

    predictions = run_voice_predictions(
        features_list
    )

    print(
        json.dumps(
            predictions,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
