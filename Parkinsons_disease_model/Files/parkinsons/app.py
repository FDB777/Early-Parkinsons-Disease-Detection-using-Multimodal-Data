from __future__ import annotations
from datetime import datetime, UTC
import importlib.util
import json
import os
from pathlib import Path
import re
from typing import Any
import cv2
import joblib
import mysql.connector
import numpy as np
import nibabel as nib
import nolds
import neurokit2 as nk
import parselmouth
import pandas as pd
from skimage.transform import resize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from flask import Flask, flash, redirect, render_template, request, session, url_for
from mysql.connector import Error
from pyrpde import rpde
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from tensorflow.keras.models import Model, load_model
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
app.secret_key = "dev-secret-key-change-if-needed"
app.config["UPLOAD_FOLDER"] = "static/uploads"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
app.config["DB_HOST"] = os.getenv("DB_HOST", "127.0.0.1")
app.config["DB_PORT"] = int(os.getenv("DB_PORT", "3306"))
app.config["DB_USER"] = os.getenv("DB_USER", "root")
app.config["DB_PASSWORD"] = os.getenv("DB_PASSWORD", "fdb7")
app.config["DB_NAME"] = os.getenv("DB_NAME", "parkinsons")

ALLOWED_AUDIO = {"wav"}
ALLOWED_IMAGE = {"jpg", "jpeg", "png"}
ALLOWED_MRI = {"jpg", "jpeg", "png", "nii", "nii.gz"}
IMAGE_MIME_MAP = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}
MRI_MIME_MAP = {
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "nii": "nii",
    "nii.gz": "nii",
}
DEEP_REGION_MIN_AFFECTED_PERCENT = 25.0
DEEP_REGION_PENALTY_FACTOR = 0.1
OCCLUSION_INFLUENCE_PERCENTILE = 85.0

VOICE_DATASET_PATH = os.getenv("VOICE_DATASET_PATH", r"C:\\Users\\fbash\\OneDrive\\Desktop\\github\\voice_models\\voice_dataset.csv")
_NORMALIZATION_CACHE: dict[tuple, dict[str, Any]] = {}

PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "project"))
MRI_DATASET_MEAN_PATH = os.getenv("MRI_DATASET_MEAN_PATH", os.path.join(PROJECT_DIR, "mri", "dataset_mean.npy"))
MRI_DATASET_STD_PATH = os.getenv("MRI_DATASET_STD_PATH", os.path.join(PROJECT_DIR, "mri", "dataset_std.npy"))
MODEL_CACHE: dict[str, Any] = {}
_HANDWRITING_PREPROCESS_FN = None
MODALITY_DB_CONFIG = {
    "VOICE": {
        "table": "processed_voice",
        "pk": "voice_id",
        "raw_path_col": "raw_voice_path",
        "mime_col": None,
        "processed_path_col": None,
    },
    "HANDWRITING": {
        "table": "processed_handwriting",
        "pk": "handwriting_id",
        "raw_path_col": "raw_handwriting_path",
        "mime_col": "raw_handwriting_mime",
        "processed_path_col": None,
    },
    "MRI": {
        "table": "processed_mri",
        "pk": "mri_id",
        "raw_path_col": "raw_mri_path",
        "mime_col": "raw_mri_mime",
        "processed_path_col": "processed_mri_path",
    },
}

def get_db_connection():
    return mysql.connector.connect(
        host=app.config["DB_HOST"],
        port=app.config["DB_PORT"],
        user=app.config["DB_USER"],
        password=app.config["DB_PASSWORD"],
        database=app.config["DB_NAME"],
    )

def compact_json(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"))

def close_db(cursor, conn) -> None:
    if cursor:
        cursor.close()
    if conn:
        conn.close()

def ensure_user_table_compatibility(cursor, conn):
    cursor.execute(
        """
        SELECT CHARACTER_MAXIMUM_LENGTH AS max_len
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = %s
          AND TABLE_NAME = 'user'
          AND COLUMN_NAME = 'password_hash'
        """,
        (app.config["DB_NAME"],),
    )
    row = cursor.fetchone()
    max_len = int(row["max_len"]) if row and row.get("max_len") else 0
    if max_len and max_len < 128:
        cursor.execute("ALTER TABLE `user` MODIFY password_hash VARCHAR(255) NOT NULL")
        conn.commit()

def get_normalized_extension(filename: str) -> str:
    lowered = filename.lower()
    if lowered.endswith(".nii.gz"):
        return "nii.gz"
    if "." not in lowered:
        return ""
    return lowered.rsplit(".", 1)[1]

def allowed_file(filename: str, allowed_set: set[str]) -> bool:
    return get_normalized_extension(filename) in allowed_set

def save_upload(file_obj, prefix: str) -> str:
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    safe_name = secure_filename(file_obj.filename)
    unique_suffix = os.urandom(4).hex()
    final_name = f"{prefix}_{unique_suffix}_{safe_name}"
    abs_path = os.path.join(app.config["UPLOAD_FOLDER"], final_name)
    file_obj.save(abs_path)
    return abs_path.replace("\\", "/")
def require_login(message: str, endpoint: str):
    if "user_id" not in session:
        flash(message, "error")
        return redirect(url_for(endpoint))
    return None

def validate_uploads(voice, handwriting, mri) -> list[str]:
    errors: list[str] = []
    if voice and voice.filename and not allowed_file(voice.filename, ALLOWED_AUDIO):
        errors.append("Voice file must be .wav")

    if handwriting and handwriting.filename and not allowed_file(handwriting.filename, ALLOWED_IMAGE):
        errors.append("Handwriting file must be .jpg, .jpeg, or .png")

    if mri and mri.filename and not allowed_file(mri.filename, ALLOWED_MRI):
        errors.append("MRI file must be .jpg, .jpeg, .png, .nii, or .nii.gz")

    if not ((voice and voice.filename) or (handwriting and handwriting.filename) or (mri and mri.filename)):
        errors.append("Upload at least one modality (voice, handwriting, or MRI).")

    return errors

def weighted_average(modality_probs: dict[str, float]) -> float:
    if not modality_probs:
        raise ValueError("No modality probabilities provided")

    weights = {
        "VOICE": 0.20,
        "HANDWRITING": 0.30,
        "MRI": 0.50,
    }

    total_weight = 0.0
    weighted_sum = 0.0

    for modality, prob in modality_probs.items():
        weight = weights.get(modality, 0.0)
        if weight <= 0:
            continue
        weighted_sum += prob * weight
        total_weight += weight

    if total_weight <= 0:
        raise ValueError("No valid modality weights for weighted average")

    return float(weighted_sum / total_weight)

def get_model_ids(cursor) -> tuple[dict[str, int], int]:
    model_ids = {
        "VOICE": get_or_create_model(cursor, "voice_lgbm_mdvp_v1", "VOICE"),
        "HANDWRITING": get_or_create_model(cursor, "hand_cnn_lgbm_v1", "HANDWRITING"),
        "MRI": get_or_create_model(cursor, "mri_cnn_lgbm_occlusion_v1", "MRI"),
    }
    ensemble_model_id = get_or_create_model(cursor, "voice_hand_mri_soft_vote_v1", "ENSEMBLE")
    return model_ids, ensemble_model_id

def process_voice_upload(cursor, input_id: int, user_id: int, voice_file, model_id: int):
    voice_path = save_upload(voice_file, f"voice_u{user_id}")
    voice_ext = get_normalized_extension(voice_file.filename)
    upsert_processed_modality(cursor, input_id, "VOICE", voice_path, voice_file.mimetype or f"audio/{voice_ext}")
    voice_prob, voice_features, voice_feature_map = predict_voice(voice_path)
    update_modality_inference(cursor, input_id, "VOICE", voice_features, model_id)
    upsert_modality_prediction(cursor, input_id, model_id, "VOICE", voice_prob)
    return voice_prob, voice_feature_map

def process_handwriting_upload(cursor, input_id: int, user_id: int, handwriting_file, model_id: int):
    hand_path = save_upload(handwriting_file, f"hw_u{user_id}")
    hand_ext = get_normalized_extension(handwriting_file.filename)
    upsert_processed_modality(cursor, input_id, "HANDWRITING", hand_path, IMAGE_MIME_MAP[hand_ext])
    hand_prob, hand_features = predict_handwriting(hand_path)
    update_modality_inference(cursor, input_id, "HANDWRITING", hand_features, model_id)
    upsert_modality_prediction(cursor, input_id, model_id, "HANDWRITING", hand_prob)
    return hand_prob

def process_mri_upload(cursor, input_id: int, user_id: int, mri_file, model_id: int):
    mri_path = save_upload(mri_file, f"mri_u{user_id}")
    mri_ext = get_normalized_extension(mri_file.filename)
    mri_mime_value = MRI_MIME_MAP[mri_ext]
    upsert_processed_modality(cursor, input_id, "MRI", mri_path, mri_mime_value)
    (
        mri_prob,
        mri_features,
        processed_mri_path,
        _,
        mri_template_db_path,
        _,
    ) = predict_mri(mri_path, user_id)
    update_modality_inference(
        cursor,
        input_id,
        "MRI",
        mri_features,
        model_id,
        processed_mri_path,
    )
    upsert_modality_prediction(cursor, input_id, model_id, "MRI", mri_prob)
    return mri_prob, processed_mri_path, mri_template_db_path

def load_models() -> dict[str, Any]:
    if MODEL_CACHE:
        return MODEL_CACHE
    hand_cnn = load_model(os.path.join(PROJECT_DIR, "handwriting", "hand_cnn.h5"))
    hand_lgbm = joblib.load(os.path.join(PROJECT_DIR, "handwriting", "hand_lgbm.pkl"))
    mri_cnn = load_model(os.path.join(PROJECT_DIR, "mri", "final_cnn.h5"))
    mri_lgbm = joblib.load(os.path.join(PROJECT_DIR, "mri", "final_lgbm.pkl"))
    mri_mean = np.load(os.path.join(PROJECT_DIR, "mri", "final_mean.npy"))
    mri_std = np.load(os.path.join(PROJECT_DIR, "mri", "final_std.npy"))
    mri_dataset_mean = np.load(MRI_DATASET_MEAN_PATH) if os.path.exists(MRI_DATASET_MEAN_PATH) else None
    mri_dataset_std = np.load(MRI_DATASET_STD_PATH) if os.path.exists(MRI_DATASET_STD_PATH) else None
    voice_model = joblib.load(os.path.join(PROJECT_DIR, "voice", "voice_lgbm.pkl"))

    MODEL_CACHE.update(
        {
            "hand_cnn": hand_cnn,
            "hand_lgbm": hand_lgbm,
            "hand_extractor": Model(inputs=hand_cnn.inputs, outputs=hand_cnn.layers[-3].output),
            "mri_cnn": mri_cnn,
            "mri_lgbm": mri_lgbm,
            "mri_extractor": Model(inputs=mri_cnn.input, outputs=mri_cnn.get_layer("feature_vector").output),
            "mri_mean": mri_mean,
            "mri_std": mri_std,
            "mri_dataset_mean": mri_dataset_mean,
            "mri_dataset_std": mri_dataset_std,
            "voice_model": voice_model,
        }
    )
    return MODEL_CACHE

def load_handwriting_preprocess_fn():
    global _HANDWRITING_PREPROCESS_FN
    if _HANDWRITING_PREPROCESS_FN is not None:
        return _HANDWRITING_PREPROCESS_FN

    preprocess_path = os.path.join(PROJECT_DIR, "handwriting", "preprocess.py")
    if not os.path.exists(preprocess_path):
        _HANDWRITING_PREPROCESS_FN = None
        return None

    spec = importlib.util.spec_from_file_location("handwriting_preprocess_module", preprocess_path)
    if not spec or not spec.loader:
        _HANDWRITING_PREPROCESS_FN = None
        return None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _HANDWRITING_PREPROCESS_FN = getattr(module, "preprocess_image", None)
    return _HANDWRITING_PREPROCESS_FN

def to_float_list(array: np.ndarray, max_len: int = 128) -> list[float]:
    flat = np.asarray(array).flatten().astype(np.float32)
    if flat.size > max_len:
        flat = flat[:max_len]
    return [float(v) for v in flat]

def safe_mean(values: np.ndarray, default: float = 0.0) -> float:
    return float(np.mean(values)) if len(values) else float(default)

def safe_min(values: np.ndarray, default: float = 0.0) -> float:
    return float(np.min(values)) if len(values) else float(default)

def safe_max(values: np.ndarray, default: float = 0.0) -> float:
    return float(np.max(values)) if len(values) else float(default)

def compute_jitter_ddp(periods: np.ndarray) -> float:
    if len(periods) < 3:
        return 0.0
    diff1 = np.diff(periods)
    diff2 = np.diff(diff1)
    denom = np.mean(periods) + 1e-9
    return float(np.mean(np.abs(diff2)) / denom)

def compute_ppe(periods: np.ndarray, bins: int = 10) -> float:
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

def compute_nhr_from_hnr(hnr: float) -> float:
    return float(1.0 / (hnr + 1e-9)) if np.isfinite(hnr) else 0.0

def safe_praat_call(obj: Any, command: str, *args, default: float = 0.0) -> float:
    try:
        value = parselmouth.praat.call(obj, command, *args)
        if value is None or not np.isfinite(value):
            return float(default)
        return float(value)
    except Exception:
        return float(default)

def extract_rqa_value(rqa_obj: Any, key: str, default: float = 0.0) -> float:
    try:
        if isinstance(rqa_obj, dict):
            value = rqa_obj.get(key, default)
            if isinstance(value, (list, tuple, np.ndarray)):
                return float(value[0]) if len(value) else default
            return float(value)

        value = rqa_obj[key]
        if hasattr(value, "iloc"):
            return float(value.iloc[0])
        if isinstance(value, (list, tuple, np.ndarray)):
            return float(value[0]) if len(value) else default
        return float(value)
    except Exception:
        return float(default)

def compute_rqa_measures(signal: np.ndarray) -> tuple[float, float, float]:
    if len(signal) < 64:
        return 0.0, 0.0, 0.0

    try:
        rqa = nk.complexity_rqa(signal, dimension=3, delay=1)
        spread1 = extract_rqa_value(rqa, "Lmax", 0.0)
        spread2 = extract_rqa_value(rqa, "LAM", 0.0)
        d2 = extract_rqa_value(rqa, "DET", 0.0)
        return spread1, spread2, d2
    except Exception:
        return 0.0, 0.0, 0.0

def sanitize_feature_cols(cols: list[str]) -> list[str]:
    return [re.sub(r"[^A-Za-z0-9_]+", "_", str(col)) for col in cols]

def load_voice_normalization_stats(feature_cols: list[str], used_sanitized: bool) -> dict[str, pd.Series]:
    if not os.path.exists(VOICE_DATASET_PATH):
        raise FileNotFoundError(f"Normalization dataset not found at {VOICE_DATASET_PATH}")

    cache_key = (tuple(feature_cols), used_sanitized)
    stats = _NORMALIZATION_CACHE.get(cache_key)
    if stats is not None:
        return stats

    ds = pd.read_csv(VOICE_DATASET_PATH)
    if used_sanitized:
        ds.columns = sanitize_feature_cols(list(ds.columns))

    missing = [col for col in feature_cols if col not in ds.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    ds_numeric = ds[feature_cols].apply(pd.to_numeric, errors="coerce")
    means = ds_numeric.mean()
    stds = ds_numeric.std(ddof=0).replace(0, 1.0)

    stats = {"means": means, "stds": stds}
    _NORMALIZATION_CACHE[cache_key] = stats
    return stats

def extract_voice_feature_map(audio_path: str) -> dict[str, float]:
    sound = parselmouth.Sound(audio_path)
    samples = sound.values.T.flatten().astype(np.float64)

    max_abs = np.max(np.abs(samples)) if samples.size else 0.0
    if max_abs > 0:
        samples = samples / max_abs
    if samples.size >= 2:
        samples = np.diff(samples)

    pitch = sound.to_pitch(pitch_floor=75, pitch_ceiling=500)
    f0 = pitch.selected_array["frequency"]
    f0 = f0[f0 > 0]
    if len(f0) == 0:
        raise ValueError("Voice file has no voiced pitch frames")

    f0_mean = safe_mean(f0)
    f0_min = safe_min(f0)
    f0_max = safe_max(f0)

    periods = 1.0 / (f0 + 1e-9)
    pp = parselmouth.praat.call(sound, "To PointProcess (periodic, cc)", 75, 500)

    jitter_local = safe_praat_call(
        pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3
    )
    jitter_rap = safe_praat_call(
        pp, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3
    )
    jitter_ppq = safe_praat_call(
        pp, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3
    )

    jitter_abs = safe_praat_call(
        pp,
        "Get jitter (local, absolute)",
        0,
        0,
        0.0001,
        0.02,
        1.3,
    )

    jitter_ddp = compute_jitter_ddp(periods)

    shimmer_local = safe_praat_call(
        [sound, pp],
        "Get shimmer (local)",
        0,
        0,
        0.0001,
        0.02,
        1.3,
        1.6,
    )
    shimmer_db = safe_praat_call(
        [sound, pp],
        "Get shimmer (local_dB)",
        0,
        0,
        0.0001,
        0.02,
        1.3,
        1.6,
    )
    shimmer_apq3 = safe_praat_call(
        [sound, pp],
        "Get shimmer (apq3)",
        0,
        0,
        0.0001,
        0.02,
        1.3,
        1.6,
    )
    shimmer_apq5 = safe_praat_call(
        [sound, pp],
        "Get shimmer (apq5)",
        0,
        0,
        0.0001,
        0.02,
        1.3,
        1.6,
    )
    shimmer_apq = safe_praat_call(
        [sound, pp],
        "Get shimmer (apq11)",
        0,
        0,
        0.0001,
        0.02,
        1.3,
        1.6,
    )
    shimmer_dda = safe_praat_call(
        [sound, pp],
        "Get shimmer (dda)",
        0,
        0,
        0.0001,
        0.02,
        1.3,
        1.6,
    )

    harmonicity = sound.to_harmonicity_cc()
    hnr = safe_praat_call(harmonicity, "Get mean", 0, 0)
    nhr = compute_nhr_from_hnr(hnr)

    try:
        rpde_val, _ = rpde(samples, tau=30, dim=4, epsilon=0.01)
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
        "NHR": nhr,
        "HNR": hnr,
        "RPDE": rpde_val,
        "DFA": dfa_val,
        "spread1": spread1,
        "spread2": spread2,
        "D2": d2,
        "PPE": ppe_val,
    }

def predict_voice(audio_path: str) -> tuple[float, list[float], dict[str, float]]:
    models = load_models()
    feature_map = extract_voice_feature_map(audio_path)
    model = models["voice_model"]

    df = pd.DataFrame([feature_map])
    if hasattr(model, "feature_names_in_"):
        feature_cols = list(model.feature_names_in_)
    else:
        feature_cols = list(df.columns)

    missing = [col for col in feature_cols if col not in df.columns]
    used_sanitized = False
    if missing:
        df_sanitized = df.copy()
        df_sanitized.columns = sanitize_feature_cols(list(df_sanitized.columns))
        missing_sanitized = [col for col in feature_cols if col not in df_sanitized.columns]
        if not missing_sanitized:
            df = df_sanitized
            used_sanitized = True
            missing = []

    if missing:
        raise ValueError(f"Missing required feature columns for model: {missing}")

    x_data = df[feature_cols].copy()

    stats = load_voice_normalization_stats(feature_cols, used_sanitized)
    x_data = (x_data - stats["means"]) / stats["stds"]

    model_to_use = model
    if hasattr(model, "steps"):
        scaler = getattr(model, "named_steps", {}).get("scaler")
        if scaler is not None and hasattr(scaler, "mean_") and hasattr(scaler, "scale_"):
            if np.max(np.abs(scaler.mean_)) < 0.1 and np.max(np.abs(scaler.scale_ - 1.0)) < 0.1:
                model_to_use = model.named_steps.get("knn") or model

    x_input = x_data if hasattr(model_to_use, "feature_names_in_") else x_data.to_numpy()
    prob = float(model_to_use.predict_proba(x_input)[:, 1][0])

    if used_sanitized:
        sanitized_map = {
            re.sub(r"[^A-Za-z0-9_]+", "_", str(key)): value
            for key, value in feature_map.items()
        }
        ordered_features = [float(sanitized_map[col]) for col in feature_cols]
        ordered_feature_map = {col: float(sanitized_map[col]) for col in feature_cols}
    else:
        ordered_features = [float(feature_map[col]) for col in feature_cols]
        ordered_feature_map = {col: float(feature_map[col]) for col in feature_cols}

    return prob, ordered_features, ordered_feature_map

def predict_handwriting(image_path: str) -> tuple[float, list[float]]:
    models = load_models()

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Unable to read handwriting image")

    preprocess_fn = load_handwriting_preprocess_fn()
    if preprocess_fn:
        try:
            preprocessed = preprocess_fn(image, IMG_SIZE=128)
            if preprocessed.ndim == 2:
                preprocessed = np.stack([preprocessed, preprocessed, preprocessed], axis=-1)
            image = preprocessed.astype(np.float32)
        except Exception as exc:  # noqa: BLE001
            app.logger.warning("Handwriting preprocessing failed, using fallback resize: %s", exc)
            image = cv2.resize(image, (128, 128)).astype(np.float32) / 255.0
    else:
        image = cv2.resize(image, (128, 128)).astype(np.float32) / 255.0

    image = np.expand_dims(image, axis=0)

    features = models["hand_extractor"].predict(image, verbose=0)
    prob = float(models["hand_lgbm"].predict_proba(features)[:, 1][0])
    return prob, to_float_list(features)

def force_axial_orientation(img: np.ndarray) -> np.ndarray:
    """Rotate tall images to a consistent axial-equivalent layout."""
    h, w = img.shape
    if h > w:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return img

def extract_mri_slice(mri_path: str) -> np.ndarray:
    lowered = mri_path.lower()
    if lowered.endswith(".nii") or lowered.endswith(".nii.gz"):
        img = nib.load(mri_path).get_fdata()
        mid = img.shape[2] // 2
        slices = img[:, :, max(0, mid - 2) : mid + 3]
        slice_2d = np.mean(slices, axis=2).astype(np.float32)
        if float(np.count_nonzero(slice_2d)) / float(slice_2d.size) < 0.1:
            raise ValueError("MRI image appears empty or too sparse")
        resized = resize(
            slice_2d,
            (128, 128),
            mode="constant",
            anti_aliasing=True,
        )
        return resized.astype(np.float32)

    image = cv2.imread(mri_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Unable to read MRI image")
    image = force_axial_orientation(image)
    slice_2d = image.astype(np.float32)

    if float(np.count_nonzero(slice_2d)) / float(slice_2d.size) < 0.1:
        raise ValueError("MRI image appears empty or too sparse")
    resized = resize(
        slice_2d,
        (128, 128),
        mode="constant",
        anti_aliasing=True,
    )
    return resized.astype(np.float32)

def is_mri_image_path(mri_path: str) -> bool:
    ext = get_normalized_extension(mri_path)
    return ext in {"png", "jpg", "jpeg"}

def apply_mri_dataset_normalization(slice_2d: np.ndarray, models: dict[str, Any]) -> np.ndarray:
    mean = models.get("mri_dataset_mean")
    std = models.get("mri_dataset_std")
    if mean is None or std is None:
        return slice_2d
    return (slice_2d - float(mean)) / (float(std) + 1e-8)

def occlusion_sensitivity(model: Any, image: np.ndarray, patch: int = 12, stride: int = 6) -> np.ndarray:
    base_pred = float(model.predict(image, verbose=0)[0][0])
    heatmap = np.zeros((128, 128), dtype=np.float32)

    for y in range(0, 128 - patch, stride):
        for x in range(0, 128 - patch, stride):
            occluded = image.copy()
            occluded[0, y : y + patch, x : x + patch, 0] = 0
            pred = float(model.predict(occluded, verbose=0)[0][0])
            drop = max(0.0, base_pred - pred)
            heatmap[y : y + patch, x : x + patch] += drop

    if float(np.max(heatmap)) > 0:
        heatmap /= float(np.max(heatmap))

    return heatmap

def deep_region_mask() -> np.ndarray:
    h, w = 128, 128
    cy, cx = h // 2, w // 2
    y_grid, x_grid = np.ogrid[:h, :w]
    ry = int(0.22 * h)
    rx = int(0.28 * w)
    ellipse = (((y_grid - cy) / ry) ** 2 + ((x_grid - cx) / rx) ** 2) <= 1
    return ellipse.astype(np.float32)

def extract_brain_template(slice_2d: np.ndarray) -> np.ndarray:
    norm = slice_2d - np.min(slice_2d)
    norm /= float(np.max(norm) + 1e-8)
    return (norm > 0.25).astype(np.float32)

def section_brain_pd(mask: np.ndarray) -> dict[str, np.ndarray]:
    h, w = mask.shape
    cy, cx = h // 2, w // 2
    y_grid, x_grid = np.ogrid[:h, :w]
    ry = int(0.22 * h)
    rx = int(0.28 * w)

    ellipse = (((y_grid - cy) / ry) ** 2 + ((x_grid - cx) / rx) ** 2) <= 1
    deep_region = ellipse.astype(np.float32) * mask
    cortical_region = (mask - deep_region).clip(0, 1)
    mid_w = w // 2

    regions = {
        "Left-Deep": np.zeros_like(mask),
        "Right-Deep": np.zeros_like(mask),
        "Left-Cortical": np.zeros_like(mask),
        "Right-Cortical": np.zeros_like(mask),
    }
    regions["Left-Deep"][:, :mid_w] = deep_region[:, :mid_w]
    regions["Right-Deep"][:, mid_w:] = deep_region[:, mid_w:]
    regions["Left-Cortical"][:, :mid_w] = cortical_region[:, :mid_w]
    regions["Right-Cortical"][:, mid_w:] = cortical_region[:, mid_w:]
    return regions

def save_mri_visuals(slice_2d: np.ndarray, heatmap: np.ndarray, prefix: str) -> tuple[str, str, str, str]:
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S%f")
    occlusion_filename = f"{prefix}_{timestamp}_mri_occlusion.png"
    template_filename = f"{prefix}_{timestamp}_mri_template.png"
    occlusion_abs_path = os.path.join(app.config["UPLOAD_FOLDER"], occlusion_filename)
    template_abs_path = os.path.join(app.config["UPLOAD_FOLDER"], template_filename)
    plt.figure(figsize=(6, 6))
    plt.imshow(slice_2d, cmap="gray")
    occlusion_overlay = np.zeros((128, 128, 4))
    occlusion_overlay[heatmap > np.percentile(heatmap, 85)] = (1, 0, 0, 0.5)
    plt.imshow(occlusion_overlay)
    plt.title("Occlusion Sensitivity (Top 15%)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(occlusion_abs_path, dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.close()
    brain_template = extract_brain_template(slice_2d)
    regions = section_brain_pd(brain_template)
    plt.figure(figsize=(6, 6))
    plt.imshow(slice_2d, cmap="gray")
    template_overlay = np.zeros((128, 128, 4))
    template_overlay[regions["Left-Cortical"] > 0] = (0, 0, 1, 0.35)
    template_overlay[regions["Right-Cortical"] > 0] = (1, 1, 0, 0.35)
    template_overlay[regions["Left-Deep"] > 0] = (1, 0, 0, 0.45)
    template_overlay[regions["Right-Deep"] > 0] = (0, 1, 0, 0.45)
    plt.imshow(template_overlay)
    legend_elements = [
        Patch(facecolor="blue", edgecolor="blue", label="Left Cortical"),
        Patch(facecolor="yellow", edgecolor="yellow", label="Right Cortical"),
        Patch(facecolor="red", edgecolor="red", label="Left Deep"),
        Patch(facecolor="green", edgecolor="green", label="Right Deep"),
    ]
    plt.legend(handles=legend_elements, loc="lower center", bbox_to_anchor=(0.5, -0.15), ncol=2)
    plt.title("PD-Oriented Partitioning")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(template_abs_path, dpi=150, bbox_inches="tight", pad_inches=0.1)
    plt.close()

    occlusion_db_path = f"static/uploads/{occlusion_filename}"
    template_db_path = f"static/uploads/{template_filename}"
    occlusion_static_rel = f"uploads/{occlusion_filename}"
    template_static_rel = f"uploads/{template_filename}"
    return occlusion_db_path, occlusion_static_rel, template_db_path, template_static_rel

def predict_mri(mri_path: str, user_id: int) -> tuple[float, list[float], str, str, str, str]:
    models = load_models()

    slice_2d = extract_mri_slice(mri_path)
    if is_mri_image_path(mri_path):
        slice_2d = apply_mri_dataset_normalization(slice_2d, models)
    x_data = slice_2d[np.newaxis, ..., np.newaxis]
    x_data = (x_data - models["mri_mean"]) / (models["mri_std"] + 1e-8)

    features = models["mri_extractor"].predict(x_data, verbose=0)
    base_prob = float(models["mri_lgbm"].predict_proba(features)[:, 1][0])

    heatmap = occlusion_sensitivity(models["mri_cnn"], x_data)
    deep_mask = deep_region_mask() > 0
    influence_threshold = float(np.percentile(heatmap, OCCLUSION_INFLUENCE_PERCENTILE))
    influenced = heatmap > influence_threshold
    if int(np.sum(influenced)) == 0:
        influenced = heatmap > 0

    influenced_count = int(np.sum(influenced))
    if influenced_count > 0:
        deep_influenced_count = int(np.sum(influenced & deep_mask))
        deep_influence_percent = (float(deep_influenced_count) / float(influenced_count)) * 100.0
    else:
        deep_influence_percent = 0.0

    mri_prob = base_prob
    if deep_influence_percent < DEEP_REGION_MIN_AFFECTED_PERCENT:
        penalty = deep_influence_percent * DEEP_REGION_PENALTY_FACTOR
        mri_prob = float(max(0.0, min(1.0, base_prob - penalty)))

    occlusion_db_path, occlusion_static_rel, template_db_path, template_static_rel = save_mri_visuals(
        slice_2d,
        heatmap,
        f"mri_u{user_id}",
    )
    return (
        mri_prob,
        to_float_list(features),
        occlusion_db_path,
        occlusion_static_rel,
        template_db_path,
        template_static_rel,
    )

def get_or_create_processed_input(cursor, user_id: int) -> int:
    cursor.execute("SELECT input_id FROM processed_input WHERE user_id = %s", (user_id,))
    row = cursor.fetchone()
    if row:
        return row["input_id"]

    cursor.execute(
        """
        INSERT INTO processed_input (user_id, fused_feature_vector)
        VALUES (%s, %s)
        """,
        (user_id, compact_json({})),
    )
    return int(cursor.lastrowid)

def get_or_create_model(cursor, model_name: str, model_type: str) -> int:
    cursor.execute(
        """
        SELECT model_id FROM model
        WHERE model_name = %s AND model_type = %s
        ORDER BY model_id DESC LIMIT 1
        """,
        (model_name, model_type),
    )
    row = cursor.fetchone()
    if row:
        return int(row["model_id"])

    cursor.execute(
        """
        INSERT INTO model (model_name, model_type, model_status, trained_date)
        VALUES (%s, %s, 'ACTIVE', CURDATE())
        """,
        (model_name, model_type),
    )
    return int(cursor.lastrowid)

def upsert_processed_modality(cursor, input_id: int, modality: str, raw_path: str, mime_value: str):
    cfg = MODALITY_DB_CONFIG[modality]
    cursor.execute(f"SELECT {cfg['pk']} FROM {cfg['table']} WHERE input_id = %s", (input_id,))
    if cursor.fetchone():
        update_parts = [f"{cfg['raw_path_col']} = %s"]
        params: list[Any] = [raw_path]
        if cfg["mime_col"]:
            update_parts.append(f"{cfg['mime_col']} = %s")
            params.append(mime_value)
        if cfg["processed_path_col"]:
            update_parts.append(f"{cfg['processed_path_col']} = NULL")
        update_parts.extend(
            [
                "feature_vector = %s",
                "model_id = NULL",
                "processed_at = CURRENT_TIMESTAMP",
            ]
        )
        params.append(compact_json([]))
        params.append(input_id)
        cursor.execute(
            f"UPDATE {cfg['table']} SET {', '.join(update_parts)} WHERE input_id = %s",
            tuple(params),
        )
        return

    columns = ["input_id", cfg["raw_path_col"]]
    values = ["%s", "%s"]
    params = [input_id, raw_path]
    if cfg["mime_col"]:
        columns.append(cfg["mime_col"])
        values.append("%s")
        params.append(mime_value)
    if cfg["processed_path_col"]:
        columns.append(cfg["processed_path_col"])
        values.append("NULL")
    columns.extend(["feature_vector", "model_id"])
    values.extend(["%s", "NULL"])
    params.append(compact_json([]))
    cursor.execute(
        f"INSERT INTO {cfg['table']} ({', '.join(columns)}) VALUES ({', '.join(values)})",
        tuple(params),
    )

def update_modality_inference(
    cursor,
    input_id: int,
    modality: str,
    feature_vector: list[float],
    model_id: int,
    processed_path: str | None = None,
):
    cfg = MODALITY_DB_CONFIG[modality]
    update_parts = []
    params: list[Any] = []
    if cfg["processed_path_col"] and processed_path is not None:
        update_parts.append(f"{cfg['processed_path_col']} = %s")
        params.append(processed_path)
    update_parts.extend(
        [
            "feature_vector = %s",
            "model_id = %s",
            "processed_at = CURRENT_TIMESTAMP",
        ]
    )
    params.extend([compact_json(feature_vector), model_id, input_id])
    cursor.execute(
        f"UPDATE {cfg['table']} SET {', '.join(update_parts)} WHERE input_id = %s",
        tuple(params),
    )

def upsert_modality_prediction(cursor, input_id: int, model_id: int, modality_type: str, probability: float):
    label = "PD" if probability >= 0.5 else "NON_PD"
    cursor.execute(
        """
        INSERT INTO modality_prediction (input_id, model_id, modality_type, predicted_label, confidence_score)
        VALUES (%s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            model_id = VALUES(model_id),
            predicted_label = VALUES(predicted_label),
            confidence_score = VALUES(confidence_score),
            predicted_at = CURRENT_TIMESTAMP
        """,
        (input_id, model_id, modality_type, label, round(probability, 4)),
    )

def clear_missing_modality_predictions(cursor, input_id: int, present_modalities: set[str]):
    all_modalities = {"VOICE", "HANDWRITING", "MRI"}
    missing = all_modalities - present_modalities
    if not missing:
        return

    placeholders = ", ".join(["%s"] * len(missing))
    cursor.execute(
        f"""
        DELETE FROM modality_prediction
        WHERE input_id = %s
          AND modality_type IN ({placeholders})
        """,
        (input_id, *sorted(missing)),
    )

def upsert_prediction(cursor, user_id: int, input_id: int, model_id: int, probability: float):
    label = "PD" if probability >= 0.5 else "NON_PD"
    cursor.execute(
        """
        INSERT INTO prediction (user_id, input_id, model_id, prediction_result, confidence_score)
        VALUES (%s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            model_id = VALUES(model_id),
            prediction_result = VALUES(prediction_result),
            confidence_score = VALUES(confidence_score),
            predicted_at = CURRENT_TIMESTAMP
        """,
        (user_id, input_id, model_id, label, round(probability, 4)),
    )

def get_prediction_id(cursor, user_id: int, input_id: int) -> int:
    cursor.execute(
        """
        SELECT prediction_id
        FROM prediction
        WHERE user_id = %s AND input_id = %s
        LIMIT 1
        """,
        (user_id, input_id),
    )
    row = cursor.fetchone()
    if not row:
        raise RuntimeError("Prediction row missing after upsert")
    return int(row["prediction_id"])

def upsert_report(
    cursor,
    user_id: int,
    prediction_id: int,
    final_label: str,
    final_prob: float,
    voice_prob: float | None,
    hand_prob: float | None,
    mri_prob: float | None,
    mri_occlusion_path: str | None,
    mri_template_path: str | None,
    voice_feature_map: dict[str, float],
):
    modalities_payload: dict[str, Any] = {}
    summary_parts: list[str] = [f"Final: {final_label} ({final_prob * 100:.2f}%)."]

    if voice_prob is not None:
        modalities_payload["voice"] = {"label": "PD" if voice_prob >= 0.5 else "NON_PD", "confidence": round(voice_prob, 4)}
        summary_parts.append(f"Voice: {voice_prob * 100:.2f}%")
    if hand_prob is not None:
        modalities_payload["handwriting"] = {
            "label": "PD" if hand_prob >= 0.5 else "NON_PD",
            "confidence": round(hand_prob, 4),
        }
        summary_parts.append(f"Handwriting: {hand_prob * 100:.2f}%")
    if mri_prob is not None:
        modalities_payload["mri"] = {"label": "PD" if mri_prob >= 0.5 else "NON_PD", "confidence": round(mri_prob, 4)}
        summary_parts.append(f"MRI: {mri_prob * 100:.2f}%")

    report_summary = " ".join(summary_parts)
    report_payload = {
        "modalities": modalities_payload,
        "voice_features": voice_feature_map,
        "mri_images": {
            "occlusion_path": mri_occlusion_path,
            "occlusion_caption": "MRI Occlusion Sensitivity (Top 15% influential regions in red)",
            "template_path": mri_template_path,
            "template_caption": "PD-oriented template partition (Blue: Left cortical, Yellow: Right cortical, Red: Left deep, Green: Right deep)",
        },
        "ensemble": {"label": final_label, "confidence": round(final_prob, 4)},
    }

    cursor.execute(
        """
        INSERT INTO report (user_id, prediction_id, report_summary, report_data)
        VALUES (%s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            prediction_id = VALUES(prediction_id),
            report_summary = VALUES(report_summary),
            report_data = VALUES(report_data),
            report_date = CURRENT_TIMESTAMP
        """,
        (user_id, prediction_id, report_summary, compact_json(report_payload)),
    )

def get_user_profile(cursor, user_id: int) -> dict[str, Any]:
    cursor.execute(
        """
        SELECT user_id, name, email, age
        FROM `user`
        WHERE user_id = %s
        LIMIT 1
        """,
        (user_id,),
    )
    row = cursor.fetchone()
    if not row:
        raise RuntimeError("User row missing while generating report")
    return row

def db_static_to_fs(path_value: str | None) -> str | None:
    if not path_value:
        return None
    cleaned = path_value.replace("\\", "/")
    if cleaned.startswith("static/"):
        cleaned = cleaned[7:]
    return os.path.join(BASE_DIR, "static", cleaned.replace("/", os.sep))

def generate_report_pdf(
    user_data: dict[str, Any],
    input_id: int,
    prediction_id: int,
    final_label: str,
    final_prob: float,
    voice_prob: float | None,
    hand_prob: float | None,
    mri_prob: float | None,
    voice_feature_map: dict[str, float],
    occlusion_db_path: str | None,
    template_db_path: str | None,
) -> str:
    reports_dir = os.path.join(BASE_DIR, "static", "reports")
    os.makedirs(reports_dir, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S%f")
    pdf_filename = f"report_u{user_data['user_id']}_{timestamp}.pdf"
    abs_pdf_path = os.path.join(reports_dir, pdf_filename)

    c = canvas.Canvas(abs_pdf_path, pagesize=A4)
    page_w, page_h = A4
    y = page_h - 40

    c.setFont("Helvetica-Bold", 15)
    c.drawString(40, y, "Parkinson's Prediction Report")
    y -= 28

    c.setFont("Helvetica", 10)
    c.drawString(40, y, f"User ID: {user_data['user_id']}    Name: {user_data['name']}    Age: {user_data['age']}")
    y -= 16
    c.drawString(40, y, f"Email: {user_data['email']}")
    y -= 22
    c.drawString(40, y, f"Input ID: {input_id}    Prediction ID: {prediction_id}")
    y -= 20

    c.setFont("Helvetica-Bold", 11)
    c.drawString(40, y, "Model Outputs")
    y -= 14
    c.setFont("Helvetica", 10)
    if voice_prob is not None:
        c.drawString(40, y, f"Voice: {'PD' if voice_prob >= 0.5 else 'NON_PD'} ({voice_prob:.4f})")
        y -= 14
    if hand_prob is not None:
        c.drawString(40, y, f"Handwriting: {'PD' if hand_prob >= 0.5 else 'NON_PD'} ({hand_prob:.4f})")
        y -= 14
    if mri_prob is not None:
        c.drawString(40, y, f"MRI: {'PD' if mri_prob >= 0.5 else 'NON_PD'} ({mri_prob:.4f})")
        y -= 14
    c.drawString(40, y, f"Final Ensemble: {final_label} ({final_prob:.4f})")
    y -= 24

    if voice_prob is not None:
        c.setFont("Helvetica-Bold", 11)
        c.drawString(40, y, "Voice Feature Values")
        y -= 14
        c.setFont("Helvetica", 9)
        if voice_feature_map:
            for key in sorted(voice_feature_map.keys()):
                if y < 60:
                    c.showPage()
                    y = page_h - 40
                    c.setFont("Helvetica", 9)
                c.drawString(40, y, f"{key}: {voice_feature_map[key]:.6f}")
                y -= 12
        else:
            c.drawString(40, y, "No voice feature values available.")
            y -= 14
        y -= 8

    occlusion_fs = db_static_to_fs(occlusion_db_path)
    template_fs = db_static_to_fs(template_db_path)

    if occlusion_fs and os.path.exists(occlusion_fs):
        c.setFont("Helvetica-Bold", 10)
        c.drawString(40, y, "MRI Occlusion Sensitivity")
        y -= 8
        c.drawImage(ImageReader(occlusion_fs), 40, y - 180, width=240, height=180, preserveAspectRatio=True, mask="auto")
        y -= 194
        c.setFont("Helvetica", 9)
        c.drawString(40, y, "Caption: Top 15% influential regions highlighted in red.")
        y -= 16

    if template_fs and os.path.exists(template_fs):
        if y < 230:
            c.showPage()
            y = page_h - 40
        c.setFont("Helvetica-Bold", 10)
        c.drawString(40, y, "MRI PD-Oriented Template")
        y -= 8
        c.drawImage(ImageReader(template_fs), 40, y - 180, width=240, height=180, preserveAspectRatio=True, mask="auto")
        y -= 194
        c.setFont("Helvetica", 9)
        c.drawString(40, y, "Caption: Blue Left cortical, Yellow Right cortical, Red Left deep, Green Right deep.")
        y -= 16

    if y < 80:
        c.showPage()
        y = page_h - 40
    c.setFont("Helvetica-Bold", 11)
    if final_label == "PD":
        c.drawString(40, y, "Final Note: You are probably having Parkinson's.")
    else:
        c.drawString(40, y, "Final Note: You are probably healthy.")

    c.save()
    return f"static/reports/{pdf_filename}"

def update_report_pdf_path(cursor, user_id: int, prediction_id: int, report_pdf_path: str):
    cursor.execute(
        """
        UPDATE report
        SET prediction_id = %s,
            report_pdf_path = %s,
            report_date = CURRENT_TIMESTAMP
        WHERE user_id = %s
        """,
        (prediction_id, report_pdf_path, user_id),
    )

def update_fused_features(cursor, input_id: int, payload: dict[str, Any]):
    cursor.execute(
        """
        UPDATE processed_input
        SET fused_feature_vector = %s,
            processed_at = CURRENT_TIMESTAMP
        WHERE input_id = %s
        """,
        (compact_json(payload), input_id),
    )

def get_upload_status(user_id: int) -> dict[str, Any]:
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT input_id FROM processed_input WHERE user_id = %s", (user_id,))
        row = cursor.fetchone()
        if not row:
            return {"has_input": False}

        input_id = row["input_id"]
        cursor.execute("SELECT voice_id FROM processed_voice WHERE input_id = %s", (input_id,))
        has_voice = cursor.fetchone() is not None
        cursor.execute("SELECT mri_id FROM processed_mri WHERE input_id = %s", (input_id,))
        has_mri = cursor.fetchone() is not None
        cursor.execute("SELECT handwriting_id FROM processed_handwriting WHERE input_id = %s", (input_id,))
        has_handwriting = cursor.fetchone() is not None

        return {
            "has_input": True,
            "input_id": input_id,
            "has_voice": has_voice,
            "has_mri": has_mri,
            "has_handwriting": has_handwriting,
        }
    except Error:
        return {"has_input": False}
    finally:
        close_db(cursor, conn)

def get_latest_prediction(user_id: int) -> dict[str, Any] | None:
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            """
            SELECT pi.input_id,
                   p.prediction_result,
                   p.confidence_score,
                   p.predicted_at,
                   pm.processed_mri_path,
                   r.report_data
            FROM processed_input pi
            LEFT JOIN prediction p ON p.input_id = pi.input_id
            LEFT JOIN processed_mri pm ON pm.input_id = pi.input_id
            LEFT JOIN report r ON r.user_id = pi.user_id
            WHERE pi.user_id = %s
            """,
            (user_id,),
        )
        row = cursor.fetchone()
        if not row or row["prediction_result"] is None:
            return None

        input_id = row["input_id"]
        cursor.execute(
            """
            SELECT modality_type, predicted_label, confidence_score
            FROM modality_prediction
            WHERE input_id = %s
            """,
            (input_id,),
        )
        modalities = {m["modality_type"]: m for m in cursor.fetchall()}

        mri_visual_rel = None
        if row.get("processed_mri_path"):
            mri_visual_rel = row["processed_mri_path"].replace("\\", "/")
            if mri_visual_rel.startswith("static/"):
                mri_visual_rel = mri_visual_rel[7:]

        report_data = {}
        if row.get("report_data"):
            try:
                report_data = json.loads(row["report_data"])
            except json.JSONDecodeError:
                report_data = {}
        report_modalities = report_data.get("modalities", {})

        voice_entry = report_modalities.get("voice")
        hand_entry = report_modalities.get("handwriting")
        mri_entry = report_modalities.get("mri")
        has_mri_in_report = mri_entry is not None

        return {
            "input_id": input_id,
            "final_label": row["prediction_result"],
            "final_confidence": float(row["confidence_score"]),
            "predicted_at": row["predicted_at"],
            "voice_label": (voice_entry or modalities.get("VOICE", {})).get("label")
            or modalities.get("VOICE", {}).get("predicted_label"),
            "voice_prob": float(
                (voice_entry or {}).get("confidence", modalities.get("VOICE", {}).get("confidence_score", 0.0))
            ),
            "hand_label": (hand_entry or modalities.get("HANDWRITING", {})).get("label")
            or modalities.get("HANDWRITING", {}).get("predicted_label"),
            "hand_prob": float(
                (hand_entry or {}).get("confidence", modalities.get("HANDWRITING", {}).get("confidence_score", 0.0))
            ),
            "mri_label": (mri_entry or modalities.get("MRI", {})).get("label")
            or modalities.get("MRI", {}).get("predicted_label"),
            "mri_prob": float((mri_entry or {}).get("confidence", modalities.get("MRI", {}).get("confidence_score", 0.0)),
            ),
            "mri_visual_rel": mri_visual_rel if has_mri_in_report else None,
        }
    except Error:
        return None
    finally:
        close_db(cursor, conn)

def db_static_to_rel(path_value: str | None) -> str | None:
    if not path_value:
        return None
    cleaned = path_value.replace("\\", "/")
    if cleaned.startswith("static/"):
        return cleaned[7:]
    return cleaned

def get_report_data(user_id: int) -> dict[str, Any] | None:
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            """
            SELECT u.user_id,
                   u.name,
                   u.email,
                   u.age,
                   pi.input_id,
                   p.prediction_id,
                   p.prediction_result,
                   p.confidence_score AS final_confidence,
                   p.predicted_at,
                   r.report_summary,
                   r.report_data,
                   r.report_pdf_path,
                   pm.processed_mri_path
            FROM user u
            JOIN processed_input pi ON pi.user_id = u.user_id
            JOIN prediction p ON p.input_id = pi.input_id
            LEFT JOIN report r ON r.user_id = u.user_id
            LEFT JOIN processed_mri pm ON pm.input_id = pi.input_id
            WHERE u.user_id = %s
            LIMIT 1
            """,
            (user_id,),
        )
        row = cursor.fetchone()
        if not row:
            return None

        cursor.execute(
            """
            SELECT modality_type, predicted_label, confidence_score
            FROM modality_prediction
            WHERE input_id = %s
            """,
            (row["input_id"],),
        )
        modalities = {item["modality_type"]: item for item in cursor.fetchall()}

        report_data = {}
        if row.get("report_data"):
            try:
                report_data = json.loads(row["report_data"])
            except json.JSONDecodeError:
                report_data = {}

        occlusion_rel = db_static_to_rel(report_data.get("mri_images", {}).get("occlusion_path"))
        template_rel = db_static_to_rel(report_data.get("mri_images", {}).get("template_path"))

        if not occlusion_rel:
            occlusion_rel = db_static_to_rel(row.get("processed_mri_path"))

        report_modalities = report_data.get("modalities", {})
        voice_from_report = report_modalities.get("voice")
        hand_from_report = report_modalities.get("handwriting")
        mri_from_report = report_modalities.get("mri")

        voice_label = (voice_from_report or {}).get("label") or modalities.get("VOICE", {}).get("predicted_label")
        voice_conf = (voice_from_report or {}).get("confidence", modalities.get("VOICE", {}).get("confidence_score"))
        hand_label = (hand_from_report or {}).get("label") or modalities.get("HANDWRITING", {}).get("predicted_label")
        hand_conf = (hand_from_report or {}).get("confidence", modalities.get("HANDWRITING", {}).get("confidence_score"))
        mri_label = (mri_from_report or {}).get("label") or modalities.get("MRI", {}).get("predicted_label")
        mri_conf = (mri_from_report or {}).get("confidence", modalities.get("MRI", {}).get("confidence_score"))

        return {
            "user": {
                "user_id": row["user_id"],
                "name": row["name"],
                "email": row["email"],
                "age": row["age"],
            },
            "input_id": row["input_id"],
            "prediction_id": row["prediction_id"],
            "predicted_at": row["predicted_at"],
            "final_label": row["prediction_result"],
            "final_confidence": float(row["final_confidence"]),
            "report_summary": row.get("report_summary"),
            "voice_label": voice_label,
            "voice_confidence": float(voice_conf) if voice_conf is not None else None,
            "hand_label": hand_label,
            "hand_confidence": float(hand_conf) if hand_conf is not None else None,
            "mri_label": mri_label,
            "mri_confidence": float(mri_conf) if mri_conf is not None else None,
            "occlusion_rel": occlusion_rel,
            "occlusion_caption": report_data.get("mri_images", {}).get(
                "occlusion_caption",
                "MRI Occlusion Sensitivity (Top 15% influential regions in red)",
            ),
            "template_rel": template_rel,
            "template_caption": report_data.get("mri_images", {}).get(
                "template_caption",
                "PD-oriented template partition (Blue: Left cortical, Yellow: Right cortical, Red: Left deep, Green: Right deep)",
            ),
            "voice_features": report_data.get("voice_features", {}),
            "report_pdf_rel": db_static_to_rel(row.get("report_pdf_path")),
        }
    except Error:
        return None
    finally:
        close_db(cursor, conn)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        name = request.form.get("name", "").strip()
        age = request.form.get("age", "").strip()
        password = request.form.get("password", "").strip()

        if not all([email, name, age, password]):
            flash("All fields are required.", "error")
            return render_template("signup.html")

        try:
            age_value = int(age)
            if age_value < 1 or age_value > 120:
                raise ValueError
        except ValueError:
            flash("Age must be between 1 and 120.", "error")
            return render_template("signup.html")

        conn = None
        cursor = None
        try:
            conn = get_db_connection()
            cursor = conn.cursor(dictionary=True)
            ensure_user_table_compatibility(cursor, conn)
            cursor.execute("SELECT user_id FROM `user` WHERE email = %s", (email,))
            if cursor.fetchone():
                flash("An account with that email already exists.", "error")
                return render_template("signup.html")

            password_hash = generate_password_hash(password)
            cursor.execute(
                """
                INSERT INTO `user` (name, email, age, password_hash)
                VALUES (%s, %s, %s, %s)
                """,
                (name, email, age_value, password_hash),
            )
            conn.commit()
            flash("Account created! Please log in.", "success")
            return redirect(url_for("login"))
        except Error as db_err:
            flash(f"Database error during signup: {db_err}", "error")
            return render_template("signup.html")
        finally:
            close_db(cursor, conn)

    return render_template("signup.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "").strip()

        conn = None
        cursor = None
        try:
            conn = get_db_connection()
            cursor = conn.cursor(dictionary=True)
            cursor.execute(
                "SELECT user_id, name, email, password_hash FROM `user` WHERE email = %s",
                (email,),
            )
            user = cursor.fetchone()
            if user and check_password_hash(user["password_hash"], password):
                session["user_id"] = user["user_id"]
                session["user"] = user["email"]
                session["name"] = user["name"]
                return redirect(url_for("predict"))

            flash("Invalid email or password.", "error")
        except Error as db_err:
            flash(f"Database error during login: {db_err}", "error")
        finally:
            close_db(cursor, conn)

    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))

@app.route("/report")
def report():
    blocked = require_login("Please log in to view the report.", "login")
    if blocked:
        return blocked

    data = get_report_data(session["user_id"])
    if not data:
        flash("No report available yet. Run a prediction first.", "error")
        return redirect(url_for("predict"))

    return render_template("report.html", report=data)

@app.route("/predict", methods=["GET", "POST"])
def predict():
    blocked = require_login("Please log in to access uploads.", "login")
    if blocked:
        return blocked

    if request.method == "POST":
        voice = request.files.get("voice")
        handwriting = request.files.get("handwriting")
        mri = request.files.get("mri")

        errors = validate_uploads(voice, handwriting, mri)

        if errors:
            for message in errors:
                flash(message, "error")
            status = get_upload_status(session["user_id"])
            result = get_latest_prediction(session["user_id"])
            return render_template("predict.html", status=status, result=result)

        conn = None
        cursor = None
        try:
            conn = get_db_connection()
            cursor = conn.cursor(dictionary=True)

            input_id = get_or_create_processed_input(cursor, session["user_id"])

            model_ids, ensemble_model_id = get_model_ids(cursor)

            modality_probs: dict[str, float] = {}
            voice_feature_map: dict[str, float] = {}
            processed_mri_path = None
            mri_template_db_path = None
            hand_prob = None
            mri_prob = None

            if voice and voice.filename:
                voice_prob, voice_feature_map = process_voice_upload(
                    cursor, input_id, session["user_id"], voice, model_ids["VOICE"]
                )
                modality_probs["VOICE"] = voice_prob

            if handwriting and handwriting.filename:
                hand_prob = process_handwriting_upload(
                    cursor, input_id, session["user_id"], handwriting, model_ids["HANDWRITING"]
                )
                modality_probs["HANDWRITING"] = hand_prob

            if mri and mri.filename:
                mri_prob, processed_mri_path, mri_template_db_path = process_mri_upload(
                    cursor, input_id, session["user_id"], mri, model_ids["MRI"]
                )
                modality_probs["MRI"] = mri_prob

            if not modality_probs:
                raise RuntimeError("No valid modality inputs were provided for prediction.")

            clear_missing_modality_predictions(cursor, input_id, set(modality_probs.keys()))
            final_prob = weighted_average(modality_probs)
            decision_threshold = 0.5
            upsert_prediction(cursor, session["user_id"], input_id, ensemble_model_id, final_prob)
            prediction_id = get_prediction_id(cursor, session["user_id"], input_id)
            upsert_report(
                cursor,
                session["user_id"],
                prediction_id,
                "PD" if final_prob >= decision_threshold else "NON_PD",
                final_prob,
                modality_probs.get("VOICE"),
                hand_prob,
                mri_prob,
                processed_mri_path,
                mri_template_db_path,
                voice_feature_map,
            )
            user_profile = get_user_profile(cursor, session["user_id"])
            pdf_path = generate_report_pdf(
                user_profile,
                input_id,
                prediction_id,
                "PD" if final_prob >= decision_threshold else "NON_PD",
                final_prob,
                modality_probs.get("VOICE"),
                hand_prob,
                mri_prob,
                voice_feature_map,
                processed_mri_path,
                mri_template_db_path,
            )
            update_report_pdf_path(cursor, session["user_id"], prediction_id, pdf_path)

            update_fused_features(
                cursor,
                input_id,
                {
                    "modalities_used": sorted(modality_probs.keys()),
                    "voice_probability": round(modality_probs["VOICE"], 4) if "VOICE" in modality_probs else None,
                    "handwriting_probability": round(modality_probs["HANDWRITING"], 4)
                    if "HANDWRITING" in modality_probs
                    else None,
                    "mri_probability": round(modality_probs["MRI"], 4) if "MRI" in modality_probs else None,
                    "ensemble_probability": round(final_prob, 4),
                    "ensemble_threshold": round(decision_threshold, 4),
                    "ensemble_label": "PD" if final_prob >= decision_threshold else "NON_PD",
                },
            )

            conn.commit()
            flash("Prediction completed with uploaded modalities. Report PDF generated.", "success")
        except Exception as exc:  # noqa: BLE001
            if conn:
                conn.rollback()
            app.logger.exception("Prediction failed.")
            print("[PREDICT ERROR]", repr(exc))
            flash("Prediction failed due to an internal error. Please try again.", "error")
        finally:
            close_db(cursor, conn)

        return redirect(url_for("predict"))

    status = get_upload_status(session["user_id"])
    result = get_latest_prediction(session["user_id"])
    return render_template("predict.html", status=status, result=result)

if __name__ == "__main__":
    app.run(debug=True)














