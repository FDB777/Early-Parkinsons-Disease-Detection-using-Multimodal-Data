import os
import numpy as np
import pandas as pd
import cv2
import nibabel as nib
import joblib
import tensorflow as tf
from skimage.transform import resize
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print("Loading models...")
voice_model = joblib.load(os.path.join(BASE_DIR, "voice", "voice_knn.pkl"))
hand_cnn = load_model(os.path.join(BASE_DIR, "handwriting", "hand_cnn.h5"))
hand_lgbm = joblib.load(os.path.join(BASE_DIR, "handwriting", "hand_lgbm.pkl"))
mri_cnn = load_model(os.path.join(BASE_DIR, "mri", "final_cnn.h5"))
mri_lgbm = joblib.load(os.path.join(BASE_DIR, "mri", "final_lgbm.pkl"))
mri_mean = np.load(os.path.join(BASE_DIR, "mri", "final_mean.npy"))
mri_std = np.load(os.path.join(BASE_DIR, "mri", "final_std.npy"))

print("✅ All models loaded successfully")

def predict_voice(csv_path):

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Voice file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    X = df.drop(columns=["name", "status"], errors="ignore")
    prob = voice_model.predict_proba(X)[:, 1]
    return prob[0]

IMG_SIZE = 128

def predict_handwriting(image_path):

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    _ = hand_cnn.predict(img, verbose=0)

    feature_extractor = tf.keras.Model(
        inputs=hand_cnn.inputs,
        outputs=hand_cnn.layers[-3].output
    )

    features = feature_extractor.predict(img, verbose=0)
    prob = hand_lgbm.predict_proba(features)[:, 1]
    return prob[0]

def occlusion_sensitivity(model, image, patch=12, stride=6):

    base_pred = model.predict(image, verbose=0)[0][0]

    heatmap = np.zeros((128,128))

    for y in range(0, 128-patch, stride):
        for x in range(0, 128-patch, stride):

            occluded = image.copy()
            occluded[0, y:y+patch, x:x+patch, 0] = 0

            pred = model.predict(occluded, verbose=0)[0][0]

            drop = base_pred - pred
            heatmap[y:y+patch, x:x+patch] += drop

    heatmap = np.maximum(heatmap,0)

    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    return heatmap

def deep_region_mask():

    h, w = 128, 128
    cy, cx = h//2, w//2

    Y, X = np.ogrid[:h, :w]

    ry = int(0.22*h)
    rx = int(0.28*w)

    ellipse = (((Y-cy)/ry)**2 + ((X-cx)/rx)**2) <= 1

    return ellipse.astype(np.float32)

def section_brain_pd(mask):

    h, w = mask.shape
    cy, cx = h//2, w//2

    Y, X = np.ogrid[:h, :w]

    ry = int(0.22*h)
    rx = int(0.28*w)

    ellipse = (((Y-cy)/ry)**2 + ((X-cx)/rx)**2) <= 1

    deep_region = ellipse.astype(np.float32) * mask
    cortical_region = (mask - deep_region).clip(0,1)

    mid_w = w//2

    regions = {}

    regions["Left-Deep"] = np.zeros_like(mask)
    regions["Right-Deep"] = np.zeros_like(mask)
    regions["Left-Cortical"] = np.zeros_like(mask)
    regions["Right-Cortical"] = np.zeros_like(mask)

    regions["Left-Deep"][:, :mid_w] = deep_region[:, :mid_w]
    regions["Right-Deep"][:, mid_w:] = deep_region[:, mid_w:]

    regions["Left-Cortical"][:, :mid_w] = cortical_region[:, :mid_w]
    regions["Right-Cortical"][:, mid_w:] = cortical_region[:, mid_w:]

    return regions

def extract_brain_template(slice_img):

    # normalize slice
    norm = (slice_img - slice_img.min()) / (slice_img.max() - slice_img.min() + 1e-8)

    # simple threshold to isolate brain region
    template = (norm > 0.25).astype(np.float32)

    return template

def predict_mri(nii_path):

    if not os.path.exists(nii_path):
        raise FileNotFoundError(f"MRI file not found: {nii_path}")

    img = nib.load(nii_path).get_fdata()

    mid = img.shape[2] // 2
    slices = img[:, :, max(0, mid - 2): mid + 3]
    slice_2d = np.mean(slices, axis=2)

    slice_2d = resize(slice_2d, (128, 128),
                      mode="constant",
                      anti_aliasing=True)

    X = slice_2d[np.newaxis, ..., np.newaxis]

    X = (X - mri_mean) / (mri_std + 1e-8)

    _ = mri_cnn.predict(X, verbose=0)

    extractor = tf.keras.Model(
        inputs=mri_cnn.input,
        outputs=mri_cnn.get_layer("feature_vector").output
    )

    features = extractor.predict(X, verbose=0)
    base_prob = mri_lgbm.predict_proba(features)[:, 1][0]

    # -----------------------------
    # OCCLUSION SENSITIVITY
    # -----------------------------
    heatmap = occlusion_sensitivity(mri_cnn, X)
    brain_template = extract_brain_template(slice_2d)

    deep_mask = deep_region_mask()
    regions = section_brain_pd(brain_template)
    deep_activation = np.sum(heatmap * deep_mask)
    total_activation = np.sum(heatmap) + 1e-8

    deep_ratio = deep_activation / total_activation

    # -----------------------------
    # WEIGHT REDUCTION
    # -----------------------------
    weighted_prob = base_prob * (1 - 0.3 * deep_ratio)

    print("\nMRI probability BEFORE weighting:", round(base_prob,4))
    print("Deep activation ratio:", round(deep_ratio,4))
    print("MRI probability AFTER weighting:", round(weighted_prob,4))
    print("Deep region reduced probability by:",
        round(base_prob - weighted_prob,4))

    plt.figure(figsize=(6,6))
    plt.imshow(slice_2d, cmap="gray")

    overlay = np.zeros((128,128,4))
    overlay[heatmap > np.percentile(heatmap,85)] = (1,0,0,0.5)

    plt.imshow(overlay)
    plt.title("Occlusion Sensitivity (Top 15%)")
    plt.axis("off")
    plt.show()
    regions = section_brain_pd(brain_template)
    # -----------------------------
    # PD-ORIENTED REGION TEMPLATE
    # -----------------------------
    plt.figure(figsize=(6,6))
    plt.imshow(slice_2d, cmap="gray")

    overlay = np.zeros((128,128,4))

    # Left cortical (blue)
    overlay[regions["Left-Cortical"] > 0] = (0,0,1,0.35)

    # Right cortical (yellow)
    overlay[regions["Right-Cortical"] > 0] = (1,1,0,0.35)

    # Left deep (red)
    overlay[regions["Left-Deep"] > 0] = (1,0,0,0.45)

    # Right deep (green)
    overlay[regions["Right-Deep"] > 0] = (0,1,0,0.45)

    plt.imshow(overlay)

    # -------- Legend --------
    legend_elements = [
        Patch(facecolor='blue', edgecolor='blue', label='Left Cortical'),
        Patch(facecolor='yellow', edgecolor='yellow', label='Right Cortical'),
        Patch(facecolor='red', edgecolor='red', label='Left Deep'),
        Patch(facecolor='green', edgecolor='green', label='Right Deep')
    ]

    plt.legend(handles=legend_elements,
            loc='lower center',
            bbox_to_anchor=(0.5,-0.15),
            ncol=2)

    plt.title("PD-Oriented Partitioning")
    plt.axis("off")
    plt.show()
    return weighted_prob
def multimodal_soft_voting(voice_csv, hand_img, mri_nii):

    print("\nRunning predictions...")

    voice_prob = predict_voice(voice_csv)
    hand_prob = predict_handwriting(hand_img)
    mri_prob = predict_mri(mri_nii)

    final_prob = (voice_prob + hand_prob + mri_prob) / 3
    final_label = 1 if final_prob > 0.5 else 0

    print("\n========== INDIVIDUAL PROBABILITIES ==========")
    print("Voice:", round(voice_prob, 4))
    print("Handwriting:", round(hand_prob, 4))
    print("MRI:", round(mri_prob, 4))

    print("\n========== FINAL ENSEMBLE RESULT ==========")
    print("Final Probability:", round(final_prob, 4))
    print("Prediction:", "Parkinson's" if final_label == 1 else "Healthy")

    return final_prob, final_label

if __name__ == "__main__":

    voice_csv = r"C:\Users\fbash\OneDrive\Desktop\github\project\voice\voice_test.csv"
    hand_img = r"C:\Users\fbash\OneDrive\Desktop\github\handwriting\processed_images\processed_images\SpiralControl\0104-3_aug0.jpg"
    mri_nii = r"C:\Users\fbash\OneDrive\Desktop\github\mri\nii_preprocessed3\pd\FL_C_Ax_T1_013.nii"
    multimodal_soft_voting(voice_csv, hand_img, mri_nii)