import os
import logging
import random
import numpy as np
import cv2
import joblib
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"   
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  
logging.getLogger('tensorflow').setLevel(logging.FATAL)
random.seed(SEED)
np.random.seed(SEED)
import tensorflow as tf
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D,
    GlobalAveragePooling2D,
    Dense, Dropout, BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import nibabel as nib
from skimage.transform import resize
import shap
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
from scipy.ndimage import label

tf.random.set_seed(SEED)
def load_mri(folder, label, target_size=(128, 128)):
    X, y = [], []
    for file in os.listdir(folder):
        if file.endswith(".nii") or file.endswith(".nii.gz"):
            img = nib.load(os.path.join(folder, file)).get_fdata()
            mid = img.shape[2] // 2
            slices = img[:, :, max(0, mid - 2): mid + 3]
            slice_2d = np.mean(slices, axis=2)
            slice_2d = resize(slice_2d, target_size, mode="constant", anti_aliasing=True)
            if np.count_nonzero(slice_2d) / slice_2d.size < 0.1:
                continue
            X.append(slice_2d)
            y.append(label)
    return X, y
PD_PATH = r"C:\Users\fbash\OneDrive\Desktop\github\mri\nii_preprocessed3\pd"
NONPD_PATH = r"C:\Users\fbash\OneDrive\Desktop\github\mri\nii_preprocessed3\nonpd"
X_pd, y_pd = load_mri(PD_PATH, 1)
X_nonpd, y_nonpd = load_mri(NONPD_PATH, 0)
X = np.array(X_pd + X_nonpd)[..., np.newaxis]
y = np.array(y_pd + y_nonpd)
print("Total samples:", X.shape[0])
def build_cnn(input_shape):
    inputs = Input(shape=input_shape)

    x = Conv2D(8, 3, padding="same", activation="relu")(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D()(x)

    x = Conv2D(16, 3, padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D()(x)

    x = Conv2D(32, 3, padding="same", activation="relu", name="last")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D()(x)

    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)

    features = Dense(64, activation="relu", name="feature_vector")(x)
    output = Dense(1, activation="sigmoid")(features)

    model = Model(inputs, output)
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.05
)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
cv_scores = []
last_train_feat = None
last_val_feat = None
last_svm = None
all_fold_shap = []
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    print(f"\n========== Fold {fold} ==========")

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    class_weight = {
        0: len(y_train) / (2 * np.sum(y_train == 0)),
        1: len(y_train) / (2 * np.sum(y_train == 1))
    }
    mean = np.mean(X_train, axis=(0,1,2), keepdims=True)
    std = np.std(X_train, axis=(0,1,2), keepdims=True) + 1e-7
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    cnn = build_cnn(X_train.shape[1:])
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )
    cnn.fit(
        datagen.flow(X_train, y_train, batch_size=8),
        validation_data=(X_val, y_val),
        epochs=25,
        callbacks=[early_stop],
        class_weight=class_weight,
        verbose=0
    )

    extractor = Model(cnn.input, cnn.get_layer("feature_vector").output)
    train_feat = extractor.predict(X_train, batch_size=32, verbose=0)
    val_feat = extractor.predict(X_val, batch_size=32, verbose=0)
    lgbm = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary",
        random_state=SEED,
        n_jobs=-1
    )
    lgbm.fit(train_feat, y_train)
    preds = lgbm.predict(val_feat)
    # ==========================
# SHAP (STORE PER-FOLD)
# ==========================

    background = train_feat[:20]  # deterministic
    explainer = shap.Explainer(lgbm.predict, background)


    shap_values = explainer(val_feat[:50])  # subset for stability
    fold_importance = np.abs(shap_values.values).mean(axis=0)

    all_fold_shap.append(fold_importance)

    last_train_feat = train_feat
    last_val_feat = val_feat
    last_svm = lgbm   
    last_cnn = cnn
    # ==========================
# SAVE MATCHED MODELS (LAST FOLD ONLY)
# ==========================

    if fold == skf.n_splits:  # only save final fold
        
        # Save CNN
        cnn.save("final_cnn.h5")
        
        # Save LightGBM
        joblib.dump(lgbm, "final_lgbm.pkl")
        
        # Save normalization parameters
        np.save("final_mean.npy", mean)
        np.save("final_std.npy", std)
        
        print("\n✅ Saved matched CNN + LGBM + normalization parameters.")
    acc = accuracy_score(y_val, preds)
    cv_scores.append(acc)

    print("Fold Accuracy:", acc)
    print(classification_report(y_val, preds, digits=4))
print("\n✅ Mean CV Accuracy:", np.mean(cv_scores))
# ==========================
# GLOBAL FEATURE IMPORTANCE
# ==========================

# ==========================
# GLOBAL SHAP FEATURE IMPORTANCE
# ==========================

all_fold_shap = np.array(all_fold_shap)   # (folds, features)
global_importance = all_fold_shap.mean(axis=0)

top10_idx = np.argsort(global_importance)[-10:][::-1]
top10_vals = global_importance[top10_idx]

print("\n📊 Global SHAP Feature Importance (Averaged Across Folds):\n")
for rank, idx in enumerate(top10_idx, 1):
    print(f"{rank}. Feature {idx} → Mean |SHAP| = {global_importance[idx]:.6f}")

plt.figure(figsize=(8,5))
plt.bar(range(10), top10_vals)
plt.xticks(range(10), top10_idx)
plt.xlabel("CNN Feature Index")
plt.ylabel("Mean |SHAP| across folds")
plt.title("Top 10 Globally Important Features (All Folds)")
plt.show()

# ==========================
# SELECT TOP 5 FEATURES
# ==========================

TOP_K = 5
top_features = top10_idx[:TOP_K]
print("🎯 Visualizing top 5 SHAP-selected features:", top_features)

# ==========================
# BRAIN MASK
# ==========================

def simple_brain_mask(img_2d):
    mask = img_2d > 0.1
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask.astype(np.float32)

# ==========================
# FEATURE ACTIVATION MAP
# ==========================

def feature_activation_map(model, image, dense_feature_idx):
    # Get last conv output
    conv_model = Model(model.input, model.get_layer("last").output)
    conv_output = conv_model.predict(image, verbose=0)[0]  # (H, W, 32)

    # Get Dense layer weights
    dense_layer = model.get_layer("feature_vector")
    W, _ = dense_layer.get_weights()  # W shape = (32, 64)

    # Weights for this Dense feature
    weights = W[:, dense_feature_idx]  # (32,)

    # Weighted sum of conv maps
    fmap = np.zeros(conv_output.shape[:2])
    for k in range(conv_output.shape[2]):
        fmap += weights[k] * conv_output[:, :, k]

    fmap = np.maximum(fmap, 0)

    fmap = resize(fmap, image.shape[1:3], mode="constant")
    fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-8)

    return fmap


# ==========================
# APPLY TO ONE MRI
# ==========================

img = X_val[0:1]

img_show = img[0, :, :, 0]
img_show = (img_show - img_show.min()) / (img_show.max() - img_show.min())

brain_mask = simple_brain_mask(img_show)

# Compute activation maps
activation_maps = {}
for f in top_features:
    fmap = feature_activation_map(last_cnn, img, f)
    fmap *= brain_mask
    activation_maps[f] = fmap

# ==========================
# VISUALIZATION WITH LEGEND
# ==========================

from matplotlib.lines import Line2D
import matplotlib.cm as cm

plt.figure(figsize=(6,6))
plt.imshow(img_show, cmap="gray")

cmap = cm.get_cmap("tab10", TOP_K)
legend_elements = []

for i, (feature_idx, fmap) in enumerate(activation_maps.items()):
    color = cmap(i)

    nonzero = fmap[fmap > 0]
    if len(nonzero) == 0:
        continue

    thresh = np.percentile(nonzero, 85)
    mask = fmap >= thresh

    colored = np.zeros((*fmap.shape, 4))
    colored[mask] = (*color[:3], 0.6)

    plt.imshow(colored)

    legend_elements.append(
        Line2D(
            [0], [0],
            marker='s',
            linestyle='',
            markersize=10,
            color=color,
            label=f'Feature {feature_idx}'
        )
    )

plt.legend(
    handles=legend_elements,
    title="CNN Feature Index",
    loc="upper right"
)

plt.axis("off")
plt.title("Top 5 SHAP-Selected CNN Features on MRI\n(Color → Feature Number)")
plt.show()





