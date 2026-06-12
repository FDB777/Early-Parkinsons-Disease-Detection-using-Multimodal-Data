import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier
import joblib

# =====================================
# SETTINGS
# =====================================
IMG_SIZE = 128
DATA_DIR = r"C:\Users\fbash\OneDrive\Desktop\github\handwriting\processed_images\processed_images"
EPOCHS = 10
BATCH_SIZE = 32
N_SPLITS = 5
RANDOM_STATE = 42

# =====================================
# CLASS NAMES
# =====================================
CLASS_NAMES = ["SpiralControl", "SpiralPatients"]

# =====================================
# LOAD DATA
# =====================================
def load_data():
    X = []
    y = []

    for label, class_name in enumerate(CLASS_NAMES):
        class_path = os.path.join(DATA_DIR, class_name)

        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)

            img = cv2.imread(img_path)
            if img is None:
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img / 255.0

            X.append(img)
            y.append(label)

    return np.array(X), np.array(y)

# =====================================
# CNN MODEL
# =====================================
def build_cnn(num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu',
                      input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),

        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# =====================================
# MAIN
# =====================================
print("Loading dataset...")
X, y = load_data()
num_classes = len(CLASS_NAMES)

print("Dataset shape:", X.shape)

skf = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

fold_accuracies = []
fold_number = 1

for train_idx, test_idx in skf.split(X, y):

    print(f"\n========== Fold {fold_number} ==========")

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # ---------------------------
    # Class Weights
    # ---------------------------
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))

    # ---------------------------
    # Train CNN
    # ---------------------------
    cnn_model = build_cnn(num_classes)

    cnn_model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0,
        class_weight=class_weight_dict
    )

    # ---------------------------
    # Feature Extraction
    # ---------------------------
    feature_extractor = tf.keras.Model(
        inputs=cnn_model.inputs,
        outputs=cnn_model.layers[-3].output
    )

    X_train_features = feature_extractor.predict(X_train, verbose=0)
    X_test_features = feature_extractor.predict(X_test, verbose=0)

    # ---------------------------
    # LightGBM Classifier
    # ---------------------------
    lgbm_model = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        class_weight='balanced',
        random_state=RANDOM_STATE
    )

    lgbm_model.fit(X_train_features, y_train)

    # ---------------------------
    # Evaluation
    # ---------------------------
    y_pred = lgbm_model.predict(X_test_features)
    acc = accuracy_score(y_test, y_pred)

    print(f"Fold {fold_number} Accuracy: {acc:.4f}")
    if fold_number == N_SPLITS:
        cnn_model.save("hand_cnn.h5")
        joblib.dump(lgbm_model, "hand_lgbm.pkl")
        print("✅ Saved final handwriting CNN + LGBM")
    fold_accuracies.append(acc)
    fold_number += 1

# =====================================
# FINAL RESULTS
# =====================================
mean_accuracy = np.mean(fold_accuracies)

print("\n===================================")
print("Fold Accuracies:", fold_accuracies)
print("Mean Accuracy:", round(mean_accuracy, 4))
print("===================================")