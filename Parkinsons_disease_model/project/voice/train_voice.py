import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import shap
import numpy as np
import matplotlib.pyplot as plt
import joblib

df = pd.read_csv("voice_normalized.csv")

X = df.drop(columns=["name", "status"])
y = df["status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

knn = KNeighborsClassifier(n_neighbors=2, weights="distance")
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print("KNN WITH class weights (distance) accuracy:",
      accuracy_score(y_test, y_pred))

# Create explainer
explainer = shap.KernelExplainer(knn.predict_proba, X_train[:50])

# Compute SHAP values (use small sample for speed)
shap_values = explainer(X_test[:100])

# If new SHAP version (3D output)
if len(shap_values.values.shape) == 3:
    pd_shap = shap_values.values[:, :, 1]   # class 1 = Parkinson's
else:
    pd_shap = shap_values[1]

# Mean absolute SHAP importance
importance = np.abs(pd_shap).mean(axis=0)

# Sort features
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

# Plot bar chart
plt.figure(figsize=(10,6))
plt.barh(feature_importance["Feature"], feature_importance["Importance"])
plt.gca().invert_yaxis()
plt.title("SHAP Feature Importance for Parkinson's Detection (KNN)")
plt.xlabel("Mean |SHAP Value|")
plt.show()
joblib.dump(knn, "voice_knn.pkl")

print("✅ Voice model saved as voice_knn.pkl")