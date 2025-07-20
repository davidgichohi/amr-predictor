import os
import joblib
from glob import glob
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report
import numpy as np

# === Configuration ===
MODEL_DIR = "models/"  # Replace with your actual model directory
OUTPUT_DIR = MODEL_DIR  # You can change if saving elsewhere
param_grid = {
    'max_depth': [3, 5],
    'n_estimators': [50],
    'learning_rate': [0.1],
}

# === Discover Feature and Label Files ===
X_files = sorted(glob(os.path.join(MODEL_DIR, "*_X.pkl")))
y_files = sorted(glob(os.path.join(MODEL_DIR, "*_y.pkl")))

print("🔁 Starting clean model retraining...")

for X_file, y_file in zip(X_files, y_files):
    base_name = os.path.basename(X_file).replace("_X.pkl", "")
    try:
        X = joblib.load(X_file)
        y = joblib.load(y_file)

        # Determine number of unique classes
        num_classes = len(np.unique(y))
        print(f"\n📦 Retraining {base_name} - Classes: {np.unique(y)}")

        # Define base model
        xgb_base = XGBClassifier(
            objective='multi:softmax' if num_classes > 2 else 'binary:logistic',
            num_class=num_classes if num_classes > 2 else None,
            eval_metric='mlogloss',
            use_label_encoder=False,
            verbosity=0
        )

        # Grid search
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        grid_search = GridSearchCV(xgb_base, param_grid, cv=cv, scoring='f1_weighted', n_jobs=-1)
        grid_search.fit(X, y)

        best_model = grid_search.best_estimator_
        print(f"✅ Best params: {grid_search.best_params_}")

        # Save updated model
        model_out_path = os.path.join(OUTPUT_DIR, f"{base_name}_xgb_model.pkl")
        joblib.dump(best_model, model_out_path)
        print(f"💾 Saved: {model_out_path}")

    except Exception as e:
        print(f"❌ Error retraining {base_name}: {e}")

print("\n🎉 Retraining complete.")
