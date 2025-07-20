import os
import joblib
from xgboost import XGBClassifier

# Directory where your models and training data are stored
models_dir = "models"

# List to keep track of successfully updated models
updated_models = []

print("🔁 Starting model update...")

# Loop through each model file
for filename in os.listdir(models_dir):
    if filename.endswith("_xgb_model.pkl"):
        model_path = os.path.join(models_dir, filename)

        try:
            # Load the existing model
            model = joblib.load(model_path)

            # Get model parameters and remove deprecated ones
            params = model.get_xgb_params()
            params.pop("use_label_encoder", None)  # Remove deprecated param

            # Create a new XGBClassifier with cleaned parameters
            new_model = XGBClassifier(**params)

            # Derive the base name of the model
            base_name = filename.replace("_xgb_model.pkl", "")
            X_path = os.path.join(models_dir, f"{base_name}_feature_matrix.pkl")
            y_path = os.path.join(models_dir, f"{base_name}_label_vector.pkl")

            if os.path.exists(X_path) and os.path.exists(y_path):
                X_train = joblib.load(X_path)
                y_train = joblib.load(y_path)

                # Refit the new model
                new_model.fit(X_train, y_train)

                # Overwrite original model with updated one
                joblib.dump(new_model, model_path)
                updated_models.append(filename)
                print(f"✅ Updated and saved: {filename}")
            else:
                print(f"❌ Training data not found for {base_name}, skipping.")

        except Exception as e:
            print(f"❌ Error updating {filename}: {e}")

# Final summary
print("\n🎉 Update complete.")
print("Updated models:")
for name in updated_models:
    print(f"  - {name}")
