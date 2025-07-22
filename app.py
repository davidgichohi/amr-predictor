from flask import Flask, render_template, request, jsonify
import pandas as pd
import json
from xgboost import XGBClassifier
import os

app = Flask(__name__)

# -------------------------------
# Load all models and metadata
# -------------------------------
def load_models():
    model_dir = "models"
    models = {}

    for file in os.listdir(model_dir):
        if file.endswith("_model.json"):
            abx = file.replace("_xgb_model.json", "")

            # Load model
            model = XGBClassifier()
            model.load_model(os.path.join(model_dir, file))

            # Load expected feature columns
            with open(os.path.join(model_dir, f"{abx}_feature_columns.txt"), "r") as f:
                columns = f.read().strip().split(",")

            # Load label mapping
            with open(os.path.join(model_dir, f"{abx}_label_mapping.json"), "r") as f:
                labels = json.load(f)
                labels = {int(v): k for k, v in labels.items()}

            models[abx] = {
                "model": model,
                "columns": columns,
                "labels": labels
            }

    print("✅ Loaded models:", models.keys())
    return models

models = load_models()

# -------------------------------
# Load encoder mapping
# -------------------------------
with open("encoder_mapping.json", "r") as f:
    encoder_mapping = json.load(f)

# -------------------------------
# Dropdown values
# -------------------------------
dropdowns = {
    "species_list": list(encoder_mapping["species"].keys()),
    "family_list": list(encoder_mapping["family"].keys()),
    "country_list": list(encoder_mapping["country"].keys()),
    "gender_list": list(encoder_mapping["gender"].keys()),
    "age_list": list(encoder_mapping["age"].keys()),
    "speciality_list": list(encoder_mapping["speciality"].keys()),
    "source_list": list(encoder_mapping["source"].keys())
}

# -------------------------------
# Home route
# -------------------------------
@app.route('/')
def index():
    return render_template("index.html", dropdowns=dropdowns)

# -------------------------------
# Dropdown fetch API
# -------------------------------
@app.route('/dropdowns', methods=['GET'])
def get_dropdowns():
    return jsonify(dropdowns)

# -------------------------------
# Predict handler
# -------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    print("🧾 Received form data:", data)
    results = []

    for abx, model_data in models.items():
        model = model_data["model"]
        expected_cols = model_data["columns"]
        label_map = model_data["labels"]

        # Create a row with all zeros for expected columns
        row = {col: 0 for col in expected_cols}

        # Encode fields using encoder_mapping, include only valid expected columns
        for field, value in data.items():
            if field in encoder_mapping and value in encoder_mapping[field]:
                for col_name, col_val in encoder_mapping[field][value].items():
                    if col_name in expected_cols:
                        row[col_name] = col_val

        # Construct DataFrame in the correct column order
        input_df = pd.DataFrame([row], columns=expected_cols)
        print(f"🧬 Input for model '{abx}': {input_df.shape} => {input_df.columns.tolist()}")

        try:
            pred = model.predict(input_df)[0]
            prediction = label_map.get(int(pred), "Unknown")
        except Exception as e:
            prediction = f"Error: {str(e)}"

        results.append({
            "antibiotic": abx.replace("_", " ").title(),
            "prediction": prediction
        })

    print("📊 Predictions:", results)
    return jsonify({"status": "success", "predictions": results})

# -------------------------------
# Entry point
# -------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
