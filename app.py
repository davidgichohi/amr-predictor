from flask import Flask, render_template, request, jsonify
import pandas as pd
import json
from xgboost import XGBClassifier
import os

app = Flask(__name__)

# Load all models and their metadata
def load_models():
    model_dir = "models"
    models = {}

    for file in os.listdir(model_dir):
        if file.endswith("_model.json"):
            abx = file.replace("_xgb_model.json", "")

            # Load model
            model = XGBClassifier()
            model.load_model(os.path.join(model_dir, file))

            # Load features
            with open(os.path.join(model_dir, f"{abx}_feature_columns.txt"), "r") as f:
                columns = f.read().splitlines()

            # Load label mappings
            with open(os.path.join(model_dir, f"{abx}_label_mapping.json"), "r") as f:
                labels = json.load(f)
                labels = {int(v): k for k, v in labels.items()}  # ensure keys are ints

            models[abx] = {
                "model": model,
                "columns": columns,
                "labels": labels
            }
    print("✅ Loaded models:", models.keys())
    return models

models = load_models()

# Dropdown options
dropdowns = {
    "species_list": [
    'Acinetobacter baumannii', 'Acinetobacter baylyi', 'Acinetobacter bereziniae',
    'Acinetobacter calcoaceticus', 'Acinetobacter colistiniresistens', 'Acinetobacter courvalinii',
    'Acinetobacter dijkshoorniae', 'Acinetobacter guillouiae', 'Acinetobacter haemolyticus',
    'Acinetobacter johnsonii', 'Acinetobacter junii', 'Acinetobacter lactucae',
    'Acinetobacter lwoffii', 'Acinetobacter nosocomialis', 'Acinetobacter pitii',
    'Acinetobacter proteolyticus', 'Acinetobacter radioresistens', 'Acinetobacter schindleri',
    'Acinetobacter seifertii', 'Acinetobacter soli', 'Acinetobacter spp', 'Acinetobacter ursingii',
    'Acinetobacter variabilis', 'Aeromonas hydrophila', 'Alcaligenes faecalis', 'Bordetella trematum',
    'Burkholderia cenocepacia', 'Citrobacter amalonaticus', 'Citrobacter braakii', 'Citrobacter freundii',
    'Citrobacter koseri', 'Citrobacter sedlakii', 'Citrobacter spp', 'Enterobacter asburiae',
    'Enterobacter bugandensis', 'Enterobacter cloacae', 'Enterobacter hormaechi',
    'Enterobacter kobei', 'Enterobacter ludwigii', 'Enterobacter roggenkampii', 'Enterobacter spp',
    'Enterobacter xiangfangensis', 'Enterococcus canintestini', 'Enterococcus faecalis',
    'Enterococcus faecium', 'Enterococcus hirae', 'Enterococcus raffinosus', 'Enterococcus spp',
    'Escherichia coli', 'Escherichia spp', 'Haemophilus influenzae', 'Klebsiella aerogenes',
    'Klebsiella oxytoca', 'Klebsiella pneumoniae', 'Klebsiella spp', 'Klebsiella variicola',
    'Moraxella catarrhalis', 'Morganella morganii', 'Myroides odoratimimus', 'Proteus hauseri',
    'Proteus mirabilis', 'Proteus penneri', 'Proteus spp', 'Proteus vulgaris',
    'Providencia alcalifaciens', 'Providencia rettgeri', 'Providencia spp', 'Providencia stuartii',
    'Pseudomonas aeruginosa', 'Pseudomonas fulva', 'Pseudomonas putida',
    'Pseudomonas putida/fluorescens Group', 'Pseudomonas spp', 'Raoultella ornithinolytica',
    'Salmonella spp', 'Serratia liquefaciens', 'Serratia marcescens', 'Serratia spp',
    'Serratia ureilytica', 'Staphylococcus Coagulase Negative', 'Staphylococcus arlettae',
    'Staphylococcus aureus', 'Staphylococcus cohnii', 'Staphylococcus epidermidis',
    'Staphylococcus haemolyticus', 'Staphylococcus hominis', 'Staphylococcus lugdunensis',
    'Staphylococcus pasteuri', 'Staphylococcus saccharolyticus', 'Staphylococcus saprophyticus',
    'Staphylococcus sciuri', 'Staphylococcus simulans', 'Staphylococcus spp',
    'Staphylococcus ureilyticus', 'Staphylococcus xylosus', 'Stenotrophomonas maltophilia',
    'Streptococcus agalactiae', 'Streptococcus anginosus', 'Streptococcus constellatus',
    'Streptococcus dysgalactiae', 'Streptococcus pneumoniae', 'Streptococcus pyogenes',
    'Streptococcus, Beta Hemolytic'
],
    "family_list": [
    'Enterobacterales', 'Enterobacteriaceae', 'Enterococcus spp', 'Haemophilus spp',
    'Moraxellaceae', 'Non-Enterobacterales', 'Non-Enterobacteriaceae',
    'Staphylococcus spp', 'Streptococcus pneumoniae', 'Streptococcus spp (no S. pneumo)'
],
    "country_list": [
    'Cameroon', 'Ghana', 'Ivory Coast', 'Kenya', 'Malawi', 'Morocco',
    'Nigeria', 'South Africa', 'Uganda'
],
    "gender_list": ['Male', 'Female'],
    "age_list": [
    '0 to 2 Years', '13 to 18 Years', '19 to 64 Years', '3 to 12 Years',
    '65 to 84 Years', '85 and Over', 'Unknown'
],
    "speciality_list": [
    'Emergency Room', 'General Unspecified ICU', 'Medicine General', 'Medicine ICU',
    'None Given', 'Other', 'Pediatric General', 'Pediatric ICU', 'Surgery General', 'Surgery ICU'
],
    "source_list": [
    'Abscess', 'Appendix', 'Bladder', 'Blood', 'Bone', 'Bronchiole', 'Bronchoalveolar lavage',
    'Bronchus', 'Burn', 'CSF', 'Carbuncle', 'Catheters', 'Cellulitis', 'Cervix', 'Colon',
    'Decubitus', 'Diverticulum', 'Endotracheal aspirate', 'Eye', 'Furuncle', 'Gall Bladder',
    'Gastric Abscess', 'Genitourinary: Other', 'Impetiginous lesions', 'Intestinal: Other',
    'Kidney', 'Liver', 'Lungs', 'None Given', 'Pancreas', 'Peritoneal Fluid', 'Prostate',
    'Respiratory: Other', 'Skin', 'Skin: Other', 'Sputum', 'Stomach', 'Thoracentesis Fluid',
    'Ulcer', 'Ureter', 'Urethra', 'Urine', 'Uterus', 'Wound'
]
}

@app.route('/')
def index():
    return render_template("index.html", dropdowns=dropdowns)

@app.route('/dropdowns', methods=['GET'])
def get_dropdowns():
    return jsonify(dropdowns)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_df = pd.DataFrame([data])
    print("🧾 Received form data:", data)
    results = []

    for abx, model_data in models.items():
        model = model_data["model"]
        expected_cols = model_data["columns"]
        label_map = model_data["labels"]

        # One-hot encode and align
        input_encoded = pd.get_dummies(input_df)
        input_encoded = input_encoded.reindex(columns=expected_cols, fill_value=0)

        try:
            pred = model.predict(input_encoded)[0]
            label_inv = {v: k for k, v in label_map.items()}
            prediction = label_inv.get(pred, "Unknown")
        except Exception as e:
            prediction = f"Error: {str(e)}"

        results.append({
            "antibiotic": abx.replace("_", " ").title(),
            "prediction": prediction
        })

    print("📊 Predictions:", results)
    return jsonify({"status": "success", "predictions": results})

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # use Render's PORT or fallback to 5000 locally
    app.run(host="0.0.0.0", port=port)
