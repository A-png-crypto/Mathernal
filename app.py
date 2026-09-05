
from flask import Flask, render_template, request, jsonify
import numpy as np
import os

app = Flask(__name__)


# ============================================================
# LOAD ANN MODELS
# ============================================================

# We will connect the actual trained ANN models here.
# The files should be placed inside the "models" folder.

polyclinic_model = None
mendeley_model = None
uci_model = None


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():
    return render_template("index.html")


# ============================================================
# PREDICTION
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:
        data = request.get_json()

        if not data:
            return jsonify({
                "success": False,
                "error": "No data was received."
            }), 400

        dataset = data.get("dataset")

        # ----------------------------------------------------
        # POLY CLINIC DATASET
        # ----------------------------------------------------

        if dataset == "polyclinic":

            age = float(data["age"])
            systolic_bp = float(data["systolic_bp"])
            diastolic_bp = float(data["diastolic_bp"])
            blood_glucose = float(data["blood_glucose"])
            bmi = float(data["bmi"])
            heart_rate = float(data["heart_rate"])
            parity = float(data["parity"])
            hemoglobin = float(data["hemoglobin"])

            features = np.array([[
                age,
                systolic_bp,
                diastolic_bp,
                blood_glucose,
                bmi,
                heart_rate,
                parity,
                hemoglobin
            ]])

            # ------------------------------------------------
            # TEMPORARY
            # ------------------------------------------------
            # The actual trained ANN model will be connected
            # here after the model files are added.

            if polyclinic_model is None:
                return jsonify({
                    "success": False,
                    "error": "Poly Clinic ANN model has not been connected yet."
                }), 500


            prediction = polyclinic_model.predict(features)

            predicted_class = np.argmax(prediction, axis=1)[0]

            risk_levels = {
                0: "Low Risk",
                1: "Medium Risk",
                2: "High Risk"
            }

            risk = risk_levels.get(
                int(predicted_class),
                "Unknown"
            )


        # ----------------------------------------------------
        # MENDELEY DATASET
        # ----------------------------------------------------

        elif dataset == "mendeley":

            age = float(data["age"])
            systolic_bp = float(data["systolic_bp"])
            diastolic_bp = float(data["diastolic_bp"])
            blood_sugar = float(data["blood_sugar"])
            body_temperature = float(data["body_temperature"])
            bmi = float(data["bmi"])
            previous_complications = float(
                data["previous_complications"]
            )
            preexisting_diabetes = float(
                data["preexisting_diabetes"]
            )
            gestational_diabetes = float(
                data["gestational_diabetes"]
            )
            mental_health = float(
                data["mental_health"]
            )
            heart_rate = float(data["heart_rate"])

            features = np.array([[
                age,
                systolic_bp,
                diastolic_bp,
                blood_sugar,
                body_temperature,
                bmi,
                previous_complications,
                preexisting_diabetes,
                gestational_diabetes,
                mental_health,
                heart_rate
            ]])

            # Actual Mendeley ANN will be connected here.

            if mendeley_model is None:
                return jsonify({
                    "success": False,
                    "error": "Mendeley ANN model has not been connected yet."
                }), 500


            prediction = mendeley_model.predict(features)

            predicted_class = np.argmax(prediction, axis=1)[0]

            risk_levels = {
                0: "Low Risk",
                1: "Medium Risk",
                2: "High Risk"
            }

            risk = risk_levels.get(
                int(predicted_class),
                "Unknown"
            )


        # ----------------------------------------------------
        # UCI DATASET
        # ----------------------------------------------------

        elif dataset == "uci":

            age = float(data["age"])
            systolic_bp = float(data["systolic_bp"])
            diastolic_bp = float(data["diastolic_bp"])
            blood_sugar = float(data["blood_sugar"])
            body_temperature = float(data["body_temperature"])
            heart_rate = float(data["heart_rate"])

            features = np.array([[
                age,
                systolic_bp,
                diastolic_bp,
                blood_sugar,
                body_temperature,
                heart_rate
            ]])

            # Actual UCI ANN will be connected here.

            if uci_model is None:
                return jsonify({
                    "success": False,
                    "error": "UCI ANN model has not been connected yet."
                }), 500


            prediction = uci_model.predict(features)

            predicted_class = np.argmax(prediction, axis=1)[0]

            risk_levels = {
                0: "Low Risk",
                1: "Medium Risk",
                2: "High Risk"
            }

            risk = risk_levels.get(
                int(predicted_class),
                "Unknown"
            )


        else:

            return jsonify({
                "success": False,
                "error": "Invalid dataset selected."
            }), 400


        # ====================================================
        # RETURN RESULT
        # ====================================================

        return jsonify({
            "success": True,
            "dataset": dataset,
            "risk": risk
        })


    except KeyError as e:

        return jsonify({
            "success": False,
            "error": f"Missing field: {str(e)}"
        }), 400


    except ValueError:

        return jsonify({
            "success": False,
            "error": "Please enter valid numerical values."
        }), 400


    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=True
    )
