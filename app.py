from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load pipeline
with open("churn_pipeline.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = {
        "tenure": float(request.form["tenure"]),
        "MonthlyCharges": float(request.form["monthlycharges"]),
        "Contract": request.form["contract"],
        "PaymentMethod": request.form["paymentmethod"]
    }

    input_df = pd.DataFrame([data])

    prob = model.predict_proba(input_df)[0][1]
    prediction = model.predict(input_df)[0]

    result = "Customer likely to churn" if prediction == 1 else "Customer likely to stay"

    return render_template(
        "index.html",
        prediction_text=result,
        probability=round(prob * 100, 2)
    )

if __name__ == "__main__":
    app.run(debug=True)
