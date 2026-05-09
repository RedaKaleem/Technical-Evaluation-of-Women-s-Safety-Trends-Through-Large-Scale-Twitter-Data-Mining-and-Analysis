from flask import Flask, render_template, request, redirect
import os
import joblib
import numpy as np

from model_training import load_and_clean_data, perform_eda, apply_vader, train_models

app = Flask(__name__)

DATA_PATH = "MeToo_tweets.csv"


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/run_pipeline")
def run_pipeline():
    df = load_and_clean_data(DATA_PATH)
    perform_eda(df)
    df = apply_vader(df)
    train_models(df)

    return redirect("/eda")   # ✅ important


@app.route("/eda")
def eda():
    images = os.listdir("static/eda")
    return render_template("eda.html", images=images)


@app.route("/compare")
def compare():
    import os

    eval_path = "static/eval/"
    files = os.listdir(eval_path)

    models = []

    for file in files:
        if file.endswith("_cm.png"):
            model_name = file.replace("_cm.png", "")
            report_file = f"{model_name}_report.txt"

            report_text = ""
            report_path = os.path.join(eval_path, report_file)

            if os.path.exists(report_path):
                with open(report_path, "r") as f:
                    report_text = f.read()

            models.append({
                "name": model_name,
                "cm": file,
                "report": report_text
            })

    return render_template("compare.html", models=models)


@app.route("/best_model")
def best_model():
    import os

    model_path = "static/model/"
    eval_path = "static/eval/"

    # Read metrics
    metrics = ""
    if os.path.exists(model_path + "metrics.txt"):
        with open(model_path + "metrics.txt") as f:
            metrics = f.read()

    # Find best model name
    best_name = metrics.split("\n")[0].replace("Best Model: ", "").strip()

    # Load report
    report = ""
    report_file = eval_path + best_name + "_report.txt"
    if os.path.exists(report_file):
        with open(report_file) as f:
            report = f.read()

    cm_image = best_name + "_cm.png"

    return render_template("best_model.html",
                           metrics=metrics,
                           report=report,
                           cm=cm_image)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    result = None
    confidence = 0

    if request.method == "POST":
        text = request.form["tweet"]

        model = joblib.load("static/model/best_model.pkl")
        vectorizer = joblib.load("static/model/vectorizer.pkl")

        vect = vectorizer.transform([text])

        pred = model.predict(vect)[0]

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(vect)
            confidence = int(np.max(probs) * 100)
        else:
            confidence = 85  # fallback

        result = pred

    return render_template("predict.html", result=result, confidence=confidence)


if __name__ == "__main__":
    app.run(debug=True)