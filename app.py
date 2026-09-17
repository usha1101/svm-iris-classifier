from flask import Flask, render_template, request
import pandas as pd
from sklearn.svm import SVC
app = Flask(__name__)
names = ["sepal-length","sepal-width","petal-length","petal-width","target"]
df = pd.read_csv("iris_dataset.csv", names=names)
X = df.drop("target", axis=1)
y = df["target"]
model = SVC(kernel="poly",C=0.1,gamma=0.1)
model.fit(X, y)
@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        sepal_length = float(request.form["sepal_length"])
        sepal_width = float(request.form["sepal_width"])
        petal_length = float(request.form["petal_length"])
        petal_width = float(request.form["petal_width"])
        input_data = [[sepal_length,sepal_width,petal_length,petal_width]]
        prediction = model.predict(input_data)[0]
    return render_template("index.html",prediction=prediction)
if __name__ == "__main__":
    app.run(debug=True)