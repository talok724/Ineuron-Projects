from flask import Flask, render_template ,request
import pickle
import pandas as pd
import numpy as np
from flask_cors import CORS,cross_origin
from sklearn.preprocessing import PolynomialFeatures,StandardScaler,LabelEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
app = Flask(__name__)

model=pickle.load(open('modelH.pkl',"rb"))
income = pd.read_csv("df1.csv")

for i in income.columns:
    if income[i].dtypes == "object":
        le = LabelEncoder()
        income[i] = le.fit_transform(income[i].astype(str))

@app.route('/',methods=["GET","POST"])
def index():
    age = sorted(income["age"].unique())
    workclass = (income["workclass"].unique())
    fnlwgt = sorted(income["fnlwgt"].unique())
    education = sorted(income["education"].unique())
    education_num = sorted(income["education-num"].unique(), reverse=True)
    marital_status = (income["marital-status"].unique())
    occupation = (income["occupation"].unique())
    relationship = (income["relationship"].unique())
    race = (income["race"].unique())
    sex = (income["sex"].unique())
    capital_gain = sorted(income["capital-gain"].unique())
    capital_loss = sorted(income["capital-loss"].unique())
    hours_per_week = sorted(income["hours-per-week"].unique(), reverse=True)
    country = (income["country"].unique())


    return render_template("index2.html", age=age, workclass=workclass, fnlwgt=fnlwgt, education=education,
                           education_num=education_num, marital_status=marital_status, occupation=occupation,
                           relationship=relationship, race=race, sex=sex, capital_gain=capital_gain,
                           capital_loss=capital_loss, hours_per_week=hours_per_week, country=country)

@app.route('/predict',methods=["POST"])
@cross_origin()

def predict():

    age=int(request.form.get('age'))
    workclass = str(request.form.get('workclass'))
    fnlwgt = int(request.form.get('fnlwgt'))
    education = request.form.get('education')
    education_num = int(request.form.get('education_num'))
    marital_status = request.form.get('marital_status')
    occupation = request.form.get('occupation')
    relationship = request.form.get('relationship')
    race = request.form.get('race')
    sex = request.form.get('sex')
    capital_gain = int(request.form.get('capital_gain'))
    capital_loss = int(request.form.get('capital_loss'))
    hours_per_week = int(request.form.get('hours_per_week'))
    country = request.form.get('country')

    prediction=model.predict(pd.DataFrame(columns=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week','country'],
                              data=np.array([age,workclass,fnlwgt,education,education_num,marital_status,occupation,relationship,race,sex,capital_gain,capital_loss,hours_per_week,country]).reshape(1, 14)))
    print(prediction)

    return str(np.round(prediction[0],2))

if __name__ == "__main__":
    app.run(debug=True)
