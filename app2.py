from flask import Flask ,render_template,request
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

model=pickle.load(open('RandomForestClassifier.pkl',"rb"))
df = pd.read_csv("df1.csv")
#
 for i in df.columns:
   if df[i].dtypes == "object":
      le=LabelEncoder()
      df[i]=le.fit_transform(df[i].astype(str))
# #
# X =df.drop(columns='salary',axis=1)
# y =df['salary']
#
# y=le.fit_transform(y)
#
# random_sampler = RandomOverSampler(random_state=30)
# random_sampler.fit(X,y)
# X_new,y_new = random_sampler.fit_resample(X, y)
#
# X_train,X_test,y_train,y_test =train_test_split(X,y,test_size=0.2,random_state=50)
#
# RF = RandomForestClassifier(n_estimators=200 , oob_score=True,random_state=2)
# RF.fit(X_train, y_train)
# y_pred = RF.predict(X_test)
# print(accuracy_score(y_test,y_pred))
# print(X_train)


app = Flask(__name__)
@app.route('/',methods=["GET","POST"])
def index():
   age = sorted(df["age"].unique())
   workclass = (df["workclass"].unique())
   fnlwgt = sorted(df["fnlwgt"].unique())
   education = sorted(df["education"].unique())
   education_num = sorted(df["education-num"].unique(), reverse=True)
   marital_status = (df["marital-status"].unique())
   occupation = (df["occupation"].unique())
   relationship = (df["relationship"].unique())
   race = (df["race"].unique())
   sex = (df["sex"].unique())
   capital_gain = sorted(df["capital-gain"].unique())
   capital_loss = sorted(df["capital-loss"].unique())
   hours_per_week = sorted(df["hours-per-week"].unique(), reverse=True)
   country = (df["country"].unique())

   return render_template("index2.html", age=age, workclass=workclass, fnlwgt=fnlwgt, education=education,
                          education_num=education_num, marital_status=marital_status, occupation=occupation,
                          relationship=relationship, race=race, sex=sex, capital_gain=capital_gain,
                          capital_loss=capital_loss, hours_per_week=hours_per_week, country=country)

@app.route('/predict',methods=["POST"])
def predict():
   age = int(request.form.get('age'))
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

   prediction = model.predict(pd.DataFrame(
      columns=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
               'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'country'],
      data=np.array(
         [age, workclass, fnlwgt, education, education_num, marital_status, occupation, relationship, race, sex,
          capital_gain, capital_loss, hours_per_week, country]).reshape(1, 14)))
   print(prediction)

   return str(np.round(prediction[0], 2))

if __name__ == "__main__":
    app.run(debug=True)





