from flask import Flask
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error
from imblearn.over_sampling import RandomOverSampler


app = Flask(__name__)
@app.route("/")
def hello():
    return "Hello"

df=pd.read_csv("adult.csv")
X=df.drop(columns="salary",axis=1)
y=df["salary"]
random_sampler = RandomOverSampler(random_state=30)
random_sampler.fit(X, y)
X_new, y_new = random_sampler.fit_resample(X, y)

for i in df.columns:
    if df[i].dtypes == "object":
        le = LabelEncoder()
        df[i] = le.fit_transform(df[i].astype(str))


X_train,X_test,y_train,y_test =train_test_split(X_new,y_new,test_size=0.2,random_state=50)

RF = RandomForestClassifier(n_estimators=200 , oob_score=True,random_state=2)
RF.fit(X_train, y_train)
y_pred = RF.predict(X_test)
accuracy_score(y_test,y_pred)


if(__name__) == "__main__":
    app.run()
