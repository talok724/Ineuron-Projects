import streamlit as st
import pandas as pd
import numpy as np
import pickle

pipe = pickle.load(open("pipeline.pkl","rb"))

st.title("INCOME CLASSIFICATION")

workclass = ['Private', 'Local-gov', 'State-gov', 'Self-emp-not-inc','Self-emp-inc', 'Federal-gov', 'Never-worked', 'Without-pay']
education = ['Below 10th', 'Some-college', 'HS-grad', '11th', 'Assoc-voc','Bachelors', 'Masters', 'Assoc-acdm']
education_num = [ 4, 10,  9,  7,  3, 11, 13, 16,  5, 14, 12, 15,  8,  6,  2,  1]
occupation = ['Otheroccupation', 'Craft-repair', 'Adm-clerical', 'Other-service','Handlers-cleaners', 'Prof-specialty', 'Machine-op-inspct','Transport-moving', 'Exec-managerial', 'Sales']
sex = ['Female', 'Male']
country = ['United-States', 'Non US']
col1, col2 = st.columns(2)
with col1:
    select_age = st.number_input("Enter the age")
with col2:
    select_workclass = st.selectbox("Select Work Class",workclass)

col3, col4, col5 = st.columns(3)
with col3:
    select_fnlwgt = st.number_input("Enter the Final Weightage")
with col4:
    select_education = st.selectbox("Select Level of Education",education)
with col5:
    select_education_num = st.selectbox("Select received marks of Education", education_num)

col6, col7 = st.columns(2)
with col6:
    select_occupation = st.selectbox("Enter the occupation",occupation)
with col7:
    select_sex = st.selectbox("Select Sex",sex)

col8, col9, col10 = st.columns(3)
with col8:
    select_capital_gain = st.number_input("Enter the Capital Gain")
with col9:
    select_capital_loss = st.number_input("Enter the Capital Loss")
with col10:
    select_hours_per_week = st.number_input("Number of hour worked per week ")

select_country = st.selectbox("Country",country)

input_df = pd.DataFrame({'age':[select_age], 'workclass':[select_workclass], 'fnlwgt':[select_fnlwgt], 'education':[select_education], 'education-num':[select_education_num],
       'occupation':[select_occupation], 'sex':[select_sex], 'capital-gain':[select_capital_gain], 'capital-loss':[select_capital_loss], 'hours-per-week':[select_hours_per_week],
       'country':[select_country]})

st.table(input_df)
st.button("Predict the salary")
result = pipe.predict(input_df)
if result == 1:
    st.subheader("The Salary is more than 50K")
else:
    st.subheader("The Salary is less than 50K")
