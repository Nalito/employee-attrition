import pickle
import streamlit as st
import numpy as np
import pandas as pd

model = pickle.load(open("model.pkl", "rb"))
pred = ''

def attrition_prediction(input_data):
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
      return 'The employee will stay'
    else:
      return 'The employee will leave'
  

def main():
    st.title("Employee Attrition Prediction")

    age = st.number_input("Age")
    education = st.number_input("Education")
    env = st.number_input("Environment Satisfaction")
    gender = st.number_input("Gender (1 for Male, 2 for Female)")
    hour_rate = st.number_input("Hourly Rate")
    job_sat = st.number_input("Job Satisfaction")
    married = st.number_input("Marital Status (I for Single, 2 for Married, 3 for Divorced)")
    income = st.number_input("Monthly Income")
    overtime = st.number_input("Over Time (1 for Yes, 0 for No)")
    wlb = st.number_input("Work Life Balance")
    pr = st.number_input("Performance Rating")
    years = st.number_input("Years At Company")
    pred = ''
    prob = ''
    word = ''

    if st.button('Predict'):
        pred = attrition_prediction([age, education, env, gender, hour_rate, job_sat, married, income, overtime, wlb, pr, years])
        df = pd.DataFrame({"Age": [age], "Education": [education], "EnvironmentSatisfaction": [env], "Gender": [gender], "HourlyRate":  [hour_rate], "JobSatisfaction": [job_sat], "MaritalStatus": [married], "MonthlyIncome": [income], "OverTime": [overtime], "WorkLifeBalance": [wlb], "PerformanceRating": [pr], "YearsAtCompany": [years]})
        pred = model.predict(df)[0]
        prob = (model.predict_proba(df).max())*100

    if pred == 0:
       word = "stay"
    else:
       word = "leave"
    
    st.success("The employee will {} (Confidence Interval: {}%)".format(word, round(prob, 2)))


if __name__ == '__main__':
    main()