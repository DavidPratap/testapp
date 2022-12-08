import streamlit as st
import pandas as pd
import pickle
st.title("Medical Diagnostic Web App")

# Step1 : Load the Model
model=open("rfc.pickle", 'rb')
rfc_pipeline=pickle.load(model)
model.close()

# Step2 : Create a UI for frontend user
pregs=st.slider('Pregnancies',0, 20, step=1) 
glucose=st.slider('Glucose', 40, 200, 40)
bp=st.slider('BloodPressure', 20, 140, 20)
skin=st.slider('SkinThickness', 5.0,100.0,5.0)
insulin=st.slider('Insulin',10, 900,10)
bmi=st.slider('BMI', 15,70,15)
dpf=st.slider('DiabetesPedigreeFunction', 0.5,3.0,0.5)
age=st.slider('Age', 21, 90, 21)

# Step3: Convert the users input as model data 
data={'Pregnancies':pregs, 'Glucose':glucose, 'BloodPressure':bp, 'SkinThickness':skin, 'Insulin':insulin,
       'BMI':bmi, 'DiabetesPedigreeFunction':dpf, 'Age':age}
input_data=pd.DataFrame([data])

# step4 : Get the predictions and print the results
prediction=rfc_pipeline.predict(input_data)[0]
st.write(prediction)
if st.button("Predict"):
    if prediction==1:
        st.success("has Diab")
    if prediction==0:
        st.success('Diab Free')
        
