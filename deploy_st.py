import streamlit as st
import pickle

# load the pickled model
with open('model_lr.pkl','rb') as file:
    model_lr=pickle.load(file)

# create streamlit web app for linear regression to predict salary

# creating header for our app
st.header('Salary predictor')

st.sidebar.header('This is my web app')

# creating a slider for our app
X_test=st.sidebar.slider('Select X to get Y hat',0,10,5)

st.write('X test is: ',X_test)

yhat_test = model_lr.predict([[X_test]])

st.write('b0 is: ',round(model_lr.intercept_,2))
st.write('b1 is: ',round(model_lr.coef_[0],2))
st.write('yhat test is: ',yhat_test)