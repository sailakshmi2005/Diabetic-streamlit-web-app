import numpy as np
import streamlit as st
import pickle

# Load the model
loaded_model = pickle.load(open("Diabetic_model.sav",'rb'))

def Diabetic_system(input_data):

    # convert data into as array
    input_data_asarray = np.asarray(input_data)

    # reshape the data as we predicting for one instance
    reshaped_data = input_data_asarray.reshape(1,-1)
    prediction = loaded_model.predict(reshaped_data)

    if (prediction[0] == 0):
        return 'This person is Non diabetic'
    else:
        return 'This person is diabetic'
    
def main():

    # Giving a title
    st.title('🩺 Diabetes Prediction Web App')

    # getting the input from the user
    Pregnancies = st.number_input('Number of Pragnancies',min_value=0)
    Glucose = st.number_input('Glucose Level',min_value=0)
    BloodPressure = st.number_input('BloodPressure value',min_value=0)
    SkinThickness = st.number_input('SkinThickness value')
    Insulin = st.number_input('Insulin level')
    BMI = st.number_input('BMI value')
    DiabetesPedigreeFunction = st.number_input('DiabetesPedigreeFunction value')
    Age = st.number_input('Age of the person')

    # code for diabetics'
    diagnosis = ''

    if st.button('Diabetes Test Result'):
        diagnosis = Diabetic_system([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])

        st.success(diagnosis)

if __name__ == '__main__':
    main()

