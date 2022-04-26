import numpy as np
import pickle
import streamlit as st
import matplotlib.pyplot as plt

# load the saved model
loaded_model = pickle.load(open('C:/Users/user/Desktop/AI PROJECTS/Laptop_Specs_Project/trained_model.sav', 'rb'))


# create a function for prediction
def laptop_price_prediction(input_data):

    # changing the input _data to numpy_array
    input_data_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)
    return prediction


def main():

    # give title for application
    st.title('Electronics Price Predictions Web App')

    # getting the input data from the user

    name = st.text_input('Brand of machine')
    processor = st.text_input('Type of processor')
    ram = st.text_input('RAM size')
    storage = st.text_input('Storage size')
    display = st.text_input('Screen size')
    warranty = st.text_input('How many Years Warranty')

    # code for prediction
    price_predict = ''

    # creating button for prediction
    if st.button('laptop_price_predict_button'):
        price_predict = laptop_price_prediction([name, processor, ram, storage, display, warranty])

        st.success(price_predict)
        st.snow()


if __name__ == '__main__':
    main()


