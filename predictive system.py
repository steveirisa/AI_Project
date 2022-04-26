import numpy as np 
import pickle 

# loading the model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))


# Making a predictive system
input_data = (67, 24, 1, 4, 13, 3)

#changing the input _data to numpy_array
input_data_numpy_array = np.asarray(input_data)

#reshape the array as we are predicting for one instance
input_data_reshaped = input_data_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(f'the price of the laptop you want is :', prediction)