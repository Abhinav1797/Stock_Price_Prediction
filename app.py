import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
import pandas_datareader as data
import streamlit as st
from keras.layers import Dense,LSTM
import matplotlib.pyplot as plt
from keras.models import load_model


#getting the data
start_date = '2018-06-18'
end_date =   '2023-06-17'
st.title('Stock Price Prediction')

user_input = st.text_input('Enter Stock Ticker','MSFT')
#df = pd.read_csv("/Users/abhinavsharma/Desktop/Project/MSFT.csv")
#df = data.DataReader(user_input,'yahoo',start,end)
import yfinance as yf
df = yf.download(user_input, start=start_date, end=end_date)

#describing data
st.subheader('Data From 2018 - 2023')
st.write(df.describe())

#visualisations
#visualise the result
st.subheader('Closing Prices')
fig = plt.figure(figsize=(19, 6))
plt.plot(df['Close'])
plt.title('Stock Price')
plt.xlabel('Year',fontsize=18)
plt.ylabel('Closing Price',fontsize = 18)
plt.xticks(rotation=0)
plt.show()
st.pyplot(fig)

#splitting data into training and testing

#create a new dataframe with only close column
data = df.filter(['Close'])
#convert the dataframe to a numpy array
dataset = data.values
#get the number of rows to train the model on
training_data_len = math.ceil(len(dataset) * .8)
#80 percent of data is allocated to training_set
#scale the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

#create the training dataset
#create the scaled training dataset
train_data = scaled_data[0:training_data_len, :]

#split the data into x_train and y_train


#load model
model = load_model('keras_model.h5')

#create the testing dataset
#create a new array containing scaled values from inndex947 to 1258
test_data = scaled_data[training_data_len-60:]
#create the dataset x_test and y_test
x_test = []
y_test = dataset[training_data_len:]
for i in range(60,len(test_data)):
  x_test.append(test_data[i-60:i])
  
#convert data to numpy array
x_test = np.array(x_test)
#get the models predicted price values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

import matplotlib.lines as mlines

train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

# Visualize the result
st.subheader('Predictions VS Original')
fig2 = plt.figure(figsize=(16, 8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Closing Price USD ($)', fontsize=18)

# Plotting the lines with different colors
plt.plot(train['Close'], color='blue', label='Train')
plt.plot(valid['Close'], color='red', label='Actual Close')
plt.plot(valid['Predictions'], color='green', label='Predicted')

# Creating custom legends with corresponding colors
eight = mlines.Line2D([], [], color='blue', marker='s', ls='', label='Train')
nine = mlines.Line2D([], [], color='red', marker='s', ls='', label='Actual Close')
ten = mlines.Line2D([], [], color='green', marker='s', ls='', label='Predicted')
plt.legend(handles=[eight, nine, ten])

#plt.legend(['Train','Actual Closing','Predictions'],loc = 'upper left')
st.pyplot(fig2)

micro_data = yf.download(user_input, start=start_date, end=end_date)
#create a new dataframe
new_df = micro_data.filter(['Close'])

#get the last 60 day closing price values and convert the dataframe to array
last_60_days = new_df[-60:].values

#scale the data to be values between 0 and 1
last_60_days_scaled = scaler.transform(last_60_days)

#create an empty list

X_test = []

#Append the past 60 days
X_test.append(last_60_days_scaled)
#convert X_test dataset to a numpy array
X_test = np.array(X_test)

#Reshape the data
X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))

#get the predicted scaled data
pred_price = model.predict(X_test)
#undo the scaling
pred_price = scaler.inverse_transform(pred_price)
pred_price = int(pred_price[0][0])
print(pred_price)

st.write("The Predicted Price is")
st.write(pred_price)


start_date = '2018-06-18'
end_date =   '2023-06-21'
microsoft_20_June = yf.download(user_input, start=start_date, end=end_date)
print("Actual Price on 20th June")
actual_price = microsoft_20_June['Close'].iloc[-1]

st.write("The Actual Price is")
actual_price = math.trunc(actual_price)
st.write(actual_price)

