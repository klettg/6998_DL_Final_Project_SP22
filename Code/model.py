# ML/DL imports
import pandas as pd
import os
import numpy as np
import random
from tensorflow import keras
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten 
from tensorflow.keras.layers import BatchNormalization, AveragePooling2D 
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping

# ignore pandas FutureWarnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# parsing arguments
import argparse, os

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True, help='CNN or DNN')
parser.add_argument('--m', type=int, required=True, help='num days to look back')
args = parser.parse_args()
print(f'model chosen is: {args.model}')
print(f'm = {args.m}')


#load and transform data
data_path = '../data/data_v2.csv'
df = pd.read_csv(data_path)
num_rows = df.shape[0]

#Add column for each row indicating whether row + 1 is higher/lower: 
num_cols = len(df.columns) - 1
df['PriceEth_next_day'] = 0 # column 6
df['Increased'] = 1 #column 7
for i in range(0, num_rows - 1):
  curr_Price = df.iat[i,num_cols] #eth price in last column:
  next_day_price = df.iat[i+1,num_cols]
  df.iloc[i,num_cols+1] = next_day_price
  df.iloc[i,num_cols+2] = 1 if next_day_price > curr_Price else 0

#drop last day in csv since we have no comparison about price increase/decrease
df.drop(df.tail(1).index,inplace=True)

m = args.m
batch_size = 64
epochs = 100

#split into test, train 

#reminder, do not split randomly !! Split based on date

index_of_last_train_example = int(num_rows * 0.6)
index_of_first_train_example2 = int(num_rows *0.8)


#drop next_day price (saved in df in case we run regression)
df = df.drop('PriceEth_next_day', 1)
df = df.drop('Date(UTC)', 1)
df = df.drop('UnixTimeStamp', 1)

#normalize: 
for column in df:
    if column not in ['Increased', 'ERC20Transfer', 'ERC20Addr','GtrendsEth']:
      df[column] = (df[column] / df[column][0]) - 1

#Create sequences, convert to numpy (training)

train_values_arr = []
train_values_y = []

#create sequences of size m

for i in range(0, index_of_last_train_example - m): 
  sequence_indices = range(i, i+m)
  sequence = df.iloc[sequence_indices]
  sequence = sequence.drop('Increased', 1)
  seq_as_numpy = sequence.to_numpy()

  train_values_arr.append(seq_as_numpy)

  #Get y value for this sequence
  m_val_for_seq = int(df.iloc[i+m]['Increased'])
  train_values_y.append(m_val_for_seq)


for i in range(index_of_first_train_example2, len(df.index) - m): 
  sequence_indices = range(i, i+m)
  sequence = df.iloc[sequence_indices]
  sequence = sequence.drop('Increased', 1)
  seq_as_numpy = sequence.to_numpy()

  train_values_arr.append(seq_as_numpy)

  #Get y value for this sequence
  m_val_for_seq = int(df.iloc[i+m]['Increased'])
  train_values_y.append(m_val_for_seq)

train_x_np = np.array(train_values_arr)
train_y_np = np.array(train_values_y)

num_features = len(df.columns)

test_values_arr = []
test_values_y = []
test_buy = 0
#create sequences of size m

for i in range(index_of_last_train_example, index_of_first_train_example2): 
  sequence_indices = range(i, i+m)
  sequence = df.iloc[sequence_indices]
  sequence = sequence.drop('Increased', 1)
  seq_as_numpy = sequence.to_numpy()

  test_values_arr.append(seq_as_numpy)

  #Get y value for this sequence
  m_val_for_seq = int(df.iloc[i+m]['Increased'])
  if m_val_for_seq == 1: 
    test_buy = test_buy + 1

  test_values_y.append(m_val_for_seq)

test_x_np = np.array(test_values_arr)
test_y_np = np.array(test_values_y)

num_features = len(df.columns)

if args.model == 'CNN':
  model = Sequential()
  model.add(Input(shape=(m, num_features - 1, 1)))
  model.add(Conv2D(filters=36, kernel_size=(3, 18), padding='same', activation='relu'))
  model.add(Flatten())
  model.add(Dense(1, activation='sigmoid'))
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
elif args.model == 'DNN':
  model = Sequential()
  model.add(Flatten(input_shape=(m, num_features - 1, 1)))
  model.add(Dense(8*m , activation='relu'))
  model.add(Dense(8*m, activation='relu'))
  model.add(Dense(1, activation='sigmoid'))
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
else:
  print(f'invalid model {args.model}\nexiting...')
  exit(1)

hist = model.fit(train_x_np, train_y_np, epochs=epochs, batch_size=batch_size, validation_data=[test_x_np,test_y_np])
