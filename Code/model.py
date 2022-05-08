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
parser.add_argument('--m', type=int, required=False, default=30, help='num days to look back')
parser.add_argument('--epochs', type=int, required=False, default=100, help='num epochs to run')
parser.add_argument('--simulate', type=int, required=False, default=0, help='number of times to run our trading sim')
args = parser.parse_args()
print(f'model chosen is: {args.model}')
print(f'm = {args.m}')
print(f'running for {args.epochs} epochs')


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
epochs = args.epochs

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
  # CNN model was chosen
  model = Sequential()
  model.add(Input(shape=(m, num_features - 1, 1)))
  model.add(Conv2D(filters=36, kernel_size=(3, 18), padding='same', activation='relu'))
  model.add(Flatten())
  model.add(Dense(1, activation='sigmoid'))
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
elif args.model == 'DNN':
  # DNN model was chosen
  model = Sequential()
  model.add(Flatten(input_shape=(m, num_features - 1, 1)))
  model.add(Dense(8*m , activation='relu'))
  model.add(Dense(8*m, activation='relu'))
  model.add(Dense(1, activation='sigmoid'))
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
else:
  # model not recognized
  print(f'invalid model {args.model}\nexiting...')
  exit(1)

hist = model.fit(train_x_np, train_y_np, epochs=epochs, batch_size=batch_size, validation_data=[test_x_np,test_y_np])

# Evaluate the model on the test data using `evaluate`
print("\nEvaluate on test data...")
results = model.evaluate(test_x_np, test_y_np, verbose=False)
print(f"test loss, test acc: {results}\n")

if args.simulate > 0:
  #Create profitability bot: 
  tested_vals = []
  num_tests = args.simulate

  for j in range(num_tests):
    val = 10000
    correct = 0
    incorrect = 0
    rows, columns, num_features = test_x_np.shape
    predicted_buy = 0
    print(f'running simulation {j+1}/{num_tests}...')
    for i in range(rows):
      rand_index = random.randint(0, rows -1)
      i = rand_index
      bet = val * .05
      sample = np.array(test_x_np[i])
      sample = np.reshape(sample,(1,columns,num_features))
      true_y = test_y_np[i]
      prediction = model.predict(sample, verbose=False)
      #print(prediction)
      predicted_binary = 1 if prediction > 0.5 else 0
      if predicted_binary == 1: 
        predicted_buy = predicted_buy + 1
      pred_correct = 1 if predicted_binary == true_y else 0
      if pred_correct == 1:
        correct = correct + 1
        val = val + bet

      else: 
        incorrect = incorrect + 1
        val = val - bet
    print(f'bought in {predicted_buy}/{rows} days') 
    tested_vals.append(val)

  print(f'\nSimulation overall accuracy is {correct / (correct + incorrect)}') 
  from statistics import mean
  print("\naverage ending investment: ")
  print(mean(tested_vals))

