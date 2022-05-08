import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# m's to plot for
ms = [10,20,30,40,50,60,70]

# path to experiment data
path_to_data = '../../data/experiment_data/'

# plot
for m in ms:
  val_path = path_to_data + 'val_acc_' + str(m) + '.csv'
  train_path = path_to_data + 'acc_' + str(m) + '.csv'
  df_val = pd.read_csv(val_path, header=None)
  df_train = pd.read_csv(train_path, header=None)
  num_epochs = len(df_val)
  np_val = df_val.to_numpy().T[0]
  np_train = df_train.to_numpy().T[0]
  x = np.linspace(1, num_epochs, num_epochs)
  plt.plot(x, np_train,label='m=' + str(m))
  plt.xlabel('epoch')
  plt.ylabel('train accuracy')
  plt.legend()
  plt.title('train acc for various m')

plt.show()
