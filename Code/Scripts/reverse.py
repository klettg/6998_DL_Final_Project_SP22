import pandas as pd
from calendar import monthrange
from datetime import date

data_path = '/Users/nathancuevas/github/dl_proj/Data/MSTR_raw.csv'
df = pd.read_csv(data_path)
num_rows = df.shape[0]
print('Date,Open')
for i in range(num_rows):
    revI = -1 - i
    date = str(df.iat[revI, 0])
    open_ = str(df.iat[revI, 1])
    print(date + ',' + open_)

