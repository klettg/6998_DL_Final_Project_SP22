import pandas as pd
from calendar import monthrange
from datetime import date, timedelta

data_path = '/Users/nathancuevas/github/dl_proj/Data/MSTR.csv'
df = pd.read_csv(data_path)
num_rows = df.shape[0]

m = {"Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,"Jul":7,"Aug":8,"Sep":9,"Oct":10,"Nov":11,"Dec":12}

print("Date,Open")
prev_date = date(1996, 1, 11)
for i in range(num_rows):
    date_ = df.iat[i, 0]
    open_ = df.iat[i, 1]
    arr = date_.split("/")
    mo, day, year = arr[0], arr[1], '' + arr[2] 
    curr_date = date(int(year), m[mo], int(day))
    if i > 0:
        delta = curr_date - prev_date
        daysdiff = delta.days
        if daysdiff > 1:
            for i in range(daysdiff-1):
                tmp_date = prev_date + timedelta(days=i+1)
                datestr=tmp_date.strftime('%m/%d/%Y')
                print(f'{datestr},{open_}')
    datestr=curr_date.strftime('%m/%d/%Y')
    print(f'{datestr},{open_}')
    prev_date = curr_date    

    
"""
    
    if mo != prev_mo:
        # start of new month
        for j in range(num_days):
            datestr = mo + '/' + str(j+1) + '/' + year
            print(datestr + ',' + str(index))
    """

