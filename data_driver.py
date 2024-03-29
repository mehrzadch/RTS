#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on January 2021

@author: Mehrzad Malmirchegini
"""

import os
import pandas as pd
from datetime import datetime, timedelta
import datetime as dt
from api import Fetcher
from feature_creator import Feature_Creator
from send_email import send_email

os.chdir('/home/ubuntu/RTS')
os.system('git pull')

# Get the list of companies from ./data/dow30.csv
companies = pd.read_csv(os.path.join("input", "dow30.csv"))
symbols =  companies['Symbol'].values.tolist()
print(companies)
print(symbols)


date = dt.date.today
dt2list = lambda dt: [dt.year, dt.month, dt.day]
#start_date, end_date = '20201221', '20201228'
dt_end = dt.datetime.now() + timedelta(days=1)
dt_start = dt_end - timedelta(days=310)
start_date = dt2list(dt_start)
end_date = dt2list(dt_end)
comb_df = None
# Download quotes from yahoo and save the normalized quotes to ./data/{company symbol}/quotes.csv
for symbol in symbols:
    print("processing {}...".format(symbol))
    fetch = Fetcher(symbol, start_date, end_date)
    fetch.save()
    feature = Feature_Creator(symbol, fetch, mfi_days=2)
    df = feature.csv_creator()
    if comb_df is None:
        comb_df = df
    else:
        comb_df = pd.concat([comb_df, df])

comb_df = comb_df.reset_index(drop=True)
print(comb_df)
comb_df.to_csv('summary.csv', float_format='%.1f', index=True)
RECIPIENT  = ['mehrzad.chegini@gmail.com', 'rzanbaghi@gmail.com', 'ghassemialir@gmail.com', 'eng.honarvar@gmail.com']

send_email('summary.csv', RECIPIENT[0])
send_email('summary.csv', RECIPIENT[1])
send_email('summary.csv', RECIPIENT[2])
send_email('summary.csv', RECIPIENT[3])