# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 16:15:39 2020

@author: WB
"""


import pandas as pd
import yfinance as yf
import datetime
import time
import requests
import io


url="https://pkgstore.datahub.io/core/nasdaq-listings/nasdaq-listed_csv/data/7665719fb51081ba0bd834fde71ce822/nasdaq-listed_csv.csv"
s = requests.get(url).content
companies = pd.read_csv(io.StringIO(s.decode('utf-8')))

ticker_list=companies['Symbol'].tolist()

start = datetime.datetime(1990,2,1)
end = datetime.datetime(2020,12,8)
stock_final = pd.DataFrame()

for name in ticker_list:
    print(name)
    try:
        stock = yf.download(name,start=start, end=end)
        #print(stock)
        if len(stock) == 0:
            None
        else:
            stock['Name']=name
            stock_final = stock_final.append(stock)
            
    except Exception:
        None