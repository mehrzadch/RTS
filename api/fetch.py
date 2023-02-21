import sys
import os
import calendar as cal
import datetime as dt
import re
import time
import warnings

import pandas as pd
import requests
from bs4 import BeautifulSoup


try:
    from io import StringIO
except ImportError:
    from StringIO import StringIO


class Fetcher(object):
    def __init__(self, ticker, start, end=None, interval="1d"):
        """Initializes class variables and formats api_url string"""

        self.ticker = ticker.upper()
        self.header = {'Connection': 'keep-alive',
                       'Expires': '-1',
                       'Upgrade-Insecure-Requests': '1',
                       'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) \
                   AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.99 Safari/537.36'
                       }
        self.url = 'https://finance.yahoo.com/quote/{}/history'.format(self.ticker)
        self.interval = interval
        self.init()
        self.start = int(cal.timegm(dt.datetime(*start).timetuple()))

        if end is not None:
            self.end = int(cal.timegm(dt.datetime(*end).timetuple()))
        else:
            self.end = int(time.time())

    def init(self):
        """Returns a tuple pair of cookie and crumb used in the request"""
        with requests.session():
            website = requests.get(self.url, headers=self.header)
            soup = BeautifulSoup(website.text, 'lxml')
            self.crumb = re.findall('"CrumbStore":{"crumb":"(.+?)"}', str(soup))
            self.cookies = website.cookies

    def _get(self, events):

        with requests.session():
            url = 'https://query1.finance.yahoo.com/v7/finance/download/' \
                  '{stock}?period1={day_begin}&period2={day_end}&interval={interval}&events={events}&crumb={crumb}' \
                .format(stock=self.ticker, day_begin=self.start, day_end=self.end, interval=self.interval, events=events, crumb=self.crumb)

            data = requests.get(url, headers=self.header, cookies=self.cookies)
            content = StringIO(data.content.decode("utf-8"))
            out = pd.read_csv(content, sep=",")
            return out


    def getData(self, events):
        """Returns a list of historical data from Yahoo Finance"""
        warnings.warn("getData has been deprecated, use get_data instead", DeprecationWarning)
        return self._get(events)

    def getHistorical(self):
        """Returns a list of historical price data from Yahoo Finance"""
        warnings.warn("getHistorical has been deprecated, use get_historical instead", DeprecationWarning)
        return self._get("history")

    def getDividends(self):
        """Returns a list of historical dividends data from Yahoo Finance"""
        warnings.warn("getDividends has been deprecated, use get_dividends instead", DeprecationWarning)
        return self._get("div")

    def getSplits(self):
        """Returns a list of historical splits data from Yahoo Finance"""
        warnings.warn("getSplits has been deprecated, use get_splits instead", DeprecationWarning)
        return self._get("split")

    def getDatePrice(self):
        """Returns a DataFrame for Date and Price from getHistorical()"""
        warnings.warn("getDatePrice has been deprecated, use get_date_price instead", DeprecationWarning)
        return self.getHistorical().iloc[:, [0, 4]]

    def getDateVolume(self):
        """Returns a DataFrame for Date and Volume from getHistorical()"""
        warnings.warn("getDateVolume has been deprecated, use get_date_volume instead", DeprecationWarning)
        return self.getHistorical().iloc[:, [0, 6]]

    def get_historical(self):
        """PEP8 friendly version of deprecated getHistorical function"""
        return self._get("history")

    def get_dividends(self):
        """PEP8 friendly version of deprecated getDividends function"""
        return self._get("div")

    def get_splits(self):
        """PEP8 friendly version of deprecated getSplits function"""
        return self._get("split")

    def get_date_price(self):
        """PEP8 friendly version of deprecated getDatePrice function"""
        return self.get_historical().iloc[:, [0, 4]]

    def get_date_volume(self):
        """PEP8 friendly version of deprecated getDateVolume function"""
        return self.get_historical().iloc[:, [0, 6]]

    def save(self):
        file_dir = os.path.join("./data", self.ticker)
        file_name = "quotes.csv"
        data = self.get_historical()
        try:
            if data is None:
                return
            full_path = os.path.join(file_dir,file_name)
            include_index = False if data.index.name == None else True
            if os.path.isdir(file_dir):
                data.to_csv(full_path,index=include_index)
            else:
                os.makedirs(file_dir)
                data.to_csv(full_path,index=include_index)
        except OSError as err:
            print("OS error for symbol {} : {}".format(self.ticker,err))
        except:
            print("Unexpected error for symbol {} : {}".format(self.ticker, sys.exc_info()[0]))
