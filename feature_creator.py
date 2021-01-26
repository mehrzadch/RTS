import sys
import os
import math
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from collections import OrderedDict
import numpy as np
import pdb

class BaseData(object):
    def __init__(self,symbol:str):
        self.symbol = symbol

    def save(self,file_dir:str,file_name:str,data:pd.DataFrame):
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
            print("OS error for symbol {} : {}".format(self.symbol,err))
        except:
            print("Unexpected error for symbol {} : {}".format(self.symbol, sys.exc_info()[0]))

class Feature_Creator(BaseData):
    def __init__(self, symbol, fetch, mfi_days=14):
        BaseData.__init__(self,symbol)
        self.days  = mfi_days
        self.file_path = "./data/{}/quotes.csv".format(self.symbol)
        self.data = fetch.get_historical()
        self.data_normal = None

        cols = self.data.columns.values
        cols_check = "Date,Open,High,Low,Close,Adj Close,Volume".split(',')
        missing = False
        for col in cols:
            found = False
            for name in cols_check:
                if col == name:
                    found = True
                    break
            if not found:
                print("The column {} is missing.".format(col))
                missing = True
                break
        if not missing:
            self.data['Date'] = pd.to_datetime(self.data['Date'])
            self.data.sort_values('Date',inplace=True)
            self.data.reset_index(drop=True,inplace=True)
            self.data.index.name = 'index'

    def csv_creator(self):
        self.num_samples = self.data['Close'].count()
        self.calc_sma()
        self.calc_average_volume()
        self.calc_rsi()
        self.calc_macd_signal()
        df = self.create_summary_df()
        # self.calc_log_return("Adj Close")
        # self.calc_mfi()
        # self.normalize_data()
        # self.save_plain_data()
        # self.save_stock_data()
        # self.save_normalized_data()

        return df

    def create_summary_df(self):
        summary_metric_list = ['Close', 'MA', 'Volume', 'Average Volume', 'RSI', "MACD", 'Signal']
        summary_metric_list_new = []
        for l in summary_metric_list:
            if l=='MA':
                for ma_name in self.ma_names:
                    summary_metric_list_new.append(ma_name)
            else:
                summary_metric_list_new.append(l)

        dict = {}
        dict ['Company'] = self.symbol
        for l in summary_metric_list_new:
            dict[l] = self.data[l][self.num_samples - 1]

        df = pd.DataFrame(dict, index=[0])
        return df


    def add_metric(self, metric, name):
        metric_series = pd.Series(metric)
        self.data.insert(len(self.data.columns.tolist()), name, metric_series)

    def calc_sma(self):
        ma_days = [5, 10, 20, 50, 100, 200, 300]
        x = self.data['Close']

        self.ma_names = []
        for l in ma_days:
            ma_vec = []
            for n in range(self.num_samples ):
                st = np.max([0, n - l + 1])
                ma = x[st:n].mean()
                ma_vec.append(ma)
            name = "MA {} days".format(l)
            self.ma_names.append(name)
            ma_vec = np.array(ma_vec).reshape(-1)
            self.add_metric(ma_vec, name)

    def calc_average_volume(self):
        W = 63
        x = self.data["Volume"].to_numpy()
        ave_volume_vec = []
        for i in range(x.shape[0]):
            if i==0:
                ave_volume_vec.append(x[i])
            else:
                st = np.max([0, i-W+1])
                ave_volume_vec.append(np.mean(x[st:i]))

        ave_volume_vec = np.array(ave_volume_vec)
        self.add_metric(ave_volume_vec, "Average Volume")

    def calc_rsi(self):
        offset = 14
        x = self.data["Close"].to_numpy()
        delta = x[1:] - x[:-1]
        delta = np.concatenate((np.zeros(1), delta))
        ind_u = np.where(delta > 0)[0]
        ind_d = np.where(delta < 0)[0]
        delta_u = np.zeros(delta.shape)
        delta_d = np.zeros(delta.shape)
        delta_u[ind_u] = delta[ind_u]
        delta_d[ind_d] = delta[ind_d]
        gain_u = 0
        gain_d = 0
        rsi_vec = []
        for i in range(x.shape[0]):
            if i < offset:
                gain_u += delta_u[i]
                gain_d += delta_d[i]
            else:
                gain_u = (offset - 1) / offset * gain_u + 1 / offset * delta_u[i]
                gain_d = (offset - 1) / offset * gain_d + 1 / offset * delta_d[i]
            if i==0:
                rsi = 0
            else:
                if np.abs(gain_d) !=0:
                    rsi = 100 * (1 - 1 / (1+gain_u / np.abs(gain_d)))
                else:
                    rsi = 100
            rsi_vec.append(rsi)

        rsi_vec = np.array(rsi_vec)
        self.add_metric(rsi_vec, "RSI")

    def calc_macd_signal(self):
        macd = self.calc_ema(12) - self.calc_ema(26)
        self.add_metric(macd, "MACD")

        signal = self.calc_signal(9)
        self.add_metric(signal, "Signal")

    def calc_ema(self, time_period):
        x = self.data["Close"].to_numpy()
        ave_close_vec = []
        for i in range(x.shape[0]):
            close = x[i]
            if i==0:
                ema = close
            else:
                alpha = 2/(time_period+1)
                ema = (1-alpha)*ema + alpha*close
                if i == time_period-1:
                    ema = np.mean(x[0:time_period])
            ave_close_vec.append(ema)
        ave_close_vec = np.array(ave_close_vec).reshape(-1)
        return ave_close_vec

    def calc_signal(self, time_period):
        x = self.data["MACD"].to_numpy()
        signal_vec = []
        for i in range(x.shape[0]):
            macd = x[i]
            if i==0:
                signal = macd
            else:
                alpha = 2/(time_period+1)
                signal = (1-alpha)*signal + alpha*macd
                if i == time_period-1:
                    signal = np.mean(x[0:time_period])
            signal_vec.append(signal)
        signal_vec = np.array(signal_vec).reshape(-1)
        return signal_vec





    def calc_log_return(self,col_name:str):
        values = self.data[col_name].values
        log_returns = np.zeros_like(values)
        for idx in range(1,len(values)):
            log_returns[idx] = math.log(values[idx]/values[idx-1])
        self.data[col_name+"_log_returns"] = pd.Series(log_returns, index = self.data.index)

    def calc_mfi(self):
        self.data.insert(1, "Price", pd.Series((self.data["High"] + self.data["Low"] + self.data["Adj Close"]) / 3))
        typ_price = pd.DataFrame((self.data["High"] + self.data["Low"] + self.data["Adj Close"]) / 3, columns=["price"])
        typ_price['volume'] = self.data["Volume"]
        typ_price['pos'] = 0
        typ_price['neg'] = 0
        typ_price['mfi_index'] = 0.0
        for idx in range(1, len(typ_price)):

            # Calculate the positive raw money on each day based on the price and volume
            if typ_price['price'].iloc[idx] > typ_price['price'].iloc[idx - 1]:
                typ_price.at[idx, 'pos'] = typ_price['price'].iloc[idx] * typ_price['volume'].iloc[idx]

                # Calculate the negative raw money on each day based on the price and volume
            else:
                typ_price.at[idx, 'neg'] = typ_price['price'].iloc[idx] * typ_price['volume'].iloc[idx]

        pointer = 1
        for idx in range(self.days, len(typ_price)):
            pos = typ_price['pos'].iloc[pointer:idx + 1].sum()
            neg = typ_price['neg'].iloc[pointer:idx + 1].sum()

            if neg != 0:
                base = (1.0 + (pos / neg))
            else:
                base = 1.0
            typ_price.at[idx, 'mfi_index'] = 100.0 - (100.0 / base)
            pointer += 1

        self.data["mfi_index"] = pd.Series(typ_price["mfi_index"].values, index=typ_price.index)

    def normalize_data(self):
        index = self.data.index.values[self.days :]
        table = OrderedDict()
        table['close'] = self.__flatten_data('Adj Close')
        table['returns'] = self.__flatten_data('Adj Close_log_returns')
        table['mfi'] = self.__flatten_data('mfi_index')
        table['normal_close'] = self.__scale_data('Adj Close')
        table['normal_returns'] = self.__scale_data('Adj Close_log_returns')
        table['normal_mfi'] = self.__scale_data('mfi_index')
        self.data_normal = pd.DataFrame(table,index=index)
        self.data_normal.index.name = 'index'

    def __scale_data(self,col_Name:str):
        values = self.data[col_Name].iloc[self.days :].values.reshape(-1,1)
        scaler = MinMaxScaler(feature_range=(-1,1))
        return scaler.fit_transform(values).flatten()

    def __flatten_data(self,col_Name:str):
        return self.data[col_Name].iloc[self.days :].values.flatten()

    def save_plain_data(self):
        file_dir = os.path.join("./data", self.symbol)
        BaseData.save(self, file_dir, "quote_mehrzad.csv",self.data)

    def save_stock_data(self):
        file_dir = os.path.join("./data",self.symbol)
        BaseData.save(self,file_dir,"quote_processed.csv",self.data_normal)

    def save_normalized_data(self):
        file_dir = os.path.join("./data",self.symbol)
        BaseData.save(self,file_dir,"normalized.csv",self.data_normal)

    def read_csv(self):
            try:
                data = pd.read_csv(self.file_path)
                return data
            #            self.__symbol = symbol
            except OSError as err:
                print("OS error {}".format(err))
                return None

class Volatility(object):
    def __init__(self,symbol:str):
        try:
            path_norm_data = "./data/{}/normalized.csv".format(symbol)
            dataset = pd.read_csv(path_norm_data,index_col='index')
            self.__volatility = dataset['returns'].std() * math.sqrt(252)
        except:
            self.__volatility = -1

    @property
    def annual(self):
        return self.__volatility

