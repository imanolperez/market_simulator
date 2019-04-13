import datetime
import numpy as np
import pandas_datareader as pdr
from esig import tosig
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

from cvae import CVAE

class MarketGenerator:
    def __init__(self, ticker, start=datetime.date(2000, 1, 1),
                 end=datetime.date(2019, 1, 1), freq="M",
                 sig_order=4):

        self.ticker = ticker
        self.start = start
        self.end = end
        self.freq = freq
        self.order = sig_order

        self._load_data()
        self._build_dataset()
        self.generator = CVAE(n_latent=8, alpha=0.003)
        self.scaler = None

    @staticmethod
    def _leadlag(X):
        lag = []
        lead = []

        for val_lag, val_lead in zip(X[:-1], X[1:]):
            lag.append(val_lag)
            lead.append(val_lag)

            lag.append(val_lag)
            lead.append(val_lead)

        lag.append(X[-1])
        lead.append(X[-1])

        return np.c_[lag, lead]

    def _load_data(self):
        try:
            self.data = pdr.get_data_yahoo(self.ticker, self.start, self.end)["Close"]
        except:
            raise RuntimeError(f"Could not download data for {self.ticker} from {self.start} to {self.end}.")        

        self.windows = []
        for _, window in self.data.resample(self.freq):
            values = window.values / window.values[0]
            path = self._leadlag(values)

            self.windows.append(path)

    def _logsig(self, path):
        return tosig.stream2logsig(path, self.order)

    def _build_dataset(self):
        logsig = [self._logsig(path) for path in tqdm(self.windows, desc="Computing log-signatures")]
        self.scaler = MinMaxScaler(feature_range=(0.00001, 0.99999))
        logsig = self.scaler.fit_transform(logsig)        

        self.logsigs = logsig[1:]
        self.conditions = logsig[:-1]
        

    def train(self, n_epochs=10000):
        self.generator.train(self.logsigs, self.conditions, n_epochs=n_epochs)

    def generate(self, logsig, n_samples=None):
        return self.generator.generate(logsig, n_samples=n_samples)
