import pandas_ta as ta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import aiohttp
import asyncio
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

class mydata:



  def take_data(self, _df):
    res = _df.copy()
    res['open'] = pd.to_numeric(res['open'], errors='coerce')
    res['high'] = pd.to_numeric(res['high'], errors='coerce')
    res['low'] = pd.to_numeric(res['low'], errors='coerce')
    res['close'] = pd.to_numeric(res['close'], errors='coerce')
    res['ma_prs'] = res['close'].rolling(window=(120)).mean()
    res['ma_vol'] = res['volume'].rolling(window=(120)).mean()
    res['p_open'] = res['open']/res['ma_prs']
    res['p_close'] = res['close']/res['ma_prs']
    res['p_high'] = res['high']/res['ma_prs']
    res['p_low'] = res['low']/res['ma_prs']
    res['p_vol'] = res['volume']/res['ma_vol']
    res['p_30m_open'] = res.shift(6)['p_open']
    res['p_30m_close'] = res['p_close']
    res['p_30m_high']  = res['p_high'].rolling(window=(6)).max()
    res['p_30m_low']  = res['p_low'].rolling(window=(6)).min()
    res['p_30m_vol'] = res['volume'].rolling(window=(6)).sum()/res['ma_vol']
    res['p_1h_open'] = res.shift(6)['p_open']
    res['p_1h_close'] = res['p_close']
    res['p_1h_high']  = res['p_high'].rolling(window=(12)).max()
    res['p_1h_low']  = res['p_low'].rolling(window=(12)).min()
    res['p_1h_vol'] = res['volume'].rolling(window=(12)).sum()/res['ma_vol']
    res['pOBV'] = ta.obv(res['p_close'], res['p_vol'])
    res['pOBV_30m'] = ta.obv(res['p_30m_close'], res['p_30m_vol'])
    res['pOBV_1h'] = ta.obv(res['p_1h_close'], res['p_1h_vol'])
    res['OBV'] = ta.obv(res['close'], res['volume'])
    res['volatility_1h'] = res['p_close'].rolling(window=12).std()
    res['volatility_2h'] = res['p_close'].rolling(window=24).std()
    res['volatility_3h'] = res['p_close'].rolling(window=36).std()
    res['volatility_4h'] = res['p_close'].rolling(window=48).std()

    res['ATR_12'] = ta.atr(res['p_high'], res['p_low'], res['p_close'], length=12)
    res['ATR_24'] = ta.atr(res['p_high'], res['p_low'], res['p_close'], length=24)
    res['ATR_36'] = ta.atr(res['p_high'], res['p_low'], res['p_close'], length=36)
    res['ATR_30m_12'] = ta.atr(res['p_30m_high'], res['p_30m_low'], res['p_30m_close'], length=12)
    res['ATR_30m_24'] = ta.atr(res['p_30m_high'], res['p_30m_low'], res['p_30m_close'], length=24)
    res['ATR_30m_36'] = ta.atr(res['p_30m_high'], res['p_30m_low'], res['p_30m_close'], length=36)
    res['ATR_1h_12'] = ta.atr(res['p_1h_high'], res['p_1h_low'], res['p_1h_close'], length=12)
    res['ATR_1h_24'] = ta.atr(res['p_1h_high'], res['p_1h_low'], res['p_1h_close'], length=24)
    res['ATR_1h_36'] = ta.atr(res['p_1h_high'], res['p_1h_low'], res['p_1h_close'], length=36)

    res['SMA_10'] = res['p_close'].rolling(window=10).mean()
    res['SMA_20'] = res['p_close'].rolling(window=20).mean()
    res['SMA_50'] = res['p_close'].rolling(window=50).mean()
    res['EMA_10'] = res['p_close'].ewm(span=10, adjust=False).mean()
    res['EMA_20'] = res['p_close'].ewm(span=20, adjust=False).mean()
    res['EMA_50'] = res['p_close'].ewm(span=50, adjust=False).mean()

    # Оптимизация периодов
    periods = [5, 10, 20, 50]  # Примеры периодов для OBV
    for period in periods:
        res[f'OBV_SMA_{period}'] = res['OBV'].rolling(window=period).mean()
    # Модифицированный OBV с учетом волатильности
    res['Wei_pOBV_12'] = res['pOBV'] * (1 + res['ATR_12'] / res['p_close'])
    res['Wei_pOBV_24'] = res['pOBV'] * (1 + res['ATR_24'] / res['p_close'])
    res['Wei_pOBV_36'] = res['pOBV'] * (1 + res['ATR_36'] / res['p_close'])

    res['OBV_Signal'] = np.where(res['OBV'] > res['OBV_SMA_10'], 1, 0)  # Сигнал на покупку
    res['OBV_Signal'] = np.where(res['OBV'] < res['OBV_SMA_10'], -1, res['OBV_Signal'])  # Сигнал на продажу
    res['roc_12'] = ta.roc(res['p_close'], length=12)
    res['roc_24'] = ta.roc(res['p_close'], length=24)
    res['roc_36'] = ta.roc(res['p_close'], length=36)

    res.index = pd.to_datetime(res.index)
    # Добавление столбца с будущими изменениями
    res['target'] = res.shift(-12)['close'] - res['close']
    res['target'] = res['target'].rolling(window=6).mean()

    res.dropna(inplace=True)
    #print("Shape of res after filtering:", res.shape)
    return res

  def show_target(self, _res):  # смотрим на наши цели на графике
    future_change_12 = (_res['target'] > 0).astype(int)
    prs = pd.DataFrame(_res['close'])
    prs['close_ma'] = prs['close'].rolling(window=10).mean()
    prs['future_change_12'] = future_change_12  # [:, 1]

    prs['future_change_12'] = (prs['future_change_12'] - 0.5) / 10
    prs['prs_fh12'] = prs['close'] + (prs['future_change_12'] * prs['close'])
    prs['y_ma_fh12'] = prs['prs_fh12'].rolling(window=1).mean()

    prs.dropna(inplace=True)
    prs = prs.tail(555)
    self.plot_line_prs(prs, ['close', 'close_ma', 'y_ma_fh12'])

  def plot_line_prs(self, _df, arr):
    plt.figure(figsize=(16, 6))
    for nm in arr:
      plt.plot(_df.index, _df[nm], label=nm)

    plt.title('Close Price and SMA')
    plt.xlabel('Index')
    plt.ylabel('Price')
    plt.legend()
    plt.grid()
    plt.show()

  def load_data_from_local(self, filename):
    data = pd.read_json(filename)
    data = data.rename(columns={'v': 'volume'})
    data['date_time'] = data['dt'].astype('datetime64[ms]')
    data['Date'] = pd.to_datetime(data['date_time'])
    data = data.drop(['dt', 'date_time'], axis=1)
    data.set_index('Date', inplace=True)
    return data

  async def get_kline_mexc_int(self, _symbol, _int):
    async with aiohttp.ClientSession() as session:
      smb = f'{_symbol.upper()}USDT'
      pokemon_url = f"https://api.mexc.com/api/v3/klines?interval={_int}&symbol={smb}&limit=1000"
      async with session.get(pokemon_url) as resp:
        tickers = await resp.json()
        # print(tickers)
        try:
          dfplwwwt = pd.DataFrame(tickers)
          dfplwwwt.iloc[:, [1, 2, 3, 4]] = dfplwwwt.iloc[:, [1, 2, 3, 4]].astype('float64')
          res = pd.DataFrame()
          res['dt'] = dfplwwwt.iloc[:, 0].astype('int64')
          res['dt'] = res['dt'] + (60 * 60 * 3) * 1000
          res['date_time'] = res['dt'].astype('datetime64[ms]')
          res['open'] = dfplwwwt.iloc[:, 1]
          res['high'] = dfplwwwt.iloc[:, 2]
          res['low'] = dfplwwwt.iloc[:, 3]
          res['close'] = dfplwwwt.iloc[:, 4]
          res['volume'] = dfplwwwt.iloc[:, 5].astype('float64')
          # res = res.iloc[::-1].set_index(res.index)  # не надо переворачивать, здесь все по порядку

          return (res)
        except Exception as e:
          print(f"error get_kline {e}")
          return (pd.DataFrame())

  async def get_kline_bybit(self, _symbol, interval,  _from=0):
    # https://bybit-exchange.github.io/docs/v5/market/kline
    async with aiohttp.ClientSession() as session:
      smb = f'{_symbol.upper()}USDT'
      l15m = 1000 * 60 * interval  # если первый вызов
      now = datetime.now()
      end = int(now.timestamp()) - (l15m * _from)  # если первый вызов то текущее ДТ, иначе вычитаем прошлый период
      start = end - 1000 * 60 * interval
      pokemon_url = f'https://api.bybit.com/v5/market/kline?symbol={smb}&interval={interval}&limit=1000&start={start * 1000}&end={end * 1000}'
      #print(pokemon_url)
      async with session.get(pokemon_url) as resp:
        tickers = await resp.json()
        # print(tickers)
        try:
          dfplwwwt = pd.DataFrame(tickers['result']['list'])
          dfplwwwt.iloc[:, [1, 2, 3, 4]] = dfplwwwt.iloc[:, [1, 2, 3, 4]].astype('float64')
          res = pd.DataFrame()
          res['dt'] = dfplwwwt.iloc[:, 0].astype('int64')
          res['dt1'] = res['dt']
          res['date_time'] = res['dt'].astype('datetime64[ms]')
          res['open'] = dfplwwwt.iloc[:, 1]
          res['high'] = dfplwwwt.iloc[:, 2]
          res['low'] = dfplwwwt.iloc[:, 3]
          res['close'] = dfplwwwt.iloc[:, 4]
          res['volume'] = dfplwwwt.iloc[:, 5].astype('float64')
          res = res.iloc[::-1].set_index(res.index)  # не надо переворачивать, здесь все по порядку
          res.set_index('dt1', inplace=True)
          return (res)
        except Exception as e:
          print(f"error get_kline {e}")
          return (pd.DataFrame())

  async def load_save_one_year_smb(self, _smb, _interval):
    df = pd.DataFrame()
    pos = 0
    for i in range(0,320):
        res = await self.get_kline_bybit(f'{_smb}',_interval, pos)
        if len(res) > 0:
          if len(df) > 0:
            print(df.index.min(), '>', res.index.max(), ':::', df['dt'].min(), '>', res['dt'].max())
          df = pd.concat([df, res], axis=0)
          print(f'step={pos} hours={len(df)} days={int(len(df)/24)} years={round(len(df)/24/365,1)}')
          pos +=1
        else:
          continue
        await asyncio.sleep(1)
    df = df.sort_values(by='dt', ascending=True)
    f_reset = df.reset_index(drop=True)
    filepath = Path(f'data/{_smb}{_interval}.json')
    f_reset.to_json(filepath)