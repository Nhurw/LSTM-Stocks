import yfinance as yf
import pandas as pd
import numpy as np

#these are the tickers for the following companies/indexes
#^GSPC = S&P 500 index
#^DJUSDN = Dow Jones U.S. Aerospace & Defense Index
#MSFT = Microsoft
#AAPL = Apple
#NVDA = Nvidia
#AVGO = Broadcom
#ORCL = Oracle
#GOOG = Alphabet
#META = Meta Platforms
#AMZN = Amazon
#TSLA = Tesla
#LLY = Eli Lilly
#BRK-B = Berkshire Hathaway
#JPM = JPMorgan Chase
#AZN = AstraZeneca

ticker = "GCSP"
raw_train_dataframe = yf.download(ticker, start="2006-05-01", end="2024-03-5") #to have 32 day batches

raw_train_list_close = raw_train_dataframe['Close'].values.flatten()
raw_train_list_volume = raw_train_dataframe['Volume'].values.flatten()
raw_train_list_MA_5 = np.array(pd.Series(raw_train_list_close).rolling(window=5).mean())
raw_train_list_MA_10 = np.array(pd.Series(raw_train_list_close).rolling(window=10).mean())

raw_train_list_pct_change = []
for i in range(0, len(raw_train_list_close)):
    if i == 0:
        raw_train_list_pct_change.append(0)
    else:
        raw_train_list_pct_change.append(float((raw_train_list_close[i] - raw_train_list_close[i-1])/raw_train_list_close[i-1]))
raw_train_list_pct_change = np.array(raw_train_list_pct_change)

# Move these variable assignments outside the function
train_close = []
train_volume = []
train_MA_5 = []
train_MA_10 = []
train_pct_change = []
train_RSI = []
train_target = []

def compute_rsi(close_prices):
    gain_days = 0
    loss_days = 0
    for i in range(1, len(close_prices)):
        delta = close_prices[i] - close_prices[i-1]
        if delta >= 0:
            gain_days += delta
        if delta < 0:
            loss_days -= delta
    RS = float(gain_days / (loss_days + 1e-10))
    RSI = (100 - (100 / (1 + RS))) / 100 #for consistency with the other features
    return RSI

def label_target(target_pct):
    if target_pct < -0.08:
        return 0
    elif target_pct <= -0.02:
        return 1
    elif target_pct <= 0.02:
        return 2
    elif target_pct <= 0.08:
        return 3
    else:
        return 4

for i in range(0, len(raw_train_list_close)):
    if i < 14:
        pass
    elif i > len(raw_train_list_close) - 90:
        pass
    else:
        train_close.append([])
        train_volume.append([])
        train_MA_5.append([])
        train_MA_10.append([])
        train_pct_change.append([])
        train_RSI.append([])

        for j in range(0, 60):
            train_close[i-14].append(round(float(raw_train_list_close[i+j]), 6))
            train_volume[i-14].append(round(float(raw_train_list_volume[i+j]), 6))
            train_MA_5[i-14].append(round(float(raw_train_list_MA_5[i+j]), 6))
            train_MA_10[i-14].append(round(float(raw_train_list_MA_10[i+j]), 6))
            train_pct_change[i-14].append(round(float(raw_train_list_pct_change[i+j]), 6))
            train_RSI[i-14].append(round(compute_rsi(raw_train_list_close[(i+j-14):(i+j)]), 6))
        train_target.append(label_target(float((raw_train_list_close[(i-14)+90]-raw_train_list_close[i])/raw_train_list_close[i])))
        min_close_value = min(train_close[i-14])
        max_close_value = max(train_close[i-14])
        min_volume_value = min(train_volume[i-14])
        max_volume_value = max(train_volume[i-14])
        min_MA_5_value = min(train_MA_5[i-14])
        max_MA_5_value = max(train_MA_5[i-14])
        min_MA_10_value = min(train_MA_10[i-14])
        max_MA_10_value = max(train_MA_10[i-14])

        for j in range(0, 60):
            train_close[i-14][j] = (train_close[i-14][j]-min_close_value)/(max_close_value-min_close_value)
            train_volume[i-14][j] = (train_volume[i-14][j]-min_volume_value)/(max_volume_value-min_volume_value)
            train_MA_5[i-14][j] = (train_MA_5[i-14][j]-min_MA_5_value)/(max_MA_5_value-min_MA_5_value)
            train_MA_10[i-14][j] = (train_MA_10[i-14][j]-min_MA_10_value)/(max_MA_10_value-min_MA_10_value)



features = [np.array(train_close), np.array(train_volume), np.array(train_MA_5), np.array(train_MA_10), np.array(train_pct_change), np.array(train_RSI), np.array(train_target)]

shuffled_indices = np.random.permutation(len(train_close))
print("Days in dataset: ", len(train_close))
shuffled_features = [feature[shuffled_indices] for feature in features]

# Keep the function definition
def get_training_data():
    return {
        'train_close': features[0],
        'train_volume': features[1],
        'train_MA_5': features[2],
        'train_MA_10': features[3],
        'train_pct_change': features[4],
        'train_RSI': features[5],
        'train_target': features[6]
    }







