import yfinance as yf
import pandas as pd
import numpy as np
import pickle

ticker = "ITA"
raw_train_dataframe = yf.download(ticker, start="2025-09-25", end="2026-01-12")


raw_train_list_close = raw_train_dataframe['Close'].values.flatten()
raw_train_list_volume = raw_train_dataframe['Volume'].values.flatten()
print(len(raw_train_list_close))
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
#train_target = []

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
    elif i > len(raw_train_list_close) - 60:
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
        #train_target.append(label_target(float((raw_train_list_close[(i-14)+90]-raw_train_list_close[i])/raw_train_list_close[i])))
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

with open("lstm_weights.pkl", "rb") as f:
    weights_and_biases = pickle.load(f)

W_f_1, W_i_1, W_c_1, W_o_1 = weights_and_biases["W_f_1"], weights_and_biases["W_i_1"], weights_and_biases["W_c_1"], weights_and_biases["W_o_1"]
W_f_2, W_i_2, W_c_2, W_o_2 = weights_and_biases["W_f_2"], weights_and_biases["W_i_2"], weights_and_biases["W_c_2"], weights_and_biases["W_o_2"]
U_f_1, U_i_1, U_c_1, U_o_1 = weights_and_biases["U_f_1"], weights_and_biases["U_i_1"], weights_and_biases["U_c_1"], weights_and_biases["U_o_1"]
U_f_2, U_i_2, U_c_2, U_o_2 = weights_and_biases["U_f_2"], weights_and_biases["U_i_2"], weights_and_biases["U_c_2"], weights_and_biases["U_o_2"]
b_f_1, b_i_1, b_c_1, b_o_1 = weights_and_biases["b_f_1"], weights_and_biases["b_i_1"], weights_and_biases["b_c_1"], weights_and_biases["b_o_1"]
b_f_2, b_i_2, b_c_2, b_o_2 = weights_and_biases["b_f_2"], weights_and_biases["b_i_2"], weights_and_biases["b_c_2"], weights_and_biases["b_o_2"]
output_weights = weights_and_biases["output_weights"]
output_bias = weights_and_biases["output_bias"]

# Now you can access each variable like this:
"""
train_close = training_data['train_close']
train_volume = training_data['train_volume']
train_MA_5 = training_data['train_MA_5']
train_MA_10 = training_data['train_MA_10']
train_pct_change = training_data['train_pct_change']
train_RSI = training_data['train_RSI']
train_target = training_data['train_target']
"""

def sigmoid(x):
    return (1 / (1 + np.exp(-x)))

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=0, keepdims=True))  # for numerical stability
    return e_x / np.sum(e_x, axis=0, keepdims=True)

def one_layer(input_vector, cell_state, hidden_state, W_f, W_i, W_c, W_o, U_f, U_i, U_c, U_o, b_f, b_i, b_c, b_o):
    f_sum = W_f @ input_vector + U_f @ hidden_state + b_f
    f_gate = sigmoid(f_sum)
    
    i_sum = W_i @ input_vector + U_i @ hidden_state + b_i
    i_gate = sigmoid(i_sum)

    c_sum = W_c @ input_vector + U_c @ hidden_state + b_c
    c_gate = tanh(c_sum)

    o_sum = W_o @ input_vector + U_o @ hidden_state + b_o
    o_gate = sigmoid(o_sum)

    candidate = i_gate * c_gate
    cell_state = f_gate * cell_state + candidate
    hidden_state = o_gate * tanh(cell_state)
    return cell_state, hidden_state

def forward_pass(input_vector, cell_state_1, hidden_state_1, cell_state_2, hidden_state_2):
    cell_state_1 = np.zeros_like(cell_state_1)
    hidden_state_1 = np.zeros_like(hidden_state_1)
    cell_state_2 = np.zeros_like(cell_state_2)
    hidden_state_2 = np.zeros_like(hidden_state_2)
    for i in range(60):
        cell_state_1, hidden_state_1 = one_layer(input_vector[i], cell_state_1, hidden_state_1, W_f_1, W_i_1, W_c_1, W_o_1, U_f_1, U_i_1, U_c_1, U_o_1, b_f_1, b_i_1, b_c_1, b_o_1)
        cell_state_2, hidden_state_2 = one_layer(hidden_state_1, cell_state_2, hidden_state_2, W_f_2, W_i_2, W_c_2, W_o_2, U_f_2, U_i_2, U_c_2, U_o_2, b_f_2, b_i_2, b_c_2, b_o_2)
    output_vector = output_weights @ hidden_state_2 + output_bias
    normalised_output_vector = np.array(softmax(output_vector))
    print("Prediction confidence: ", max(normalised_output_vector))
    if np.argmax(normalised_output_vector) == 0:
        prediction = "Predicted: Large decrease (%<-8%)"
    if np.argmax(normalised_output_vector) == 1:
        prediction = "Predicted: Small decrease (-2%<%<-8%)"
    if np.argmax(normalised_output_vector) == 2:
        prediction = "Predicted: No significant change (-2%<%<2%)"
    if np.argmax(normalised_output_vector) == 3:
        prediction = "Predicted: Small increase (2%<%<8%)"
    if np.argmax(normalised_output_vector) == 4:
        prediction = "Predicted: Large increase (%>8%)"
    print(prediction)

cell_state_1 = np.zeros((128, 1))
hidden_state_1 = np.zeros((128, 1))
cell_state_2 = np.zeros((128, 1))
hidden_state_2 = np.zeros((128, 1))

data_vector = [[train_close[0], train_volume[0], train_MA_5[0], train_MA_10[0], train_pct_change[0], train_RSI[0]]]
data_vector = np.array(data_vector).transpose(2, 1, 0)

forward_pass(data_vector, cell_state_1, hidden_state_1, cell_state_2, hidden_state_2)

'''
actual = ""
if train_target[0] == 0:
    actual = "Actual: Large decrease (%<-8%)"
if train_target[0] == 1:
    actual = "Actual: Small decrease (-2%<%<-8%)"
if train_target[0] == 2:
    actual = "Actual: No significant change (-2%<%<2%)"
if train_target[0] == 3:
    actual = "Actual: Small increase (2%<%<8%)"
if train_target[0] == 4:
    actual = "Actual: Large increase (%>8%)"

print(actual)'''