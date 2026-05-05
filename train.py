from data import get_training_data
import pandas as pd
import numpy as np
import pickle

# Get the training data
training_data = get_training_data()

# Now you can access each variable like this:
train_close = training_data['train_close']
train_volume = training_data['train_volume']
train_MA_5 = training_data['train_MA_5']
train_MA_10 = training_data['train_MA_10']
train_pct_change = training_data['train_pct_change']
train_RSI = training_data['train_RSI']
train_target = training_data['train_target']

def sigmoid(x):
    return (1 / (1 + np.exp(-x)))

def dsigmoid(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def dtanh(x):
    return 1 - np.tanh(x)**2

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=0, keepdims=True))  # for numerical stability
    return e_x / np.sum(e_x, axis=0, keepdims=True)


def xavier_uniform(shape, n_in, n_out):
    limit = np.sqrt(6 / (n_in + n_out))
    return np.random.uniform(-limit, limit, size=shape)

def orthogonal(shape):
    a = np.random.randn(*shape)
    q, _ = np.linalg.qr(a)
    return q.astype(np.float32)

def initialize_weights_and_biases(batch_size):
    W_f_1 = xavier_uniform((128, 6), 6, 128)
    W_i_1 = xavier_uniform((128, 6), 6, 128)
    W_c_1 = xavier_uniform((128, 6), 6, 128)
    W_o_1 = xavier_uniform((128, 6), 6, 128)
    W_f_2 = xavier_uniform((128, 128), 128, 128)
    W_i_2 = xavier_uniform((128, 128), 128, 128)
    W_c_2 = xavier_uniform((128, 128), 128, 128)
    W_o_2 = xavier_uniform((128, 128), 128, 128)
    U_f_1 = orthogonal((128, 128))
    U_i_1 = orthogonal((128, 128))
    U_c_1 = orthogonal((128, 128))
    U_o_1 = orthogonal((128, 128))
    U_f_2 = orthogonal((128, 128))
    U_i_2 = orthogonal((128, 128))
    U_c_2 = orthogonal((128, 128))
    U_o_2 = orthogonal((128, 128))
    b_f_1 = np.zeros((128, 1))
    b_i_1 = np.zeros((128, 1))
    b_c_1 = np.zeros((128, 1))
    b_o_1 = np.zeros((128, 1))
    b_f_2 = np.zeros((128, 1))
    b_i_2 = np.zeros((128, 1))
    b_c_2 = np.zeros((128, 1))
    b_o_2 = np.zeros((128, 1))


    output_weights = xavier_uniform((5, 128), 128, 5) #5x128
    output_bias = np.zeros((5, 1)) #5x1

    return {
        "W_f_1": W_f_1, "W_i_1": W_i_1, "W_c_1": W_c_1, "W_o_1": W_o_1,
        "W_f_2": W_f_2, "W_i_2": W_i_2, "W_c_2": W_c_2, "W_o_2": W_o_2,
        "U_f_1": U_f_1, "U_i_1": U_i_1, "U_c_1": U_c_1, "U_o_1": U_o_1,
        "U_f_2": U_f_2, "U_i_2": U_i_2, "U_c_2": U_c_2, "U_o_2": U_o_2,
        "b_f_1": b_f_1, "b_i_1": b_i_1, "b_c_1": b_c_1, "b_o_1": b_o_1,
        "b_f_2": b_f_2, "b_i_2": b_i_2, "b_c_2": b_c_2, "b_o_2": b_o_2,
        "output_weights": output_weights,
        "output_bias": output_bias
    }

def update_cache(t, cache, x, h1_prev, c1_prev, f1, i1, o1, c1_tilde, c1, h1, W_f1, W_i1, W_o1, W_c1, U_f1, U_i1, U_o1, U_c1, b_f1, b_i1, b_o1, b_c1, h2_prev, c2_prev, f2, i2, o2, c2_tilde, c2, h2, W_f2, W_i2, W_o2, W_c2, U_f2, U_i2, U_o2, U_c2, b_f2, b_i2, b_o2, b_c2):
    cache[t] = {
        'layer1': {
            'x': x,
            'h_prev': h1_prev,
            'c_prev': c1_prev,
            'f': f1,
            'i': i1,
            'o': o1,
            'c_tilde': c1_tilde,
            'c': c1,
            'h': h1,
            'W_f': W_f1,
            'W_i': W_i1,
            'W_o': W_o1,
            'W_c': W_c1,
            'U_f': U_f1,
            'U_i': U_i1,
            'U_o': U_o1,
            'U_c': U_c1,
            'b_f': b_f1,
            'b_1': b_i1,
            'b_o': b_o1,
            'b_c': b_c1,
        },
        'layer2': {
            'x': h1,
            'h_prev': h2_prev,
            'c_prev': c2_prev,
            'f': f2,
            'i': i2,
            'o': o2,
            'c_tilde': c2_tilde,
            'c': c2,
            'h': h2,
            'W_f': W_f2,
            'W_i': W_i2,
            'W_o': W_o2,
            'W_c': W_c2,
            'U_f': U_f2,
            'U_i': U_i2,
            'U_o': U_o2,
            'U_c': U_c2,
            'b_f': b_f2,
            'b_1': b_i2,
            'b_o': b_o2,
            'b_c': b_c2,
        }
    }

    return cache

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
    return cell_state, hidden_state, f_gate, i_gate, o_gate, c_gate

cache = {}

def forward_pass(input_vector, cell_state_1, hidden_state_1, cell_state_2, hidden_state_2):
    cell_state_1 = cell_state_1
    hidden_state_1 = hidden_state_1
    cell_state_2 = cell_state_2
    hidden_state_2 = hidden_state_2
    for i in range(60):
        prev_cell_state_1 = cell_state_1
        prev_hidden_state_1 = hidden_state_1
        prev_cell_state_2 = cell_state_2
        prev_hidden_state_2 = hidden_state_2
        cell_state_1, hidden_state_1, f_gate_1, i_gate_1, o_gate_1, c_gate_1 = one_layer(input_vector[i], cell_state_1, hidden_state_1, W_f_1, W_i_1, W_c_1, W_o_1, U_f_1, U_i_1, U_c_1, U_o_1, b_f_1, b_i_1, b_c_1, b_o_1)
        cell_state_2, hidden_state_2, f_gate_2, i_gate_2, o_gate_2, c_gate_2 = one_layer(hidden_state_1, cell_state_2, hidden_state_2, W_f_2, W_i_2, W_c_2, W_o_2, U_f_2, U_i_2, U_c_2, U_o_2, b_f_2, b_i_2, b_c_2, b_o_2)
        update_cache(i, cache, input_vector[i], prev_hidden_state_1, prev_cell_state_1, f_gate_1, i_gate_1, o_gate_1, c_gate_1, cell_state_1, hidden_state_1, W_f_1, W_i_1, W_o_1, W_c_1, U_f_1, U_i_1, U_o_1, U_c_1, b_f_1, b_i_1, b_o_1, b_c_1, prev_hidden_state_2, prev_cell_state_2, f_gate_2, i_gate_2, o_gate_2, c_gate_2, cell_state_2, hidden_state_2, W_f_2, W_i_2, W_o_2, W_c_2, U_f_2, U_i_2, U_o_2, U_c_2, b_f_2, b_i_2, b_o_2, b_c_2)
    output_vector = output_weights @ hidden_state_2 + output_bias
    normalised_output_vector = np.array(softmax(output_vector))
    return normalised_output_vector
    
def cross_entropy(predicted, actual):
    m = actual.shape[0]
    log_likelihood = -np.log(predicted[actual, np.arange(m)])
    loss = np.sum(log_likelihood) / m
    return loss

def accuracy(predictions, targets):
    pred_classes = np.argmax(predictions, axis=0)
    return np.mean(pred_classes == targets)

def lstm_backward_two_layers(dh2_next, dc2_next, dh1_next, dc1_next, cache):
    T = len(cache)
    l1 = cache[0]['layer1']
    l2 = cache[0]['layer2']
    grads = {
        'W_f_1': np.zeros_like(l1['W_f']),
        'W_i_1': np.zeros_like(l1['W_i']),
        'W_c_1': np.zeros_like(l1['W_c']),
        'W_o_1': np.zeros_like(l1['W_o']),
        'U_f_1': np.zeros_like(l1['U_f']),
        'U_i_1': np.zeros_like(l1['U_i']),
        'U_c_1': np.zeros_like(l1['U_c']),
        'U_o_1': np.zeros_like(l1['U_o']),
        'b_f_1': np.zeros_like(l1['b_f']),
        'b_i_1': np.zeros_like(l1['b_1']),
        'b_c_1': np.zeros_like(l1['b_c']),
        'b_o_1': np.zeros_like(l1['b_o']),
        'W_f_2': np.zeros_like(l2['W_f']),
        'W_i_2': np.zeros_like(l2['W_i']),
        'W_c_2': np.zeros_like(l2['W_c']),
        'W_o_2': np.zeros_like(l2['W_o']),
        'U_f_2': np.zeros_like(l2['U_f']),
        'U_i_2': np.zeros_like(l2['U_i']),
        'U_c_2': np.zeros_like(l2['U_c']),
        'U_o_2': np.zeros_like(l2['U_o']),
        'b_f_2': np.zeros_like(l2['b_f']),
        'b_i_2': np.zeros_like(l2['b_1']),
        'b_c_2': np.zeros_like(l2['b_c']),
        'b_o_2': np.zeros_like(l2['b_o']),
    }

    for t in reversed(range(T)):
        l1 = cache[t]['layer1']
        l2 = cache[t]['layer2']

        do2 = dh2_next * tanh(l2['c']) * dsigmoid(l2['o'])
        dc2 = dc2_next + dh2_next * l2['o'] * dtanh(l2['c'])
        di2 = dc2 * l2['c_tilde'] * dsigmoid(l2['i'])
        df2 = dc2 * l2['c_prev'] * dsigmoid(l2['f'])
        dc_tilde2 = dc2 * l2['i'] * dtanh(l2['c_tilde'])

        grads['W_f_2'] += df2 @ l2['x'].T
        grads['U_f_2'] += df2 @ l2['h_prev'].T
        grads['b_f_2'] += np.sum(df2, axis=1, keepdims=True)
        grads['W_i_2'] += di2 @ l2['x'].T
        grads['U_i_2'] += di2 @ l2['h_prev'].T
        grads['b_i_2'] += np.sum(di2, axis=1, keepdims=True)
        grads['W_o_2'] += do2 @ l2['x'].T
        grads['U_o_2'] += do2 @ l2['h_prev'].T
        grads['b_o_2'] += np.sum(do2, axis=1, keepdims=True)
        grads['W_c_2'] += dc_tilde2 @ l2['x'].T
        grads['U_c_2'] += dc_tilde2 @ l2['h_prev'].T
        grads['b_c_2'] += np.sum(dc_tilde2, axis=1, keepdims=True)

        dx2 = (l2['W_f'].T @ df2 + l2['W_i'].T @ di2 + l2['W_o'].T @ do2 + l2['W_c'].T @ dc_tilde2)
        dh1_next = dx2
        dc1_next = l1['c'] * df2

        do1 = dh1_next * tanh(l1['c']) * dsigmoid(l1['o'])
        dc1 = dc1_next + dh1_next * l1['o'] * dtanh(l1['c'])
        di1 = dc1 * l1['c_tilde'] * dsigmoid(l1['i'])
        df1 = dc1 * l1['c_prev'] * dsigmoid(l1['f'])
        dc_tilde1 = dc1 * l1['i'] * dtanh(l1['c_tilde'])

        grads['W_f_1'] += df1 @ l1['x'].T
        grads['U_f_1'] += df1 @ l1['h_prev'].T
        grads['b_f_1'] += np.sum(df1, axis=1, keepdims=True)
        grads['W_i_1'] += di1 @ l1['x'].T
        grads['U_i_1'] += di1 @ l1['h_prev'].T
        grads['b_i_1'] += np.sum(di1, axis=1, keepdims=True)
        grads['W_o_1'] += do1 @ l1['x'].T
        grads['U_o_1'] += do1 @ l1['h_prev'].T
        grads['b_o_1'] += np.sum(do1, axis=1, keepdims=True)
        grads['W_c_1'] += dc_tilde1 @ l1['x'].T
        grads['U_c_1'] += dc_tilde1 @ l1['h_prev'].T
        grads['b_c_1'] += np.sum(dc_tilde1, axis=1, keepdims=True)

    return grads


def update_parameters(parameters, gradients, learning_rate):
    for key in gradients:
        parameters[key] -= learning_rate * gradients[key]
    return parameters


batch_size = 32

# Initialize weights and biases
weights_and_biases = initialize_weights_and_biases(batch_size)

# Initialize weights and biases
W_f_1, W_i_1, W_c_1, W_o_1 = weights_and_biases["W_f_1"], weights_and_biases["W_i_1"], weights_and_biases["W_c_1"], weights_and_biases["W_o_1"]
W_f_2, W_i_2, W_c_2, W_o_2 = weights_and_biases["W_f_2"], weights_and_biases["W_i_2"], weights_and_biases["W_c_2"], weights_and_biases["W_o_2"]
U_f_1, U_i_1, U_c_1, U_o_1 = weights_and_biases["U_f_1"], weights_and_biases["U_i_1"], weights_and_biases["U_c_1"], weights_and_biases["U_o_1"]
U_f_2, U_i_2, U_c_2, U_o_2 = weights_and_biases["U_f_2"], weights_and_biases["U_i_2"], weights_and_biases["U_c_2"], weights_and_biases["U_o_2"]
b_f_1, b_i_1, b_c_1, b_o_1 = weights_and_biases["b_f_1"], weights_and_biases["b_i_1"], weights_and_biases["b_c_1"], weights_and_biases["b_o_1"]
b_f_2, b_i_2, b_c_2, b_o_2 = weights_and_biases["b_f_2"], weights_and_biases["b_i_2"], weights_and_biases["b_c_2"], weights_and_biases["b_o_2"]
output_weights = weights_and_biases["output_weights"]
output_bias = weights_and_biases["output_bias"]

data_vector = []
target_vector = []
for i in range(0, 137):
    data_vector.append(np.array([train_close[(batch_size*i):(batch_size*(i+1))], train_volume[(batch_size*i):(batch_size*(i+1))], train_MA_5[(batch_size*i):(batch_size*(i+1))], train_MA_10[(batch_size*i):(batch_size*(i+1))], train_pct_change[(batch_size*i):(batch_size*(i+1))], train_RSI[(batch_size*i):(batch_size*(i+1))]]).transpose(2, 0, 1))
    target_vector.append(train_target[(batch_size*i):(batch_size*(i+1))])

data_vector = np.array(data_vector)
target_vector = np.array(target_vector)

# Build full input and target data arrays
total_samples = train_close.shape[0] - 60
features = [train_close, train_volume, train_MA_5, train_MA_10, train_pct_change, train_RSI]

# Prepare sequences of shape (time_steps, features, batch_size)
batches = total_samples // batch_size
sequence_length = 60

num_epochs = 10
learning_rate = 0.05

for epoch in range(num_epochs):
    epoch_loss = 0
    epoch_acc = 0
    for b in range(batches):
        # Reset hidden and cell states
        cell_state_1 = np.zeros((128, batch_size))
        hidden_state_1 = np.zeros((128, batch_size))
        cell_state_2 = np.zeros((128, batch_size))
        hidden_state_2 = np.zeros((128, batch_size))

        input_seqs = data_vector[b]  # (60, 6, batch_size)
        target_vals = target_vector[b]  # (batch_size,)

        # Reset cache
        cache = {}

        # Use weights directly from dictionary
        pred = forward_pass(input_seqs, cell_state_1, hidden_state_1, cell_state_2, hidden_state_2)

        loss = cross_entropy(pred, target_vals)
        acc = accuracy(pred, target_vals)
        epoch_loss += loss
        epoch_acc += acc

        if b % 10 == 0 and b > 0 or b == 1:
            print(f"Iteration {b}, Avg Loss: {epoch_loss / b:.4f}, Avg Accuracy: {epoch_acc / b:.4f}")

        d_logits = pred.copy()
        d_logits[target_vals, np.arange(batch_size)] -= 1
        d_logits /= batch_size

        grad_output_weights = d_logits @ cache[59]['layer2']['h'].T
        grad_output_bias = np.sum(d_logits, axis=1, keepdims=True)
        weights_and_biases['output_weights'] -= learning_rate * grad_output_weights
        weights_and_biases['output_bias'] -= learning_rate * grad_output_bias

        dh2_next = weights_and_biases['output_weights'].T @ d_logits
        dc2_next = np.zeros_like(dh2_next)
        dh1_next = np.zeros_like(dh2_next)
        dc1_next = np.zeros_like(dh2_next)

        gradients = lstm_backward_two_layers(dh2_next, dc2_next, dh1_next, dc1_next, cache)
        weights_and_biases = update_parameters(weights_and_biases, gradients, learning_rate)

        W_f_1 = weights_and_biases["W_f_1"]
        W_i_1 = weights_and_biases["W_i_1"]
        W_c_1 = weights_and_biases["W_c_1"]
        W_o_1 = weights_and_biases["W_o_1"]
        W_f_2 = weights_and_biases["W_f_2"]
        W_i_2 = weights_and_biases["W_i_2"]
        W_c_2 = weights_and_biases["W_c_2"]
        W_o_2 = weights_and_biases["W_o_2"]
        U_f_1 = weights_and_biases["U_f_1"]
        U_i_1 = weights_and_biases["U_i_1"]
        U_c_1 = weights_and_biases["U_c_1"]
        U_o_1 = weights_and_biases["U_o_1"]
        U_f_2 = weights_and_biases["U_f_2"]
        U_i_2 = weights_and_biases["U_i_2"]
        U_c_2 = weights_and_biases["U_c_2"]
        U_o_2 = weights_and_biases["U_o_2"]
        b_f_2 = weights_and_biases["b_f_2"]
        b_i_2 = weights_and_biases["b_i_2"]
        b_c_2 = weights_and_biases["b_c_2"]
        b_o_2 = weights_and_biases["b_o_2"]
        b_f_1 = weights_and_biases["b_f_1"]
        b_i_1 = weights_and_biases["b_i_1"]
        b_c_1 = weights_and_biases["b_c_1"]
        b_o_1 = weights_and_biases["b_o_1"]


    print(f"Epoch {epoch + 1}, Avg Loss: {epoch_loss / 137:.4f}, Avg Accuracy: {epoch_acc / 137:.4f}")

with open("lstm_weights.pkl", "wb") as f:
    pickle.dump(weights_and_biases, f)
