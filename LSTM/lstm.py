import numpy as np
import tensorflow as tf

inputs = np.random.random((32, 10, 8))


class LSTMCell:
    def __init__(self, input_size, hidden_size) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Initialize weights
        self.W_i = np.random.randn(input_size, hidden_size)
        self.W_f = np.random.randn(input_size, hidden_size)
        self.W_o = np.random.randn(input_size, hidden_size)
        self.W_c = np.random.randn(input_size, hidden_size)

        self.U_i = np.random.randn(hidden_size, hidden_size)
        self.U_f = np.random.randn(hidden_size, hidden_size)
        self.U_o = np.random.randn(hidden_size, hidden_size)
        self.U_c = np.random.randn(hidden_size, hidden_size)

        # Initialize biases
        self.b_i = np.zeros(hidden_size)
        self.b_f = np.zeros(hidden_size)
        self.b_o = np.zeros(hidden_size)
        self.b_c = np.zeros(hidden_size)

        # Initialize cell state and hidden state
        self.cell_state = np.zeros(hidden_size)
        self.hidden_state = np.zeros(hidden_size)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)

    def forward(self, x):
        i = self.sigmoid(
            np.dot(x, self.W_i) + np.dot(self.hidden_state, self.U_i) + self.b_i
        )
        f = self.sigmoid(
            np.dot(x, self.W_f) + np.dot(self.hidden_state, self.U_f) + self.b_f
        )
        o = self.sigmoid(
            np.dot(x, self.W_o) + np.dot(self.hidden_state, self.U_o) + self.b_o
        )
        c_hat = self.tanh(
            np.dot(x, self.W_c) + np.dot(self.hidden_state, self.U_c) + self.b_c
        )
        self.cell_state = f * self.cell_state + i * c_hat
        self.hidden_state = o * self.tanh(self.cell_state)
        return self.hidden_state[:, -1, :]


custom_lstm_cell = LSTMCell(input_size=8, hidden_size=4)
custom_output = custom_lstm_cell.forward(inputs)
lstm_layer = tf.keras.layers.LSTM(units=4)
keras_output = lstm_layer(inputs)
print("Custom LSTM output shape:", custom_output.shape)
print("Keras LSTM output shape:", keras_output.shape)
if np.allclose(custom_output, keras_output, rtol=1e-5, atol=1e-5):
    print("Outputs are approximately equal.")
else:
    print("Outputs are not equal.")
