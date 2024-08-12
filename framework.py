import numpy as np

class Linear:
    def __init__(self, input_dim, output_dim):
        self.weights = np.random.randn(input_dim, output_dim) * 0.01
        self.bias = np.zeros((1, output_dim))

    def forward(self, X):
        self.input = X
        return np.dot(X, self.weights) + self.bias

    def backward(self, g_output):
        g_input = np.dot(g_output, self.weights.T)
        self.g_weights = np.dot(self.input.T, g_output)
        self.g_bias = np.sum(g_output, axis=0, keepdims=True)
        return g_input

class ReLU:
    def forward(self, X):
        self.input = X
        return np.maximum(0, X)

    def backward(self, g_output):
        g_input = g_output * (self.input > 0)
        return g_input

class Softmax:
    def forward(self, X):
        exp_values = np.exp(X - np.max(X, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return self.output

    def backward(self, g_output):
        return g_output

class CrossEntropyLoss:
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-12, 1. - 1e-12)
        correct_confidences = y_pred_clipped[range(samples), y_true]
        return -np.mean(np.log(correct_confidences))

    def backward(self, y_pred, y_true):
        samples = len(y_pred)
        g_ = y_pred
        g_[range(samples), y_true] -= 1
        return g_ / samples

class SGD:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def step(self, layers):
        for layer in layers:
            if hasattr(layer, 'weights'):
                layer.weights -= self.learning_rate * layer.g_weights
                layer.bias -= self.learning_rate * layer.g_bias

class Model:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.optimizer = None

    def add_layer(self, layer):
        self.layers.append(layer)

    def compile(self, loss, optimizer):
        self.loss = loss
        self.optimizer = optimizer

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, g_output):
        for layer in reversed(self.layers):
            g_output = layer.backward(g_output)

    def train(self, X, y, epochs):
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss_value = self.loss.forward(y_pred, y)
            g_output = self.loss.backward(y_pred, y)
            self.backward(g_output)
            self.optimizer.step(self.layers)
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss_value:.4f}')

    def predict(self, X):
        return self.forward(X)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        loss_value = self.loss.forward(y_pred, y)
        accuracy = np.mean(np.argmax(y_pred, axis=1) == y)
        print(f'Loss: {loss_value:.4f}, Accuracy: {accuracy:.4f}')
        return loss_value, accuracy
