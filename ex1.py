# Import Required Libraries
import numpy as np

# Define Perceptron Class
class Perceptron:
    def __init__(self, num_inputs, lr=0.1, max_epochs=1000):
        self.weights = np.zeros(num_inputs + 1)  # Including bias
        self.lr = lr
        self.max_epochs = max_epochs

    def activation_fn(self, x):
        return 1 if x >= 0 else 0

    def predict(self, X):
        weighted_sum = np.dot(X, self.weights[1:]) + self.weights[0]
        return self.activation_fn(weighted_sum)

    def train(self, X_train, y_train):
        for _ in range(self.max_epochs):
            for X, y in zip(X_train, y_train):
                y_pred = self.predict(X)
                update = self.lr * (y - y_pred)
                self.weights[1:] += update * X
                self.weights[0] += update

# NAND and XOR Truth Tables
nand_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
nand_y = np.array([1, 1, 1, 0])

xor_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
xor_y = np.array([0, 1, 1, 0])

# Train and Test Perceptron for NAND
print("Training Perceptron for NAND Gate...")
nand_perceptron = Perceptron(num_inputs=2)
nand_perceptron.train(nand_X, nand_y)

print("Testing NAND Perceptron...")
for X in nand_X:
    print(f"Input: {X}, Output: {nand_perceptron.predict(X)}")

# Train and Test Perceptron for XOR
print("\nTraining Perceptron for XOR Gate...")
xor_perceptron = Perceptron(num_inputs=2)
xor_perceptron.train(xor_X, xor_y)

print("Testing XOR Perceptron...")
for X in xor_X:
    print(f"Input: {X}, Output: {xor_perceptron.predict(X)}")
