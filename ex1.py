import numpy as np

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
            for i in range(len(X_train)):
                X = X_train[i]
                y = y_train[i]
                
                y_pred = self.predict(X)
                update = self.lr * (y - y_pred)
                
                self.weights[1:] += update * X
                self.weights[0] += update

    def accuracy(self, X_test, y_test):
        return np.mean([self.predict(X) == y for X, y in zip(X_test, y_test)]) * 100

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

nand_accuracy = nand_perceptron.accuracy(nand_X, nand_y)
print(f"NAND Perceptron Accuracy: {nand_accuracy:.2f}%")

# Train and Test Perceptron for XOR
print("\nTraining Perceptron for XOR Gate...")
xor_perceptron = Perceptron(num_inputs=2, lr=0.1, max_epochs=1000)
xor_perceptron.train(xor_X, xor_y)

print("Testing XOR Perceptron...")
for X in xor_X:
    print(f"Input: {X}, Output: {xor_perceptron.predict(X)}")

xor_accuracy = xor_perceptron.accuracy(xor_X, xor_y)
print(f"XOR Perceptron Accuracy: {xor_accuracy:.2f}%")


#output of the program is:-
# Training Perceptron for NAND Gate...
# Testing NAND Perceptron...
# Input: [0 0], Output: 1
# Input: [0 1], Output: 1
# Input: [1 0], Output: 1
# Input: [1 1], Output: 0
# NAND Perceptron Accuracy: 100.00%

# Training Perceptron for XOR Gate...
# Testing XOR Perceptron...
# Input: [0 0], Output: 1
# Input: [0 1], Output: 1
# Input: [1 0], Output: 0
# Input: [1 1], Output: 0
# XOR Perceptron Accuracy: 50.00%

