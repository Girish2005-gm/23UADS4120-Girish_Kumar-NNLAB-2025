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

    def train(self, X_train, y_train):  # Correctly indented inside the class
        for _ in range(self.max_epochs):  # Loop through epochs
            for i in range(len(X_train)):  # Iterate using index
                X = X_train[i]  # Get the input example
                y = y_train[i]  # Get the expected output
                
                y_pred = self.predict(X)  # Get perceptron prediction
                update = self.lr * (y - y_pred)  # Compute update value
                
                self.weights[1:] += update * X  # Update weights
                self.weights[0] += update  # Update bias

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
