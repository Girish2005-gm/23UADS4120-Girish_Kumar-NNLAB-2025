import numpy as np
class MultiLayerPerceptron:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1, epochs=10000):
        # Initialize weights and biases for input-to-hidden and hidden-to-output layers
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) * 0.01
        self.bias_hidden = np.zeros(hidden_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size) * 0.01
        self.bias_output = np.zeros(output_size)
        self.learning_rate = learning_rate
        self.epochs = epochs

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def predict(self, X):
        # Forward pass
        hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        hidden_output = self.sigmoid(hidden_input)
        final_input = np.dot(hidden_output, self.weights_hidden_output) + self.bias_output
        final_output = self.sigmoid(final_input)
        return final_output

    def fit(self, X, y):
        for epoch in range(self.epochs):
            # Forward pass
            hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
            hidden_output = self.sigmoid(hidden_input)
            final_input = np.dot(hidden_output, self.weights_hidden_output) + self.bias_output
            final_output = self.sigmoid(final_input)

            # Compute errors
            output_error = y - final_output
            output_delta = output_error * self.sigmoid_derivative(final_output)

            hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
            hidden_delta = hidden_error * self.sigmoid_derivative(hidden_output)

            # Update weights and biases
            self.weights_hidden_output += self.learning_rate * np.dot(hidden_output.T, output_delta)
            self.bias_output += self.learning_rate * np.sum(output_delta, axis=0)
            self.weights_input_hidden += self.learning_rate * np.dot(X.T, hidden_delta)
            self.bias_hidden += self.learning_rate * np.sum(hidden_delta, axis=0)

            # Print loss every 1000 epochs
            if epoch % 1000 == 0:
                loss = np.mean(np.square(output_error))
                print(f"Epoch {epoch}, Loss: {loss:.6f}")

    def evaluate(self, X, y):
        predictions = self.predict(X)
        predictions = np.round(predictions)
        accuracy = np.mean(predictions == y)
        return accuracy

# XOR dataset
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([[0], [1], [1], [0]])

# Train MLP for XOR function
mlp = MultiLayerPerceptron(input_size=2, hidden_size=2, output_size=1, learning_rate=0.1, epochs=10000)
mlp.fit(X_xor, y_xor)

# Evaluate the MLP
accuracy = mlp.evaluate(X_xor, y_xor)
print(f"XOR Accuracy: {accuracy * 100:.2f}%")

# Display final predictions
predictions = np.round(mlp.predict(X_xor))
print("Predictions for XOR function:")
for input_val, pred in zip(X_xor, predictions):
    print(f"Input: {input_val}, Predicted Output: {int(pred[0])}")
