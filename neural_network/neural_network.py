import numpy as np
from neural_network.activation import ActivationFunctions
from neural_network.loss import LossFunctions

class NeuralNetwork:
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate=0.5):
        self.learning_rate = learning_rate

        # Inicjalizacja wag i biasów
        np.random.seed(42)
        self.W1 = np.random.randn(input_dim, hidden_dim)
        self.b1 = np.random.randn(1, hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim)
        self.b2 = np.random.randn(1, output_dim)

    def forward(self, X):
        """Propagacja w przód."""
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = ActivationFunctions.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = ActivationFunctions.sigmoid(self.z2)
        return self.a2

    def backward(self, X, y):
        """Propagacja wstecz."""
        dz2 = self.a2 - y
        dW2 = np.dot(self.a1.T, dz2 * ActivationFunctions.sigmoid_derivative(self.a2))
        db2 = np.sum(dz2 * ActivationFunctions.sigmoid_derivative(self.a2), axis=0, keepdims=True)

        dz1 = np.dot(dz2 * ActivationFunctions.sigmoid_derivative(self.a2), self.W2.T) * ActivationFunctions.sigmoid_derivative(self.a1)
        dW1 = np.dot(X.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # Aktualizacja wag i biasów
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2

    def train(self, X, y, epochs=10000):
        """Trenuje sieć przez podaną liczbę epok."""
        for epoch in range(epochs):
            predictions = self.forward(X)
            self.backward(X, y)

            if epoch % 1000 == 0:
                loss = LossFunctions.mean_squared_error(y, predictions)
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict(self, X):
        """Zwraca prognozę sieci dla danych wejściowych."""
        return self.forward(X)
git