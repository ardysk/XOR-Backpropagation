from neural_network.neural_network import NeuralNetwork
import numpy as np

# Dane wejściowe XOR
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

# Wyniki dla XOR
y = np.array([[0], [1], [1], [0]])

# Tworzenie sieci neuronowej
nn = NeuralNetwork(input_dim=2, hidden_dim=3, output_dim=1)

# Trenowanie sieci
nn.train(X, y, epochs=10000)

# Testowanie sieci
print("\nFinal predictions:")
for i in range(len(X)):
    pred = nn.predict(X[i].reshape(1, -1))
    print(f"Input: {X[i]}, Predicted: {pred[0][0]:.4f}, Actual: {y[i][0]}")
