import unittest
import numpy as np
from neural_network.neural_network import NeuralNetwork

class TestNeuralNetwork(unittest.TestCase):
    def setUp(self):
        self.nn = NeuralNetwork(input_dim=2, hidden_dim=3, output_dim=1)
        self.X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        self.y = np.array([[0], [1], [1], [0]])

    def test_forward_shape(self):
        predictions = self.nn.forward(self.X)
        self.assertEqual(predictions.shape, (4, 1))

    def test_training_reduces_loss(self):
        initial_loss = self.nn.train(self.X, self.y, epochs=1)
        self.nn.train(self.X, self.y, epochs=10)
        final_loss = self.nn.train(self.X, self.y, epochs=1)
        self.assertLess(final_loss, initial_loss)

if __name__ == "__main__":
    unittest.main()
