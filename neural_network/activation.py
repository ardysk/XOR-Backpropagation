import numpy as np

class ActivationFunctions:
    """Klasa dla funkcji aktywacji."""

    @staticmethod
    def sigmoid(x):
        """Funkcja sigmoid."""
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        """Pochodna funkcji sigmoid."""
        return x * (1 - x)
