import numpy as np

class LossFunctions:
    """Klasa dla funkcji kosztu."""

    @staticmethod
    def mean_squared_error(y_true, y_pred):
        """Średni błąd kwadratowy."""
        return np.mean((y_true - y_pred) ** 2)

    @staticmethod
    def mean_squared_error_derivative(y_true, y_pred):
        """Pochodna średniego błędu kwadratowego."""
        return 2 * (y_pred - y_true) / y_true.size
