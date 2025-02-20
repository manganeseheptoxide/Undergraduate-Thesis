import numpy as np

class LossFunction:
    def __init__(self):
        self.loss = self.loss_calculation
        self.pseudo_residual = self.pseudo_residual_calculation

    def loss_calculation(self, actual, predicted):
        raise NotImplementedError("Loss function not implemented!")

    def pseudo_residual_calculation(self, actual, predicted):
        raise NotImplementedError("Pseudo-residual calculation not implemented!")
    

class SSR(LossFunction):
    def loss_calculation(self, actual, predicted):
        y_actual = np.array(actual)
        y_predicted = np.array(predicted)
        return np.sum((y_actual - y_predicted) ** 2)

    def pseudo_residual_calculation(self, actual, predicted):
        y_actual = np.array(actual)
        y_predicted = np.array(predicted)
        return 2*(y_actual - y_predicted)
