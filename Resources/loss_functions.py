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

import numpy as np

class QuantileLoss(LossFunction):
    def __init__(self, tau=0.9):
        super().__init__()
        self.tau = tau  # Quantile level (0 < tau < 1)
    
    def loss_calculation(self, actual, predicted):
        y_actual = np.array(actual)
        y_predicted = np.array(predicted)
        residual = y_actual - y_predicted
        return np.sum(np.maximum(self.tau * residual, (self.tau - 1) * residual))

    def pseudo_residual_calculation(self, actual, predicted):
        y_actual = np.array(actual)
        y_predicted = np.array(predicted)
        return np.where(y_actual > y_predicted, self.tau, self.tau - 1)

class LogCoshLoss(LossFunction):
    def loss_calculation(self, actual, predicted):
        y_actual = np.array(actual)
        y_predicted = np.array(predicted)
        return np.sum(np.log(np.cosh(y_actual - y_predicted)))

    def pseudo_residual_calculation(self, actual, predicted):
        y_actual = np.array(actual)
        y_predicted = np.array(predicted)
        return np.tanh(y_actual - y_predicted)

class HuberLoss(LossFunction):
    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta  # Threshold for switching between MSE and MAE

    def loss_calculation(self, actual, predicted):
        y_actual = np.array(actual)
        y_predicted = np.array(predicted)
        residual = y_actual - y_predicted
        mask = np.abs(residual) <= self.delta
        return np.sum(np.where(mask, 0.5 * residual**2, self.delta * (np.abs(residual) - 0.5 * self.delta)))

    def pseudo_residual_calculation(self, actual, predicted):
        y_actual = np.array(actual)
        y_predicted = np.array(predicted)
        residual = y_actual - y_predicted
        return np.where(np.abs(residual) <= self.delta, residual, self.delta * np.sign(residual))

class AsymmetricHuberLoss(LossFunction):
    def __init__(self, delta=1.0, alpha=3.0):  # alpha > 1 penalizes underestimation more
        super().__init__()
        self.delta = delta
        self.alpha = alpha  

    def loss_calculation(self, actual, predicted):
        y_actual = np.array(actual)
        y_predicted = np.array(predicted)
        residual = y_actual - y_predicted
        mask = np.abs(residual) <= self.delta
        return np.sum(
            np.where(mask, 0.5 * residual**2, 
                     np.where(residual > 0, self.alpha * self.delta * (np.abs(residual) - 0.5 * self.delta),
                              self.delta * (np.abs(residual) - 0.5 * self.delta)))
        )

    def pseudo_residual_calculation(self, actual, predicted):
        y_actual = np.array(actual)
        y_predicted = np.array(predicted)
        residual = y_actual - y_predicted
        return np.where(np.abs(residual) <= self.delta, residual, 
                        np.where(residual > 0, self.alpha * self.delta * np.sign(residual),
                                 self.delta * np.sign(residual)))

class UpperBoundLogCoshLoss(LossFunction):
    def __init__(self, scale_factor=1.5):  # Scale up negative residuals
        super().__init__()
        self.scale_factor = scale_factor

    def loss_calculation(self, actual, predicted):
        y_actual = np.array(actual)
        y_predicted = np.array(predicted)
        residual = y_actual - y_predicted
        return np.sum(np.log(np.cosh(residual)))

    def pseudo_residual_calculation(self, actual, predicted):
        y_actual = np.array(actual)
        y_predicted = np.array(predicted)
        residual = y_actual - y_predicted
        return np.where(residual < 0, self.scale_factor * np.tanh(residual), np.tanh(residual))

class SmoothQuantileLoss(LossFunction):
    def __init__(self, tau=0.9, epsilon=1e-3):  # Smooth pinball loss
        super().__init__()
        self.tau = tau
        self.epsilon = epsilon

    def loss_calculation(self, actual, predicted):
        y_actual = np.array(actual)
        y_predicted = np.array(predicted)
        residual = y_actual - y_predicted
        return np.sum(np.where(np.abs(residual) < self.epsilon, 
                               0.5 * residual**2 / self.epsilon, 
                               np.maximum(self.tau * residual, (self.tau - 1) * residual)))

    def pseudo_residual_calculation(self, actual, predicted):
        y_actual = np.array(actual)
        y_predicted = np.array(predicted)
        residual = y_actual - y_predicted
        return np.where(np.abs(residual) < self.epsilon, residual / self.epsilon, 
                        np.where(residual > 0, self.tau, self.tau - 1))
