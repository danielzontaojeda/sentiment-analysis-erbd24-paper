import numpy as np

from quantifiers.quantifier import Quantifier


class pcc(Quantifier):
    def setTprFpr(self, X_train, y_train):
        pass

    def predict(self, X_test, **kwargs):
        # TODO: Calibrar
        calibrated_predictions = X_test
        pos_prop = np.mean(calibrated_predictions)

        return pos_prop
