import numpy as np

from quantifiers.quantifier import Quantifier


class x(Quantifier):
    def predict(self, scores, **kwargs):
        absolute_diff = np.abs((1 - self.tprfpr["tpr"]) - self.tprfpr["fpr"])
        min_index = absolute_diff.idxmin()

        threshold, fpr, tpr = self.tprfpr.iloc[min_index]
        class_prop = (scores >= threshold).sum() / len(scores)

        if (tpr - fpr) == 0:
            pos_prop = class_prop
        else:
            pos_prop = (class_prop - fpr) / (tpr - fpr)

        # Clamp pos_prop between 0 and 1
        pos_prop = max(0, min(1, pos_prop))

        return pos_prop
