import numpy as np

from quantifiers.quantifier import Quantifier


class t50(Quantifier):
    def predict(self, scores, **kwargs):
        index = np.abs(self.tprfpr["tpr"] - kwargs["threshold"]).idxmin()
        threshold, fpr, tpr = self.tprfpr.loc[index]
        class_prop = np.mean(scores >= threshold)
        if (tpr - fpr) == 0:
            pos_prop = class_prop
        else:
            pos_prop = (class_prop - fpr) / (tpr - fpr)

        # Clamp pos_prop between 0 and 1
        pos_prop = max(0, min(1, pos_prop))
        return pos_prop
