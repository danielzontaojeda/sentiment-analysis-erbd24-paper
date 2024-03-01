import numpy as np

from quantifiers.quantifier import Quantifier


class MAX(Quantifier):
    def predict(self, scores, **kwargs):
        diff_tpr_fpr = list(abs(self.tprfpr["tpr"] - self.tprfpr["fpr"]))
        max_index = diff_tpr_fpr.index(max(diff_tpr_fpr))
        threshold, fpr, tpr = self.tprfpr.loc[max_index]
        class_prop = np.mean(scores >= threshold)
        if (tpr - fpr) == 0:
            pos_prop = class_prop
        else:
            pos_prop = (class_prop - fpr) / (tpr - fpr)

        # Clamp pos_prop between 0 and 1
        pos_prop = max(0, min(1, pos_prop))

        return pos_prop
