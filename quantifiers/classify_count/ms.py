import numpy as np

from quantifiers.quantifier import Quantifier
from utils import quantifier_utils


class ms(Quantifier):
    def predict(self, scores, **kwargs):
        unique_scores = np.arange(0.01, 1, 0.01)
        prevalences_array = []

        for threshold in unique_scores:
            class_prop = np.mean(scores >= threshold)
            record = quantifier_utils.find_tprfpr_by_threshold(
                self.tprfpr, round(threshold, 2)
            )
            tpr_minus_fpr = record["tpr"] - record["fpr"]

            if tpr_minus_fpr == 0:
                prevalences_array.append(class_prop)
            else:
                final_prevalence = (class_prop - record["fpr"]) / tpr_minus_fpr
                prevalences_array.append(final_prevalence)

        pos_prop = np.median(prevalences_array)

        # Clamp pos_prop between 0 and 1
        pos_prop = max(0, min(1, pos_prop))
        return pos_prop
