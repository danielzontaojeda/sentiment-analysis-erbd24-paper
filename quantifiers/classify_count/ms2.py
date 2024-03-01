import numpy as np

from quantifiers.quantifier import Quantifier
from utils import quantifier_utils


class ms2(Quantifier):
    def predict(self, scores, **kwargs):
        index = np.where(
            abs(self.tprfpr['tpr'] - self.tprfpr['fpr']) > (1/4))[0].tolist()
        if index == 0:
            index = np.where(
                abs(self.tprfpr['tpr'] - self.tprfpr['fpr']) >= 0)[0].tolist()

        prevalances_array = []
        for i in index:

            threshold, fpr, tpr = self.tprfpr.loc[i]
            estimated_positive_ratio = len(
                np.where(scores >= threshold)[0])/len(scores)

            diff_tpr_fpr = abs(float(tpr-fpr))

            if diff_tpr_fpr == 0.0:
                diff_tpr_fpr = 1

            final_prevalence = abs(estimated_positive_ratio - fpr)/diff_tpr_fpr

            prevalances_array.append(final_prevalence)

        pos_prop = np.median(prevalances_array)
        pos_prop = max(0, min(1, pos_prop))
        return pos_prop
