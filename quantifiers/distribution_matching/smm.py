import numpy as np

from quantifiers.quantifier import Quantifier
from utils import quantifier_utils


class smm(Quantifier):
    def predict(self, test_scores, **kwargs):
        mean_pos_scr = np.mean(self.pos_scores)
        mean_neg_scr = np.mean(self.neg_scores)
        mean_te_scr = np.mean(test_scores)
        alpha = (mean_te_scr - mean_neg_scr) / (mean_pos_scr - mean_neg_scr)
        pos_prop = max(0, min(1, alpha))
        return pos_prop
