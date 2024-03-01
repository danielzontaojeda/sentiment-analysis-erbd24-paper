import numpy as np
import pandas as pd

from quantifiers.quantifier import Quantifier
from utils import quantifier_utils


class hdy(Quantifier):
    def predict(self, test_scores, **kwargs):
        bin_size = np.linspace(10, 110, 11)
        alpha_values = np.linspace(0, 1, 101)
        result = []
        for bins in bin_size:
            p_bin_count = quantifier_utils.get_hist(self.pos_scores, bins)
            n_bin_count = quantifier_utils.get_hist(self.neg_scores, bins)
            te_bin_count = quantifier_utils.get_hist(test_scores, bins)
            vDist = []
            for x in alpha_values:
                x = np.round(x, 2)
                vDist.append(
                    quantifier_utils.DyS_distance(
                        ((p_bin_count * x) + (n_bin_count * (1 - x))),
                        te_bin_count,
                        measure=kwargs["measure"],
                    )
                )

        result.append(alpha_values[np.argmin(vDist)])
        pos_prop = np.median(result)

        return pos_prop

    def set_scores(self, train_test_scores):
        scores = {
            "scores": train_test_scores["X_train"],
            "class": train_test_scores["y_train"],
        }
        df = pd.DataFrame(scores)
        self.pos_scores = df.query("`class`== 1")
        self.neg_scores = df.query("`class` == 0")
