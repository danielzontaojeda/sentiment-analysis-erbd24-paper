import numpy as np

from quantifiers.quantifier import Quantifier


class sord(Quantifier):
    def predict(self, test_scores, **kwargs):
        alpha = np.linspace(0, 1, 101)
        ts = test_scores
        vDist = []
        for k in alpha:
            pos = np.array(self.pos_scores["scores"])
            neg = np.array(self.neg_scores["scores"])
            test = np.array(ts)
            pos_prop = k

            p_w = pos_prop / len(pos)
            n_w = (1 - pos_prop) / len(neg)
            t_w = -1 / len(test)

            p = [(x, p_w) for x in pos]
            n = [(x, n_w) for x in neg]
            t = [(x, t_w) for x in test]

            v = sorted(p + n + t, key=lambda x: x[0])

            acc = v[0][1]
            total_cost = 0

            for i in range(1, len(v)):
                cost_mul = v[i][0] - v[i - 1][0]
                total_cost = total_cost + abs(cost_mul * acc)
                acc = acc + v[i][1]
            vDist.append(total_cost)
        if all(np.isnan(x) for x in vDist):
            return np.nan
        pos_prop = alpha[vDist.index(min(vDist))]
        return pos_prop
