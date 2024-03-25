import numpy as np

from quantifiers.quantifier import Quantifier
from utils import quantifier_utils


class ms(Quantifier):
    def predict(self, scores, **kwargs):
        pass
        # max_it = 1        # Max num of iterations
        # eps = 1e-6           # Small constant for stopping criterium

        # m = scores.shape[0]
        # p_tr = class_dist(train_labels, nclasses)[0]
        # p_s = np.copy(p_tr)
        # p_cond_tr = np.array(scores)
        # p_cond_s = np.zeros(p_cond_tr.shape)

        # for it in range(max_it):
        # 	r = p_s / p_tr
        # 	p_cond_s = p_cond_tr * r
        # 	#s = np.sum(p_cond_s, axis = 1)
        # 	s = np.sum(p_cond_s, axis = 0)
        # 	for c in range(m):
        # 		#p_cond_s[:,c] = p_cond_s[:,c] / s
        # 		p_cond_s[c] = p_cond_s[c] / s
        # 	p_s_old = np.copy(p_s)
        # 	p_s = np.sum(p_cond_s, axis = 0) / p_cond_s.shape[0]
        # 	if (np.sum(np.abs(p_s - p_s_old)) < eps):
        # 		break

        # return(p_s/np.sum(p_s))
