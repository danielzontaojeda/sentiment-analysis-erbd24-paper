from quantifiers.quantifier import Quantifier


class cc(Quantifier):
    def predict(self, X_test, **kwargs):
        count = len([i for i in X_test if i >= kwargs["threshold"]])
        pos_prop = round(count / len(X_test), 2)
        return pos_prop
