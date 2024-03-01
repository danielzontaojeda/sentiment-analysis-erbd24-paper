from abc import ABC, abstractmethod
from utils import quantifier_utils
import pandas as pd


class Quantifier(ABC):
    def __init__(self):
        self.needs_dist_matching = ["hdy", "dys", "sord", "smm"]

    @abstractmethod
    def predict(self, X_test):
        pass

    def setTprFpr(self, X_train, y_train):
        validation_scores = pd.DataFrame(columns=["class", "score"])
        validation_scores["class"] = y_train
        validation_scores["score"] = X_train
        self.tprfpr = quantifier_utils.getTPRandFPRbyThreshold(validation_scores)

    def set_scores(self, train_test_scores):
        scores = {
            "scores": train_test_scores["X_train"],
            "class": train_test_scores["y_train"],
        }
        df = pd.DataFrame(scores)
        self.pos_scores = df.query("`class` == 1")
        self.neg_scores = df.query("`class` == 0")
