import numpy as np
from sklearn.metrics import precision_recall_curve


class ThresholdOptimizer:
    def __init__(self, strategy="maximize_f1"):
        """
        strategy options:
        - maximize_f1
        - maximize_recall
        - maximize_precision
        """
        self.strategy = strategy
        self.optimal_threshold = 0.5
        self.results = {}

    def optimize(self, y_true, y_proba):
        """
        Compute optimal threshold based on selected strategy.
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)

        # precision and recall have length n+1
        # thresholds has length n
        # So we remove last precision/recall value to match thresholds length
        precision = precision[:-1]
        recall = recall[:-1]

        # -------- Compute F1 --------
        f1_scores = (2 * precision * recall) / (precision + recall + 1e-8)

        # -------- Store all strategies --------
        self.results["maximize_f1"] = thresholds[np.argmax(f1_scores)]
        self.results["maximize_recall"] = thresholds[np.argmax(recall)]
        self.results["maximize_precision"] = thresholds[np.argmax(precision)]

        # -------- Select chosen strategy --------
        if self.strategy not in self.results:
            raise ValueError(f"Invalid strategy: {self.strategy}")

        self.optimal_threshold = self.results[self.strategy]

        return self.optimal_threshold

    def get_all_strategies(self):
        return self.results
