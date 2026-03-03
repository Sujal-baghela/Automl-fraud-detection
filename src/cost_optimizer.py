import numpy as np
from sklearn.metrics import confusion_matrix


class BusinessCostOptimizer:
    def __init__(self, fraud_loss=10000, false_alarm_cost=200):
        """
        fraud_loss      : Cost of missing a fraud (False Negative)
        false_alarm_cost: Cost of wrongly flagging a legit transaction (False Positive)

        Default asymmetry: missing fraud is 50x more costly than a false alarm.
        This reflects real financial system priorities.
        """
        self.fraud_loss = fraud_loss
        self.false_alarm_cost = false_alarm_cost
        self.optimal_threshold = 0.5
        self.minimum_cost = float("inf")
        self.best_metrics = {}

    def optimize(self, y_true, y_proba):
        """
        Search threshold space to minimize total business cost.

        Total Cost = (FN * fraud_loss) + (FP * false_alarm_cost)
        """
        thresholds = np.unique(np.concatenate([[0.0], np.sort(y_proba), [1.0]]))

        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)

            # FIX: guard against edge-case thresholds where all predictions
            # are a single class, causing confusion_matrix to return a 1x1 matrix
            # which makes ravel() unpack incorrectly
            cm = confusion_matrix(y_true, y_pred)

            if cm.shape != (2, 2):
                continue

            tn, fp, fn, tp = cm.ravel()

            total_cost = (fn * self.fraud_loss) + (fp * self.false_alarm_cost)

            if total_cost < self.minimum_cost:
                self.minimum_cost = total_cost
                self.optimal_threshold = threshold
                self.best_metrics = {
                    "TP": int(tp),
                    "TN": int(tn),
                    "FP": int(fp),
                    "FN": int(fn),
                    "Total Cost ($)": round(float(total_cost), 2),
                    "Fraud Caught (Recall)": round(float(tp / (tp + fn + 1e-8)), 4),
                    "Precision": round(float(tp / (tp + fp + 1e-8)), 4)
                }

        if not self.best_metrics:
            raise ValueError(
                "Could not find a valid threshold. Check that y_true contains both classes."
            )

        return self.optimal_threshold

    def get_results(self):
        return {
            "Optimal Threshold": round(float(self.optimal_threshold), 5),
            "Minimum Cost ($)": round(float(self.minimum_cost), 2),
            "Metrics at Optimal Threshold": self.best_metrics
        }
