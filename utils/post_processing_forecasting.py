import numpy as np


class Post_Processing:
    def run(self, pred: np.array, measure_baseline: np.array):

        measure_baseline = measure_baseline * 6000
        pred = pred + measure_baseline

        pred = np.round(pred, decimals=3)
        return pred
