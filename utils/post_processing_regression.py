import numpy as np


class Post_Processing:
    def run(self, pred: np.array):

        pred = np.round(pred,0)
        return pred
