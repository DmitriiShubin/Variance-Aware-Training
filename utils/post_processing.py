import numpy as np


class Post_Processing:
    def run(self, pred):
        pred = np.round(pred, decimals=3)
        pred = np.argmax(pred, axis=-1)
        return pred
