import numpy as np
import torch

# from pytorch_toolbelt import losses as L


class AdversarialScheduler:
    def __init__(
        self,
        score_plat,
        is_maximize=True
    ):

        self.score_plat = score_plat
        self.is_maximize=is_maximize
        self.status=False

    def __call__(self, score):

        if self.is_maximize:
            if score >= self.score_plat:
                self.status = True
        else:
            if score <= self.score_plat:
                self.status = True

    def get_status(self):
        return self.status
