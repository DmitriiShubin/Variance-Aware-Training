import numpy as np
import torch

# from pytorch_toolbelt import losses as L


class AdversarialScheduler:
    def __init__(
        self,
        delta=0.05,
        is_maximize=True
    ):

        self.delta = delta
        self.is_maximize=is_maximize
        self.status=False
        self.previous_score = None

    def __call__(self, score):

        if self.is_maximize:
            if self.previous_score is None:
                self.previous_score = score
            else:
                if  self.delta < score -self.previous_score:
                    self.status=True

        if self.is_maximize:
            if self.previous_score is None:
                self.previous_score = score
            else:
                if self.delta < self.previous_score - score:
                    self.status=True

    def get_status(self):
        return self.status
