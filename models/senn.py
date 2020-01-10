import torch
import torch.nn as nn
import torch.nn.functional as F


class SENN(nn.Module):
    """
    """

    def __init__(self, conceptizer, parameterizer, aggregator):
        self.conceptizer = conceptizer
        self.parameterizer = parameterizer
        self.aggregator = aggregator

    def forward(self, x):
        concepts, recon_x = self.conceptizer(x)
        parameters = self.parameterizer(x)
        predictions = self.aggregator(concepts, parameters)
        explanations = (concepts, parameters)
        return predictions, explanations 

    def get_concepts(self, x):
        concepts, recon_x = self.conceptizer(x)


    
        