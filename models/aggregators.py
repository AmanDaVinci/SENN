import torch
import torch.nn as nn
import torch.nn.functional as F


class SumAggregator(nn.Module):
    def __init__(self, num_classes, **kwargs):
        """Basic Sum Aggregator that joins the concepts and relevances by summing their products.
        """
        super().__init__()
        self.num_classes = num_classes

    def forward(self, concepts, relevances):
        """Forward pass of Sum Aggregator.

        Aggregates concepts and relevances and returns the predictions for each class.

        Parameters
        ----------
        concepts : torch.Tensor
            Contains the output of the conceptizer with shape (BATCH, NUM_CONCEPTS, DIM_CONCEPT=1).
        relevances : torch.Tensor
            Contains the output of the parameterizer with shape (BATCH, NUM_CONCEPTS, NUM_CLASSES).

        Returns
        -------
        class_predictions : torch.Tensor
            Predictions for each class. Shape - (BATCH, NUM_CLASSES)
            
        """
        aggregated = torch.bmm(relevances.permute(0, 2, 1), concepts).squeeze(-1)
        return F.log_softmax(aggregated, dim=1)
