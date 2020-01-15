import torch
import torch.nn as nn
import torch.nn.functional as F

class SumAggregator(nn.Module):
    def __init__(self, **kwargs):
        """Basic Sum Aggregator that joins the concepts and relevances by summing their products.
        """
        super().__init__()

    def forward(self, concepts, relevances, num_concepts, num_classes):
        """Forward pass of Sum Aggregator.

        Aggregates concepts and relevances and returns the predictions for each class.

        Parameters
        ----------
        concepts : torch.Tensor
            Contains the output of the conceptizer with shape (BATCH, NUM_CONCEPTS, DIM_CONCEPT=1).
        relevances : torch.Tensor
            Contains the output of the parameterizer with shape (BATCH, NUM_CONCEPTS * NUM_CLASSES).
        num_concepts : int
            Number of concepts encoded by the Conceptizer.
        num_classes : int
            Number of output classes of the classifier.

        Returns
        -------
        class_predictions : torch.Tensor
            Predictions for each class.

        TODO add assertions for matching dimensions, maybe?
        """
        relevances = relevances.view(-1, num_classes, num_concepts)

        aggregated = torch.bmm(relevances, concepts).squeeze(-1)

        if num_classes == 1:
            class_predictions = torch.sigmoid(aggregated)
        else:
            class_predictions = F.log_softmax(aggregated)
        return class_predictions