import torch
import torch.nn as nn
import torch.nn.functional as F


class SENN(nn.Module):
    def __init__(self, conceptizer, parameterizer, aggregator):
        """Represents a Self Explaining Neural Network (SENN).
        (https://papers.nips.cc/paper/8003-towards-robust-interpretability-with-self-explaining-neural-networks)

        A SENN model is a neural network made explainable by design. It is made out of several submodules:
            - conceptizer
                Model that encodes raw input into interpretable feature representations of
                that input. These feature representations are called concepts.
            - parameterizer
                Model that computes the parameters theta from given the input. Each concept
                has with it associated one theta, which acts as a ``relevance score'' for that concept.
            - aggregator
                Predictions are made with a function g(theta_1 * h_1, ..., theta_n * h_n), where
                h_i represents concept i. The aggregator defines the function g, i.e. how each
                concept with its relevance score is combined into a prediction.

        Parameters
        ----------
        conceptizer : Pytorch Module
            Model that encodes raw input into interpretable feature representations of
            that input. These feature representations are called concepts.
        parameterizer : Pytorch Module
            Model that computes the parameters theta from given the input. Each concept
            has with it associated one theta, which acts as a ``relevance score'' for that concept.
        aggregator : Pytorch Module
            Predictions are made with a function g(theta_1 * h_1, ..., theta_n * h_n), where
            h_i represents concept i. The aggregator defines the function g, i.e. how each
            concept with its relevance score is combined into a prediction.
        """
        super().__init__()
        self.conceptizer = conceptizer
        self.parameterizer = parameterizer
        self.aggregator = aggregator

    def forward(self, x):
        """Forward pass of SENN module.
        
        In the forward pass, concepts and their reconstructions are created from the input x.
        The relevance parameters theta are also computed.

        Parameters
        ----------
        x : torch.Tensor
            Input data tensor of shape (BATCH, *). Only restriction on the shape is that
            the first dimension should correspond to the batch size.

        Returns
        -------
        predictions : torch.Tensor
            Predictions generated by model. Of shape (BATCH, *).
        explanations : tuple
            Model explanations given by a tuple (concepts, relevances).
            concepts : torch.Tensor
                Interpretable feature representations of input. Of shape (NUM_CONCEPTS, *).
            parameters : torch.Tensor
                Relevance scores associated with concepts. Of shape (NUM_CONCEPTS, *)
        """
        concepts, recon_x = self.conceptizer(x)
        relevances = self.parameterizer(x)
        predictions = self.aggregator(concepts, relevances)
        explanations = (concepts, relevances)
        return predictions, explanations, recon_x


class SENND(nn.Module):
    """Self-Explaining Neural Network with Disentanglement 

    SENND is an extension of the Self-Explaining Neural Network proposed by [1]
    
    SENND incorporates a constrained variational inference framework on a 
    SENN Concept Encoder to learn disentangled representations of the 
    basis concepts as in [2]. The basis concepts are then independently
    sensitive to single generative factors leading to better interpretability 
    and lesser overlap with other basis concepts. Such a strong constraint 
    better fulfills the "diversity" desiderata for basis concepts
    in a Self-Explaining Neural Network.

    References
    ----------
    [1] Alvarez Melis, et al.
    "Towards Robust Interpretability with Self-Explaining Neural Networks" NIPS 2018
    [2] Irina Higgins, et al. 
    ”β-VAE: Learning basic visual concepts with a constrained variational framework.” ICLR 2017. 
    
    """
    
    def __init__(self, vae_conceptizer, parameterizer, aggregator):
        """Instantiates the SENDD with a variational conceptizer, parameterizer and aggregator

        Parameters
        ----------
        vae_conceptizer : nn.Module
            A variational inference model that learns a disentangled distribution over
            the prior basis concepts given the input posterior
        parameterizer : nn.Module
            Model that computes the parameters theta from given the input. Each concept
            has with it associated one theta, which acts as a ``relevance score'' for that concept.
        aggregator : nn.Module
            Predictions are made with a function g(theta_1 * h_1, ..., theta_n * h_n), where
            h_i represents concept i. The aggregator defines the function g, i.e. how each
            concept with its relevance score is combined into a prediction.
        """
        super().__init__()
        self.vae_conceptizer = vae_conceptizer
        self.parameterizer = parameterizer
        self.aggregator = aggregator

    def forward(self, x):
        """Forward pass of a SENND model
        
        The forward pass computes a distribution over basis concepts
        and the corresponding relevance scores. The mean concepts 
        and relevance scores are aggregated to generate a prediction.
        
        Parameters
        ----------
        x : torch.Tensor
            Input data tensor of shape [batch_size, ...]
            
        Returns
        -------
        predictions : torch.Tensor
            Predictions generated by the SENND model of shape [batch_size, ...]
        explanations : tuple
            Explanation give by the model as a nested tuple of 
            relevance scores and concept distribution as mean and log variance:
            ((concept_mean, concept_log_variance), relevance_score)
            concept_mean : torch.Tensor
                Mean of the disentangled concept distribution of shape
                [batch_size, num_concepts, concept_dim]
            concept_log_varance : torch.Tensor
                Log Variance of the disentangled concept distribution of shape
                [batch_size, num_concepts, concept_dim]
            relevance_score : torch.Tensor
                Relevance scores (for each concept and class) of shape 
                [batch_size, num_concepts, num_classes]
        """
        concept_mean, concept_logvar, x_reconstruct = self.vae_conceptizer(x)
        relevances = self.parameterizer(x)
        predictions = self.aggregator(concept_mean, relevances)
        explanations = ((concept_mean, concept_logvar), relevances)
        return predictions, explanations, x_reconstruct