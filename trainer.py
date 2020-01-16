from models.senn import SENN
from datasets.dataloaders import get_dataloader
from losses import *

import os
from os import path
from functools import partial

import torch
import torch.nn.functional as F
import torch.optim as opt
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
from losses import robustness_loss

import numpy as np
import matplotlib.pyplot as plt
import importlib
plt.style.use('seaborn-talk')

RESULTS_DIR = 'results'
CHECKPOINT_DIR = 'checkpoints' 
LOG_DIR = 'logs'

class Trainer():
    def __init__(self, config):
        """Base Trainer class containing functions to be overloaded by a specific Trainer agent.
        
        A trainer instantiates a model to be trained. It contains logic for training, validating,
        and checkpointing the model. All the specific parameters that control the program behavior
        are contained in the config parameter.

        The models we consider here are all Self Explaining Neural Networks (SENNs).

        If `load_checkpoint` is specified in config and the model has a checkpoint, the checkpoint
        will be loaded.
        
        Parameters
        ----------
        config : types.SimpleNamespace
            Contains all (hyper)parameters that define the behavior of the program.
        """
        self.config = config


        # get appropriate models from global namespace and instantiate them
        conceptizer = getattr(importlib.import_module("models.conceptizer"), config.conceptizer)(**config.__dict__)
        parameterizer = getattr(importlib.import_module("models.parameterizer"), config.parameterizer)(**config.__dict__)
        aggregator = getattr(importlib.import_module("models.aggregator"), config.aggregator)(**config.__dict__)

        self.model = SENN(conceptizer, parameterizer, aggregator)
        self.model.to(config.device)
        self.summarize(self.model)

        self.train_loader, self.val_loader, self.test_loader = get_dataloader(config)
        if config.num_classes == 1:
            self.classification_loss = F.binary_cross_entropy
        else:
            self.classification_loss = F.nll_loss
        self.concept_loss = mse_l1_sparsity
        self.robustness_loss = robustness_loss

        self.opt = opt.Adam(self.model.parameters(), lr=config.lr)

        # trackers
        self.losses = []
        self.classification_losses = []
        self.concept_losses = []
        self.robustness_losses = []
        self.current_iter = 0
        self.current_epoch = 0

        # directories for saving results
        self.experiment_dir = path.join(RESULTS_DIR, config.experiment_dir)
        self.checkpoint_dir = path.join(self.experiment_dir, CHECKPOINT_DIR)
        self.log_dir = path.join(self.experiment_dir, LOG_DIR)
        os.makedirs(RESULTS_DIR, exist_ok=True)
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        self.writer = SummaryWriter(log_dir=self.log_dir)

        if hasattr(config, "load_checkpoint"):
            self.load_checkpoint(config.load_checkpoint)
    
    def run(self):
        """Run the training loop.
        
        If the loop is interrupted manually, finalization will still be executed.
        """
        try:
            self.train()
        except KeyboardInterrupt:
            print("CTRL+C pressed... Waiting to finalize.")

    def train(self):
        """Main training loop."""
        for epoch in range(self.current_epoch, self.config.epochs):
            self.current_epoch = epoch
            self.train_one_epoch(self.current_epoch)
            # TODO: remove next 2 lines?
            self.validate()
            self.save_checkpoint()

    def train_one_epoch(self, epoch):
        """Run one epoch of training.

        Parameters
        ----------
        epoch : int
            Current epoch.
        """
        self.model.train()

        for i, (x, labels) in enumerate(self.train_loader):
            x = x.float().to(self.config.device)
            self.opt.zero_grad()

            # run x through SENN
            y_pred, (concepts, parameters), x_reconstructed = self.model(x)

            # TODO: compute losses
            # Definition of concept loss in the paper is inconsistent with source code (need for discussion)
            classification_loss = self.classification_loss(y_pred, labels)
            # TODO: arguments of robustness loss
            robustness_loss = self.robustness_loss(x, parameters, self.model)
            concept_loss = self.concept_loss(x, x_reconstructed, self.config.sparsity, concepts)

            total_loss = classification_loss + \
                           self.config.robust_reg * robustness_loss + \
                           self.config.concept_reg * concept_loss
            total_loss.backward()
            self.opt.step()

            accuracy = self.accuracy(y_pred, labels)

            # --- Report Training Progress --- #
            self.current_iter += 1
            self.losses.append(total_loss.item())
            self.classification_losses.append(classification_loss.item())
            self.concept_losses.append(concept_loss.item())
            self.robustness_losses.append(robustness_loss.item())

            self.writer.add_scalar('Loss/Train/Classification', classification_loss, self.current_iter)
            self.writer.add_scalar('Loss/Train/Robustness', robustness_loss, self.current_iter)
            self.writer.add_scalar('Loss/Train/Concept', concept_loss, self.current_iter)
            self.writer.add_scalar('Loss/Train/Total', total_loss, self.current_iter)
            self.writer.add_scalar('Accuracy/Train', accuracy, self.current_iter)

            if i % self.config.print_freq == 0:
               report = (f"EPOCH:{epoch} STEP:{i} \t"
                         f"Total Loss:{total_loss:.3f} \t"
                         f"Classification Loss:{classification_loss.item():.3f} \t"
                         f"Robustness Loss:{robustness_loss.item():.3f} \t"
                         f"Concept Loss Loss:{concept_loss.item():.3f} \t"
                         )
               print(report)
                        
    def validate(self):
        """Validate model performance.

        Model performance is validated by computing loss and accuracy measures, storing them,
        and reporting them.
        """
        self.model.eval()
        with torch.no_grad():
            x, labels = next(iter(self.val_loader))
            x = x.float().to(self.config.device)

            # run x through SENN
            y_pred, (concepts, parameters), x_reconstructed = self.model(x)

            # TODO: compute losses
            # Definition of concept loss in the paper is inconsistent with source code (need for discussion)
            classification_loss = self.classification_loss(y_pred, labels).item()
            # TODO: arguments of robustness loss
            robustness_loss = self.robustness_loss(x, parameters, self.model)
            concept_loss = self.concept_loss(x, x_reconstructed, self.config.sparsity, concepts).item()

            total_loss = classification_loss + \
                           self.config.robust_reg * robustness_loss + \
                           self.config.concept_reg * concept_loss

            accuracy = self.accuracy(y_pred, labels)

            # --- Report Training Progress --- #
            self.writer.add_scalar('Loss/Valid/Classification', classification_loss, self.current_iter)
            self.writer.add_scalar('Loss/Valid/Robustness', robustness_loss, self.current_iter)
            self.writer.add_scalar('Loss/Valid/Concept', concept_loss, self.current_iter)
            self.writer.add_scalar('Loss/Valid/Total', total_loss, self.current_iter)
            self.writer.add_scalar('Accuracy/Valid', accuracy, self.current_iter)
            # --- Report Validation --- #
            report = (
                     f"Total Loss:{total_loss:.3f} \t"
                     f"Classification Loss:{classification_loss:.3f} \t"
                     f"Robustness Loss:{robustness_loss:.3f} \t"
                     f"Concept Loss:{concept_loss:.3f} \t"
                     f"Accuracy:{accuracy:.3f} \t"
                     )
            print(report)

    def accuracy(self, y_pred, y):
        """Return accuracy of predictions with respect to ground truth.

        Parameters
        ----------
        y_pred : torch.Tensor, shape (BATCH,)
            Predictions of ground truth.
        y : torch.Tensor, shape (BATCH,)
            Ground truth.

        Returns
        -------
        float:
            accuracy of predictions
        """
        if len(y_pred.size()) == 1:
            accuracy = (y == torch.round(y_pred)).float().mean().item()
        else:
            accuracy = (y_pred.argmax(axis=1) == y).float().mean().item()
        return accuracy


    def load_checkpoint(self, file_name):
        """Load most recent checkpoint.

        If no checkpoint exists, doesn't do anything.

        Checkpoint contains:
            - current epoch
            - current iteration
            - model state
            - optimizer state

        Parameters
        ----------
        file_name: str
            Name of the checkpoint file.
        """
        try:
            file_name = self.checkpoint_dir + file_name
            print(f"Loading checkpoint...")
            with open(file_name, 'rb') as f:
                checkpoint = torch.load(f, self.device)

            self.current_epoch = checkpoint['epoch']
            self.current_iter = checkpoint['iter']
            self.model.load_state_dict(checkpoint['model_state'])
            self.opt.load_state_dict(checkpoint['optimizer'])

            print(f"Checkpoint loaded successfully from '{file_name}'\n")

        except OSError as e:
            print("No checkpoint exists @ {self.checkpoint_dir}")
            print("**Training for the first time**")


    def save_checkpoint(self):
        """Save checkpoint in the checkpoint directory.

        Checkpoint dir and checkpoint_file need to be specified in the config.
        """
        file_name = f"Epoch[{self.current_epoch}]-Step[{self.current_iter}].pt"
        file_name = path.join(self.checkpoint_dir, file_name)
        state = {
            'epoch': self.current_epoch,
            'iter': self.current_iter,
            'model_state': self.model.state_dict(),
            'optimizer': self.opt.state_dict(),
        }
        with open(file_name, 'wb') as f:
            torch.save(state, f)
        print(f"Checkpoint saved @ {file_name}\n")

    def visuallize(self, relevances, pred, path):
        plt.rcdefaults()
        fig, ax = plt.subplots()

        # Example data
        concept_names = ['Concept {}'.format(i) for i in relevances]
        y_pos = np.arange(len(concept_names))
        relevances = float(relevances)
        colors = ['b' if r > 0 else 'r' for r in relevances]

        ax.barh(y_pos, relevances, align='center', color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(concept_names)
        ax.set_xlabel('Relevances (thetas)')
        ax.set_title('Explanation for prediction: {}'.format(pred))

        plt.savefig(path)

    def finalize(self):
        """Finalize all necessary operations before exiting training.
        
        Saves checkpoint.
        """
        print("Please wait while we finalize...")
        self.save_checkpoint()

    def summarize(self, model):
        """Print summary of given model.

        Parameters
        ----------
        model :
            A Pytorch model containing parameters.
        """
        print(model)
        total_params = sum(p.numel() for p in model.parameters())
        train_params = sum(p.numel() for p in model.parameters())
        print(f"Total Parameters: {total_params}")
        print(f"Trainable Parameters: {train_params}\n")
