from models.senn import SENN
from datasets.dataloaders import get_dataloader
from models.losses import *
from utils.concept_representations import *
from utils.plot_utils import *

import os
from os import path
import csv

import json
from pprint import pprint
from types import SimpleNamespace
from functools import partial

import torch
import torch.nn.functional as F
import torch.optim as opt
from torch.utils.tensorboard import SummaryWriter
from models.losses import compas_robustness_loss, mnist_robustness_loss

import numpy as np
import matplotlib.pyplot as plt
import importlib

plt.style.use('seaborn-talk')

RESULTS_DIR = 'results'
CHECKPOINT_DIR = 'checkpoints'
LOG_DIR = 'logs'
BEST_MODEL_FILENAME = "best_model.pt"


def init_trainer(config_file, best_model=False):
    with open(config_file, 'r') as f:
        config = json.load(f)

    if best_model:
        config["load_checkpoint"] = BEST_MODEL_FILENAME

    print("==================================================")
    print(f" EXPERIMENT: {config['exp_name']}")
    print("==================================================")
    pprint(config)
    config = SimpleNamespace(**config)
    # create the trainer class and init with config
    trainer = Trainer(config)
    return trainer


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
        try:
            conceptizer = getattr(importlib.import_module("models.conceptizer"), config.conceptizer)(**config.__dict__)
            parameterizer = getattr(importlib.import_module("models.parameterizer"), config.parameterizer)(**config.__dict__)
            aggregator = getattr(importlib.import_module("models.aggregator"), config.aggregator)(**config.__dict__)
        except:
            print("Please make sure you specify the correct Conceptizer, Parameterizer and Aggregator classes")
            exit(-1)

        # Init model
        self.model = SENN(conceptizer, parameterizer, aggregator)
        self.model.to(config.device)
        self.summarize()

        # Init data
        print("Loading data ...")
        self.train_loader, self.val_loader, self.test_loader = get_dataloader(config)

        # Init losses
        self.classification_loss = F.nll_loss
        # TODO: concept loss should return zero for identity conceptizer
        self.concept_loss = mse_l1_sparsity
        if config.dataloader == "compas":
            self.robustness_loss = compas_robustness_loss
        elif config.dataloader == "mnist":
            self.robustness_loss = mnist_robustness_loss
        else:
            raise Exception("Robustness loss not defined")

        # Init optimizer
        self.opt = opt.Adam(self.model.parameters())

        # Init trackers
        self.current_iter = 0
        self.current_epoch = 0
        self.best_accuracy = 0

        # directories for saving results
        self.experiment_dir = path.join(RESULTS_DIR, config.exp_name)
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
            if self.config.train:
                print("Training begins...")
                self.train()
            self.visualize(save_dir=self.experiment_dir)
        except KeyboardInterrupt:
            print("CTRL+C pressed... Waiting to finalize.")

    def train(self):
        """Main training loop."""
        for epoch in range(self.current_epoch, self.config.epochs):
            self.current_epoch = epoch
            self.train_one_epoch(self.current_epoch)
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
            labels = labels.long().to(self.config.device)
            self.opt.zero_grad()
            # track all operations on x for jacobian calculation
            x.requires_grad_(True)

            # run x through SENN
            y_pred, (concepts, relevances), x_reconstructed = self.model(x)

            # visualize SENN computation graph
            self.writer.add_graph(self.model, x)

            classification_loss = self.classification_loss(y_pred.squeeze(-1), labels)
            robustness_loss = self.robustness_loss(x, y_pred, concepts, relevances)
            concept_loss = self.concept_loss(x, x_reconstructed, self.config.sparsity_reg, concepts)

            total_loss = classification_loss + \
                         self.config.robust_reg * robustness_loss + \
                         self.config.concept_reg * concept_loss
            total_loss.backward()
            self.opt.step()

            accuracy = self.accuracy(y_pred, labels)

            # --- Report Training Progress --- #
            self.current_iter += 1

            self.writer.add_scalar('Loss/Train/Classification', classification_loss, self.current_iter)
            self.writer.add_scalar('Loss/Train/Robustness', robustness_loss, self.current_iter)
            self.writer.add_scalar('Loss/Train/Concept', concept_loss, self.current_iter)
            self.writer.add_scalar('Loss/Train/Total', total_loss, self.current_iter)
            self.writer.add_scalar('Accuracy/Train', accuracy, self.current_iter)

            if i % self.config.print_freq == 0:
                print(f"EPOCH:{epoch} STEP:{i}")
                self.print_n_save_metrics(filename="accuracies_losses_train.csv",
                                          total_loss=total_loss.item(),
                                          classification_loss=classification_loss.item(),
                                          robustness_loss=robustness_loss.item(),
                                          concept_loss=concept_loss.item(),
                                          accuracy=accuracy)

            if self.current_iter % self.config.eval_freq == 0:
                self.validate()

    def validate(self):
        """Validate model performance.

        Model performance is validated by computing loss and accuracy measures, storing them,
        and reporting them.
        """
        losses_val = []
        classification_losses_val = []
        concept_losses_val = []
        robustness_losses_val = []
        accuracies_val = []

        self.model.eval()
        with torch.no_grad():
            for i, (x, labels) in enumerate(self.val_loader):
                x = x.float().to(self.config.device)
                labels = labels.long().to(self.config.device)

                # run x through SENN
                y_pred, (concepts, relevances), x_reconstructed = self.model(x)

                classification_loss = self.classification_loss(y_pred.squeeze(-1), labels)
                # robustness_loss = self.robustness_loss(x, y_pred, concepts, relevances)
                robustness_loss = torch.tensor(0.0) # jacobian cannot be computed with no_grad enabled
                concept_loss = self.concept_loss(x, x_reconstructed, self.config.sparsity_reg, concepts)
                
                total_loss = classification_loss + \
                             self.config.robust_reg * robustness_loss + \
                             self.config.concept_reg * concept_loss

                accuracy = self.accuracy(y_pred, labels)

                losses_val.append(total_loss.item())
                classification_losses_val.append(classification_loss.item())
                concept_losses_val.append(concept_loss.item())
                robustness_losses_val.append(robustness_loss.item())
                accuracies_val.append(accuracy)

            classification_loss = np.mean(classification_losses_val)
            robustness_loss = np.mean(robustness_losses_val)
            concept_loss = np.mean(concept_losses_val)
            total_loss = np.mean(losses_val)
            accuracy = np.mean(accuracies_val)

            # --- Report Training Progress --- #
            self.writer.add_scalar('Loss/Valid/Classification', classification_loss, self.current_iter)
            self.writer.add_scalar('Loss/Valid/Robustness', robustness_loss, self.current_iter)
            self.writer.add_scalar('Loss/Valid/Concept', concept_loss, self.current_iter)
            self.writer.add_scalar('Loss/Valid/Total', total_loss, self.current_iter)
            self.writer.add_scalar('Accuracy/Valid', accuracy, self.current_iter)

            # --- Report Validation --- #
            print("\n\033[93m-------- Validation --------")
            self.print_n_save_metrics(filename="accuracies_losses_valid.csv",
                                      total_loss=total_loss,
                                      classification_loss=classification_loss,
                                      robustness_loss=robustness_loss,
                                      concept_loss=concept_loss,
                                      accuracy=accuracy)
            print("----------------------------\033[0m")

            if accuracy > self.best_accuracy:
                print("\033[92mCongratulations! Saving a new best model...\033[00m")
                self.best_accuracy = accuracy
                self.save_checkpoint(BEST_MODEL_FILENAME)

    def accuracy(self, y_pred, y):
        """
        Return accuracy of predictions with respect to ground truth.

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
        return (y_pred.argmax(axis=1) == y).float().mean().item()

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
            file_name = path.join(self.checkpoint_dir, file_name)
            print(f"Loading checkpoint...")
            checkpoint = torch.load(file_name, self.config.device)

            self.current_epoch = checkpoint['epoch']
            self.current_iter = checkpoint['iter']
            self.best_accuracy = checkpoint['best_accuracy']
            self.model.load_state_dict(checkpoint['model_state'])
            self.opt.load_state_dict(checkpoint['optimizer'])

            print(f"Checkpoint loaded successfully from '{file_name}'\n")

        except OSError as e:
            print(f"No checkpoint exists @ {self.checkpoint_dir}")
            print("**Training for the first time**")

    def save_checkpoint(self, file_name=None):
        """Save checkpoint in the checkpoint directory.

        Checkpoint dir and checkpoint_file need to be specified in the config.
        """
        if file_name is None:
            file_name = f"Epoch[{self.current_epoch}]-Step[{self.current_iter}].pt"

        file_name = path.join(self.checkpoint_dir, file_name)
        state = {
            'epoch': self.current_epoch,
            'iter': self.current_iter,
            'best_accuracy': self.best_accuracy,
            'model_state': self.model.state_dict(),
            'optimizer': self.opt.state_dict(),
        }
        torch.save(state, file_name)

        print(f"Checkpoint saved @ {file_name}\n")

    def print_n_save_metrics(self, filename, total_loss, classification_loss, robustness_loss, concept_loss, accuracy):
        report = (f"Total Loss:{total_loss:.3f} \t"
                  f"Classification Loss:{classification_loss:.3f} \t"
                  f"Robustness Loss:{robustness_loss:.3f} \t"
                  f"Concept Loss:{concept_loss:.3f} \t"
                  f"Accuracy:{accuracy:.3f} \t")
        print(report)

        filename = path.join(self.experiment_dir, filename)
        new_file = not os.path.exists(filename)
        with open(filename, 'a') as metrics_file:
            fieldnames = ['Accuracy', 'Loss', 'Classification_Loss', 'Robustness_Loss', 'Concept_Loss', 'Step']
            csv_writer = csv.DictWriter(metrics_file, fieldnames=fieldnames)

            if new_file: csv_writer.writeheader()

            csv_writer.writerow({'Accuracy': accuracy, 'Classification_Loss': classification_loss,
                                 'Robustness_Loss': robustness_loss, 'Concept_Loss': concept_loss,
                                 'Loss': total_loss, 'Step': self.current_iter})

    def visualize(self, save_dir):
        """Generates some plots to visualize the explanations.

        Parameters
        ----------
        save_dir : str
            Directory where the figures are saved
        """
        self.model.eval()

        # select test example
        (test_batch, test_labels) = next(iter(self.test_loader))
        for i in range(10):
            example = test_batch[i].float().to(self.config.device)
            if self.config.dataloader == 'mnist':
                save(example, path.join(save_dir, 'test_example_{}.png'.format(i)))

            # feed example to model to obtain explanation
            y_pred, (concepts, relevances), _ = self.model(example.unsqueeze(0))
            if len(y_pred.size()) > 1:
                y_pred = y_pred.argmax(1)

            create_barplot(relevances, y_pred, save_path=path.join(save_dir, 'relevances_{}.png'.format(i)))

        if hasattr(self.config, 'concept_visualization'):
            # create visualization of the concepts with method specified in config file
            save_path = path.join(save_dir, 'concept.png')
            if self.config.concept_visualization == 'activation':
                highest_activations(self.model, self.test_loader, save_path=save_path)
            elif self.config.concept_visualization == 'contrast':
                highest_contrast(self.model, self.test_loader, save_path=save_path)
            elif self.config.concept_visualization == 'filter':
                filter_concepts(self.model, save_path=save_path)

    def finalize(self):
        """Finalize all necessary operations before exiting training.
        
        Saves checkpoint.
        """
        print("Please wait while we finalize...")
        self.save_checkpoint()

    def summarize(self):
        """Print summary of given model.

        Parameters
        ----------
        model :
            A Pytorch model containing parameters.
        """
        print(self.model)
        train_params = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable Parameters: {train_params}\n")
