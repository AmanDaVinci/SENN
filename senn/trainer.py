import csv
import json
import os
from os import path
from pprint import pprint
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as opt
from torch.utils.tensorboard import SummaryWriter

from .models.losses import *
from .models.senn import SENN, DiSENN
from .models.losses import *
from .models.aggregators import *
from .models.parameterizers import *
from .models.conceptizers import *
from .utils.plot_utils import *
from .datasets.dataloaders import get_dataloader


plt.style.use('seaborn-talk')

RESULTS_DIR = 'results'
CHECKPOINT_DIR = 'checkpoints'
LOG_DIR = 'logs'
BEST_MODEL_FILENAME = "best_model.pt"


def init_trainer(config_file, best_model=False):
    """Instantiate the Trainer class based on the config parameters

    Parameters
    ----------
    config_file: str
        filename of the json config with all experiment parameters
    best_model: bool
        whether to load the previously trained best model

    Returns
    -------
    trainer: SENN_Trainer
        Trainer for SENN or DiSENNTrainer for DiSENN
    """
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
    if hasattr(config, 'model_class') and config.model_class == 'DiSENN':
        trainer = DiSENN_Trainer(config)
    else:
        trainer = SENN_Trainer(config)
    return trainer


class SENN_Trainer:
    def __init__(self, config):
        """Base SENN Trainer class.
        
        A trainer instantiates a model to be trained. It contains logic for training, validating,
        checkpointing, etc. All the specific parameters that control the experiment behaviour
        are contained in the configs json.

        The models we consider here are all Self Explaining Neural Networks (SENNs).

        If `load_checkpoint` is specified in configs and the model has a checkpoint, the checkpoint
        will be loaded.
        
        Parameters
        ----------
        config : types.SimpleNamespace
            Contains all (hyper)parameters that define the behavior of the program.
        """
        self.config = config
        print(f"Using device {config.device}")

        # Load data
        print("Loading data ...")
        self.train_loader, self.val_loader, self.test_loader = get_dataloader(config)

        if hasattr(config, "manual_seed"):
            torch.manual_seed(config.manual_seed)

        # get appropriate models from global namespace and instantiate them
        try:
            conceptizer = eval(config.conceptizer)(**config.__dict__)
            parameterizer = eval(config.parameterizer)(**config.__dict__)
            aggregator = eval(config.aggregator)(**config.__dict__)
        except:
            print("Please make sure you specify the correct Conceptizer, Parameterizer and Aggregator classes")
            exit(-1)

        # Define losses
        self.classification_loss = F.nll_loss
        self.concept_loss = mse_l1_sparsity
        self.robustness_loss = eval(config.robustness_loss)

        # Init model
        self.model = SENN(conceptizer, parameterizer, aggregator)
        self.model.to(config.device)
        self.summarize()

        # Init optimizer
        self.opt = opt.Adam(self.model.parameters(), lr=config.lr)

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
        """Main training loop. Saves a model checkpoint after every epoch."""
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

            classification_loss = self.classification_loss(y_pred.squeeze(-1), labels)
            robustness_loss = self.robustness_loss(x, y_pred, concepts, relevances)
            concept_loss = self.concept_loss(x, x_reconstructed, concepts, self.config.sparsity_reg)

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
        """Get the metrics for the validation set
        """
        return self.get_metrics(validate=True)

    def test(self):
        """Get the metrics for the test set
        """
        return self.get_metrics(validate=False)

    def get_metrics(self, validate=True):
        """Get the metrics for a validation/test set

        If the validation flag is on, the function tests the model
        with the validation dataset instead of the testing one.

        Model performance is validated by computing loss and accuracy measures, storing them,
        and reporting them.

        Parameters
        ----------
        validate : bool
            Indicates whether to use the validation or test dataset
        """
        losses_val = []
        classification_losses_val = []
        concept_losses_val = []
        robustness_losses_val = []
        accuracies_val = []

        dl = self.val_loader if validate else self.test_loader

        self.model.eval()
        with torch.no_grad():
            for x, labels in dl:
                x = x.float().to(self.config.device)
                labels = labels.long().to(self.config.device)

                # run x through SENN
                y_pred, (concepts, _), x_reconstructed = self.model(x)

                classification_loss = self.classification_loss(y_pred.squeeze(-1), labels)
                # robustness_loss = self.robustness_loss(x, y_pred, concepts, relevances)
                robustness_loss = torch.tensor(0.0)  # jacobian cannot be computed with no_grad enabled
                concept_loss = self.concept_loss(x, x_reconstructed, concepts, self.config.sparsity_reg)

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

            if validate:
                # --- Report Training Progress --- #
                self.writer.add_scalar('Loss/Valid/Classification', classification_loss, self.current_iter)
                self.writer.add_scalar('Loss/Valid/Robustness', robustness_loss, self.current_iter)
                self.writer.add_scalar('Loss/Valid/Concept', concept_loss, self.current_iter)
                self.writer.add_scalar('Loss/Valid/Total', total_loss, self.current_iter)
                self.writer.add_scalar('Accuracy/Valid', accuracy, self.current_iter)

            # --- Report statistics --- #
            print(f"\n\033[93m-------- {'Validation' if validate else 'Test'} --------")
            self.print_n_save_metrics(filename=f"accuracies_losses_{'valid' if validate else 'test'}.csv",
                                      total_loss=total_loss,
                                      classification_loss=classification_loss,
                                      robustness_loss=robustness_loss,
                                      concept_loss=concept_loss,
                                      accuracy=accuracy)
            print("----------------------------\033[0m")

            if accuracy > self.best_accuracy and validate:
                print("\033[92mCongratulations! Saving a new best model...\033[00m")
                self.best_accuracy = accuracy
                self.save_checkpoint(BEST_MODEL_FILENAME)

        return accuracy

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
        return (y_pred.argmax(axis=1) == y).float().mean().item()

    def load_checkpoint(self, file_name):
        """Load most recent checkpoint.

        If no checkpoint exists, doesn't do anything.

        Checkpoint contains:
            - current epoch
            - current iteration
            - model state
            - best accuracy achieved so far
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

        except OSError:
            print(f"No checkpoint exists @ {self.checkpoint_dir}")
            print("**Training for the first time**")

    def save_checkpoint(self, file_name=None):
        """Save checkpoint in the checkpoint directory.

        Checkpoint dir and checkpoint_file need to be specified in the configs.

        Parameters
        ----------
        file_name: str
            Name of the checkpoint file.
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
        """Prints the losses to the console and saves them in a csv file

        Parameters
        ----------
        filename: str
            Name of the csv file.
        classification_loss: float
            The value of the classification loss
        robustness_loss: float
            The value of the robustness loss
        total_loss: float
            The value of the total loss
        concept_loss: float
            The value of the concept loss
        accuracy: float
            The value of the accuracy
        """
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
        """Generates plots to visualize the explanations.

        Parameters
        ----------
        save_dir : str
            Directory where the figures are saved
        """
        self.model.eval()

        show_explainations(self.model, self.test_loader, self.config.dataloader,
                           num_explanations=10, save_path=save_dir, **self.config.__dict__)

        if self.config.dataloader == 'mnist':
            save_path = path.join(save_dir, 'concept_activation.png')
            highest_activations(self.model, self.test_loader, save_path=save_path)
            save_path = path.join(save_dir, 'concept_contrast.png')
            highest_contrast(self.model, self.test_loader, save_path=save_path)
            save_path = path.join(save_dir, 'concept_filter.png')
            filter_concepts(self.model, save_path=save_path)

        if hasattr(self.config, 'accuracy_vs_lambda'):
            save_path = path.join(save_dir, 'accuracy_vs_lambda.png')
            plot_lambda_accuracy(self.config.accuracy_vs_lambda, save_path, **self.config.__dict__)

    def finalize(self):
        """Finalize all necessary operations before exiting training.
        
        Saves checkpoint.
        """
        print("Please wait while we finalize...")
        self.save_checkpoint()

    def summarize(self):
        """Print summary of given model.
        """
        print(self.model)
        train_params = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable Parameters: {train_params}\n")


class DiSENN_Trainer(SENN_Trainer):
    """Extends general Trainer to train a DiSENN model"""

    # TODO: Refactor the inheritance code
    def __init__(self, config):
        """Instantiates a trainer for DiSENN

        Parameters
        ----------
        config : types.SimpleNamespace
            Contains all (hyper)parameters that define the behavior of the program.
        """
        super().__init__(config)

        print("Reinstantiating for DiSENN ...")
        try:
            conceptizer = eval(config.conceptizer)(**config.__dict__)
            parameterizer = eval(config.parameterizer)(**config.__dict__)
            aggregator = eval(config.aggregator)(**config.__dict__)
        except Exception:
            print("Please make sure you specify the correct Conceptizer, Parameterizer and Aggregator classes")
            exit(-1)

        # Define DiSENN losses
        self.classification_loss = F.nll_loss
        self.concept_loss = BVAE_loss
        self.robustness_loss = eval(config.robustness_loss)

        # Init model
        self.model = DiSENN(conceptizer, parameterizer, aggregator)
        self.model.to(config.device)
        self.summarize()

        # Init optimizer
        self.opt = opt.Adam(self.model.parameters())

        # Pretrain Conceptizer if required
        if self.config.pretrain_epochs > 0:
            print("Pre-training the Conceptizer... ")
            self.model.vae_conceptizer = self.pretrain(self.model.vae_conceptizer, self.config.pre_beta)

    def pretrain(self, conceptizer, beta=0.):
        """Pre-trains conceptizer on the training data to optimize the concept loss
        
        Parameters:
        ----------
        conceptizer : VaeConceptizer
            object of class VaeConceptizer to be pre-trained
        beta : float
            beta value during the pre-training of the beta-VAE
        """

        optimizer = opt.Adam(conceptizer.parameters())
        conceptizer.to(self.config.device)
        conceptizer.train()

        for epoch in range(self.config.pretrain_epochs):
            for i, (x, _) in enumerate(self.train_loader):
                optimizer.zero_grad()
                x = x.float().to(self.config.device)
                concept_mean, concept_logvar, x_reconstruct = conceptizer(x)
                recon_loss, kl_div = self.concept_loss(x, x_reconstruct, concept_mean, concept_logvar)
                loss = recon_loss + beta * kl_div
                loss.backward()
                optimizer.step()
                if i % self.config.print_freq == 0:
                    print(f"EPOCH:{epoch} STEP:{i} \t"
                          f"Concept Loss: {loss:.3f} "
                          f"Recon Loss: {recon_loss:.3f} "
                          f"KL Div: {kl_div:.3f}")

        return conceptizer

    def train_one_epoch(self, epoch):
        """Run one epoch of training.

        Parameters
        ----------
        epoch : int
            Current epoch
        """
        self.model.train()

        for i, (x, labels) in enumerate(self.train_loader):
            x = x.float().to(self.config.device)
            labels = labels.long().to(self.config.device)
            self.opt.zero_grad()

            # track all operations on x for jacobian calculation
            x.requires_grad_(True)
            y_pred, (concepts_dist, relevances), x_reconstructed = self.model(x)

            concept_mean, concept_logvar = concepts_dist
            concepts = concept_mean

            classification_loss = self.classification_loss(y_pred.squeeze(-1), labels)
            robustness_loss = self.robustness_loss(x, y_pred, concepts, relevances)
            recon_loss, kl_div = self.concept_loss(x, x_reconstructed,
                                                   concept_mean, concept_logvar)
            concept_loss = recon_loss + self.config.beta * kl_div

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
                                          recon_loss=recon_loss.item(),
                                          kl_div=kl_div.item(),
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
            for x, labels in self.val_loader:
                x = x.float().to(self.config.device)
                labels = labels.long().to(self.config.device)

                y_pred, (concepts_dist, _), x_reconstructed = self.model(x)
                concept_mean, concept_logvar = concepts_dist

                classification_loss = self.classification_loss(y_pred.squeeze(-1), labels)
                robustness_loss = torch.tensor(0.0)  # jacobian cannot be computed with no_grad enabled
                recon_loss, kl_div = self.concept_loss(x, x_reconstructed,
                                                       concept_mean, concept_logvar)
                concept_loss = recon_loss + self.config.beta * kl_div

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
                                      recon_loss=recon_loss.item(),
                                      kl_div=kl_div.item(),
                                      accuracy=accuracy)
            print("----------------------------\033[0m")

            if accuracy > self.best_accuracy:
                print("\033[92mCongratulations! Saving a new best model...\033[00m")
                self.best_accuracy = accuracy
                self.save_checkpoint(BEST_MODEL_FILENAME)

            # TODO: organize this block
            self.visualize("dummy.png")
            print("Saving model ...")
            self.save_checkpoint()

    def visualize(self, save_dir, num=3):
        """Generates some plots to visualize the explanations.

        Parameters
        ----------
        save_dir : str
            A placeholder to work with the Base Trainer class which calls visualize.
            Needs refactoring.
        num: int
            Number of examples
        """
        self.model.eval()

        # select test example
        (test_batch, _) = next(iter(self.test_loader))
        for i in range(num):
            file_path = Path("results")
            file_name = file_path / self.config.exp_name / "explanations.png"
            x = test_batch[i].float().to(self.config.device)
            self.model.explain(x, save_as=file_name)

    def print_n_save_metrics(self, filename, total_loss,
                             classification_loss, robustness_loss,
                             concept_loss, recon_loss, kl_div, accuracy):
        """Prints the losses to the console and saves them in a csv file

        Parameters
        ----------
        filename: str
            Name of the csv file.
       
        total_loss: float
            The value of the total loss
       
        classification_loss: float
            The value of the classification loss
       
        robustness_loss: float
            The value of the robustness loss
       
        concept_loss: float
            The value of the concept loss
       
        recon_loss: float
            Reconstruction loss of the VAE Conceptizer

        kl_div : float
            KL Divergence loss of VAE Conceptizer

        accuracy: float
            The value of the accuracy
        """

        report = (f"Total Loss:{total_loss:.3f} "
                  f"Accuracy:{accuracy:.3f} "
                  f"Classification Loss:{classification_loss:.3f} "
                  f"Robustness Loss:{robustness_loss:.3f} "
                  f"Concept Loss:{concept_loss:.3f} "
                  f"Recon Loss: {recon_loss:.3f} "
                  f"KL Div: {kl_div:.3f} ")
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
