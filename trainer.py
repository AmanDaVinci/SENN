from models.senn import SENN
from datasets.load_mnist import load_mnist
from models.conceptizer import *
from models.parameterizer import *
from models.aggregator import *

import os
from functools import partial

import torch
import torch.nn as nn
import torch.optim as opt
import torchvision.utils as vutils

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-talk')


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

        conceptizer = globals()[config.conceptizer]
        parameterizer = globals()[config.parameterizer]
        aggregator = globals()[config.aggregator]
        self.model = SENN(conceptizer, parameterizer, aggregator)
        self.model.to(config.device)
        self.summarize(self.vae)

        self.trainloader, self.valloader, _ = get_dataloader(config)
        # TODO: optimizer in config
        self.opt = opt.Adam(self.model.parameters(), lr=config.lr)

        # trackers
        self.losses = []
        self.classification_losses = []
        self.concept_losses = []
        self.robustness_losses = []
        self.current_iter = 0
        self.current_epoch = 0
        os.makedirs(config.checkpoint_dir, exist_ok=True)

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
            # self.validate()
            # self.save_checkpoint()

    def train_one_epoch(self, epoch):
        """Run one epoch of training.

        Parameters
        ----------
        epoch : int
            Current epoch.
        """
        self.model.train()

        for i, xb in enumerate(self.trainloader):
            xb = xb.to(self.device)
            self.opt.zero_grad()

            # TODO: GET JACOBIAN CORRECTLY
            # concepts = self.model.get_concepts(xb)

            y_pred, (concepts, parameters) = self.model(xb)

            # TODO: compute losses
            classification_loss = 0
            concept_loss = 0
            robustness_loss = 0
            loss = classification_loss + \
                   self.config.concept_reg * concept_loss + \
                   self.config.robust_reg * robustness_loss
            loss.backward()
            self.opt.step()

            # --- Report Training Progress --- #
            self.current_iter += 1
            self.losses.append(loss.item())
            self.classification_losses.append(classification_loss.item())
            self.concept_losses.append(concept_loss.item())
            self.robustness_losses.append(robustness_loss.item())

            # TODO: fix this
            # if i%self.config.print_freq == 0:
            #     report = (f"EPOCH:{epoch} STEP:{i} \t"
            #               f"Class_Loss:{l_recon.item():.3f} Reg_Loss:{l_reg.item():.3f} \t"
            #               f"ELBO: {elbo.item():.3f}\t")
            #     print(report)
                        
    def validate(self):
        """Validate model performance.

        Model performance is validated by computing loss and accuracy measures, storing them,
        and reporting them.
        """
        self.vae.eval()
        with torch.no_grad():
            xb = next(iter(self.valloader))
            xb = xb.reshape(-1, self.config.x_dim).to(self.device)
            xb_hat, mean, logvar = self.vae(xb)
            elbo, l_recon, l_reg = self.vae.elbo(xb_hat, xb, mean, logvar)
            # --- Report Validation --- #
            report = (f"VALIDATION STEP:\t"
                      f"Recon_Loss:{l_recon.item():.3f} Reg_Loss:{l_reg.item():.3f} \t"
                      f"ELBO: {elbo.item():.3f}\t")
            print(report)
            self.val_iters.append(self.current_iter)
            self.val_elbo_losses.append(elbo.item())


    # TODO: remove?
    def sample(self, epoch, step=0):
        """
        Generate images from fixed noise
        to keep track of training progress
        """
        with torch.no_grad():
            samples, _ = self.vae.sample(self.config.sample_size)
            img_samples = samples.view(self.config.sample_size, 1, 28, 28)
            grid_imgs = vutils.make_grid(img_samples, nrow=5)
            plt.imshow(np.transpose(grid_imgs, (1,2,0)), cmap='binary')
            plt.axis("off")
            plt.savefig(self.config.image_dir + f"Epoch{epoch}_Step{step}.png")

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
            file_name = self.config.checkpoint_dir + file_name
            print(f"Loading checkpoint...")
            with open(file_name, 'rb') as f:
                checkpoint = torch.load(f, self.device)

            self.current_epoch = checkpoint['epoch']
            self.current_iter = checkpoint['iter']
            self.model.load_state_dict(checkpoint['model_state'])
            self.opt.load_state_dict(checkpoint['optimizer'])

            print(f"Checkpoint loaded successfully from '{file_name}'\n")
        
        except OSError as e:
            print("No checkpoint exists @ {self.config.checkpoint_dir}")
            print("**Training for the first time**")


    def save_checkpoint(self):
        """Save checkpoint in the checkpoint directory.
        
        Checkpoint dir and checkpoint_file need to be specified in the config.
        """
        file_name = self.config.checkpoint_dir+self.config.checkpoint_file
        file_name = file_name+f"_Epoch[{self.current_epoch}]-Step[{self.current_iter}].pt"
        state = {
            'epoch': self.current_epoch,
            'iter': self.current_iter,
            'model_state': self.model.state_dict(),
            'optimizer': self.opt.state_dict(),
        }
        with open(file_name, 'wb') as f:
            torch.save(state, f)
        print(f"Checkpoint saved @ {file_name}\n")

    # TODO: remove this function?
    def plot_losses(self):
        """Generate a plot of G & D losses"""
        img_dir = self.config.image_dir
        imgname = img_dir + f"Loss_Epoch[{self.current_epoch}]-Step[{self.current_iter}].png"
        fig, ax = plt.subplots(figsize=(15,10))
        ax.scatter(self.val_iters, self.val_elbo_losses, color='red', label="Validation ELBO")
        ax.plot(self.iters, self.elbo_losses, "g:", alpha=0.7, label="Training ELBO")
        ax.plot(self.iters, self.recon_losses, alpha=0.3, label="Reconstruction Loss")
        ax.plot(self.iters, self.reg_losses, alpha=0.3, label="KL Divergence")
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Loss")
        ax.legend()
        fig.tight_layout()
        plt.savefig(imgname)
        print(f"Loss curves plotted @ {imgname}")

    def finalize(self):
        """Finalize all necessary operations before exiting training.
        
        Saves checkpoint.
        """
        print("Please wait while we finalize...")
        self.save_checkpoint()
        # TODO: remove next line?
        self.plot_losses()

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
