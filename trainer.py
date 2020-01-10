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
    """
    This base class will contain the base functions to be overloaded by any agent you will implement.
    """

    def __init__(self, config):
        """ Init config, model, optim & criterion
        """
        self.config = config

        conceptizer = globals()[config.conceptizer]
        parameterizer = globals()[config.parameterizer]
        aggregator = globals()[config.aggregator]
        self.model = SENN(conceptizer, parameterizer, aggregator)
        self.model.to(config.device)
        self.summarize(self.vae)

        self.trainloader, self.valloader, _ = get_dataloader(config)
        # TODO: opt in config
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
        """ Runs the train loop until interrupted
        """
        try:
            self.train()
        except KeyboardInterrupt:
            print("CTRL+C pressed... Waiting to finalize.")

    def train(self):
        """
        Main training loop
        :return:
        """
        for epoch in range(self.current_epoch, self.config.epochs):
            self.current_epoch = epoch
            self.train_one_epoch(self.current_epoch)
            # self.validate()
            # self.save_checkpoint()

    def train_one_epoch(self, epoch):
        """
        One epoch of training
        :return:
        """
        self.model.train()

        for i, xb in enumerate(self.trainloader):
            xb = xb.to(self.device)
            self.opt.zero_grad()

            # TODO: GET JACOBIAN CORRECTLY
            # concepts = self.model.get_concepts(xb)

            y_pred, (concepts, parameters) = self.model(xb)

            classification_loss = self.
            concept_loss = 0.
            robustness_loss = ...
            loss = classification_loss + \
                   self.config.concept_reg * concept_loss + \
                   self.config.robust_reg * robustness_loss
            # TODO: fix this shit
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
        """One cycle of model validation"""
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
            plt.savefig(self.config.image_dir+f"Epoch{epoch}_Step{step}.png")

    def load_checkpoint(self, file_name):
        """
        Latest checkpoint loader
        :param file_name: name of the checkpoint file
        :return:
        """
        try:
            file_name = self.config.checkpoint_dir+file_name
            print(f"Loading checkpoint...")
            with open(file_name, 'rb') as f:
                checkpoint = torch.load(f, self.device)

            self.current_epoch = checkpoint['epoch']
            self.current_iter = checkpoint['iter']
            self.vae.load_state_dict(checkpoint['model_state'])
            self.opt.load_state_dict(checkpoint['optimizer'])

            print(f"Checkpoint loaded successfully from '{file_name}'\n")
        
        except OSError as e:
            print("No checkpoint exists @ {self.config.checkpoint_dir}")
            print("**Training for the first time**")


    def save_checkpoint(self):
        """
        Checkpoint saver
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
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        :return:
        """
        print("Please wait while we finalize...")
        self.save_checkpoint()
        self.plot_losses()

    def summarize(self, model):
        print(model)
        total_params = sum(p.numel() for p in model.parameters())
        train_params = sum(p.numel() for p in model.parameters())
        print(f"Total Parameters: {total_params}")
        print(f"Trainable Parameters: {train_params}\n")