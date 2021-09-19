import argparse
import os
import numpy as np
import math
import itertools
import scipy
import sys
import time
import datetime

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.autograd as autograd

import torch.nn as nn
import torch.nn.functional as F
import torch

batches_done = 0
prev_time = time.time()
for epoch in range(opt.n_epochs):
    for i, batch in enumerate(dataloader):

        # Configure input
        imgs_A = Variable(batch["A"].type(FloatTensor))
        imgs_B = Variable(batch["B"].type(FloatTensor))

        optimizer_D_A.zero_grad()
        optimizer_D_B.zero_grad()

        # Generate a batch of images
        fake_A = G_BA(imgs_B).detach()
        fake_B = G_AB(imgs_A).detach()

        # ----------
        # Domain A
        # ----------

        # Compute gradient penalty for improved wasserstein training
        gp_A = compute_gradient_penalty(D_A, imgs_A.data, fake_A.data)
        # Adversarial loss
        D_A_loss = -torch.mean(D_A(imgs_A)) + torch.mean(D_A(fake_A)) + lambda_gp * gp_A

        # ----------
        # Domain B
        # ----------

        # Compute gradient penalty for improved wasserstein training
        gp_B = compute_gradient_penalty(D_B, imgs_B.data, fake_B.data)
        # Adversarial loss
        D_B_loss = -torch.mean(D_B(imgs_B)) + torch.mean(D_B(fake_B)) + lambda_gp * gp_B

        # Total loss
        D_loss = D_A_loss + D_B_loss

        D_loss.backward()
        optimizer_D_A.step()
        optimizer_D_B.step()

        if i % opt.n_critic == 0:

            # ------------------
            #  Train Generators
            # ------------------

            optimizer_G.zero_grad()

            # Translate images to opposite domain
            fake_A = G_BA(imgs_B)
            fake_B = G_AB(imgs_A)

            # Reconstruct images
            recov_A = G_BA(fake_B)
            recov_B = G_AB(fake_A)

            # Adversarial loss
            G_adv = -torch.mean(D_A(fake_A)) - torch.mean(D_B(fake_B))
            # Cycle loss
            G_cycle = cycle_loss(recov_A, imgs_A) + cycle_loss(recov_B, imgs_B)
            # Total loss
            G_loss = lambda_adv * G_adv + lambda_cycle * G_cycle

            G_loss.backward()
            optimizer_G.step()

            # --------------
            # Log Progress
            # --------------

            # Determine approximate time left
            batches_left = opt.n_epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time) / opt.n_critic)
            prev_time = time.time()

            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, cycle: %f] ETA: %s"
                % (
                    epoch,
                    opt.n_epochs,
                    i,
                    len(dataloader),
                    D_loss.item(),
                    G_adv.data.item(),
                    G_cycle.item(),
                    time_left,
                )
            )

        # Check sample interval => save sample if there
        if batches_done % opt.sample_interval == 0:
            sample_images(batches_done)

        batches_done += 1

    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(G_AB.state_dict(), "saved_models/%s/G_AB_%d.pth" % (opt.dataset_name, epoch))
        torch.save(G_BA.state_dict(), "saved_models/%s/G_BA_%d.pth" % (opt.dataset_name, epoch))
        torch.save(D_A.state_dict(), "saved_models/%s/D_A_%d.pth" % (opt.dataset_name, epoch))
        torch.save(D_B.state_dict(), "saved_models/%s/D_B_%d.pth" % (opt.dataset_name, epoch))
