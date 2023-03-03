import sys
sys.path.append("/home/bobrin_m_s/Projects/GansInAction/PyTorch/GAN")

import torch
from absl import flags
import torchvision
import matplotlib.pyplot as plt
import numpy as np

FLAGS = flags.FLAGS

class GanTrainer:
    def __init__(self, factory) -> None:
        self.factory = factory

    def _train_step(self, batch_images: torch.Tensor, num: int, epoch: int) -> None:

        # Train Discriminator
        batch_images = batch_images.to("cuda:0")
        noise_vector = torch.randn(FLAGS.batch_size, FLAGS.latent_dim).to("cuda:0")
        fake_images = self.factory.generator(noise_vector).to("cuda:0")
        disc_realimages = self.factory.discriminator(batch_images).view(-1)
        disc_fakeimages = self.factory.discriminator(fake_images.detach()).view(-1)

        lossD_reals = self.factory.criterion(disc_realimages, torch.ones_like(disc_realimages))
        lossD_fake = self.factory.criterion(disc_fakeimages, torch.zeros_like(disc_fakeimages))
        discriminator_loss = (lossD_fake + lossD_reals) / 2.0
        self.factory.discriminator.zero_grad()
        discriminator_loss.backward(retain_graph=True)
        self.factory.opt_discr.step()

        # Train Generator
        lossD_fake = self.factory.discriminator(fake_images).view(-1)
        generator_loss = self.factory.criterion(lossD_fake, torch.ones_like(lossD_fake))
        self.factory.generator.zero_grad()
        generator_loss.backward()
        self.factory.opt_gen.step()

        if num % 10:
            print(f"Epoch: {num} \t Discriminator Loss: {discriminator_loss} Generator Loss: {generator_loss}")
        if num % 50:
            self._eval_step()

    def _eval_step(self):
        noise = self.factory.generator(torch.randn(32, FLAGS.latent_dim).to("cuda:0"))

        fake_grid = torchvision.utils.make_grid(noise, normalize=True).cpu().numpy()
        plt.imshow(np.transpose(fake_grid, (1, 2, 0)), interpolation='nearest')

        plt.axis('off')
        plt.tight_layout()
        plt.savefig("/home/bobrin_m_s/Projects/GansInAction/PyTorch/GAN/assets/generated_grid.jpg")







