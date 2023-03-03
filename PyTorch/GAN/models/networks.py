from absl import flags
import torch.nn as nn
import einops

import torch

FLAGS = flags.FLAGS


class Generator(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),

            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),

            nn.Linear(1024, FLAGS.image_size **2 * 1),
            nn.Tanh()
        )

    def forward(self, noise_vec: torch.Tensor) -> torch.Tensor:
        x = self.net(noise_vec)
        return einops.rearrange(x, 'b (h w c) -> b c h w', h=28, w=28, c=1)
    

class Discriminator(nn.Module):
    def __init__(self, input_size: int) -> None:
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_size ** 2 * 1, 512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = image.view(-1, 784)
        #x = torch.flatten(image, start_dim=1, end_dim=-1)
        x = self.net(x)
        return x



if __name__ == "__main__":
    gen = Generator(latent_dim=100)
    x = torch.randn((2, 100))
    print(gen(x).shape)
