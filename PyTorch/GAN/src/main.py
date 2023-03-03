import sys
sys.path.append("/home/bobrin_m_s/Projects/GansInAction/PyTorch/GAN")

from absl import app, flags
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from models.networks import Generator, Discriminator
import os
import typing as tp
import torchvision.transforms as transforms
import torchvision
import numpy as np

from gan_training.train import GanTrainer

FLAGS = flags.FLAGS

flags.DEFINE_string("dataset_path", default="/home/bobrin_m_s/Projects/GansInAction/data/images", help="Path to the dataset.")
## for AnimeFaces dataset, size of each image is 64x64

flags.DEFINE_integer("latent_dim", default=100, help="Dimension of latent space.")
flags.DEFINE_enum("logger", "Tensorboard", ["Wandb", "Tensorboard"], help="Logger to use.")
flags.DEFINE_integer("image_size", default=28, help="Image size of all images in dataset.")
flags.DEFINE_integer("batch_size", default=32, help="Batch size to use.")
flags.DEFINE_bool("visualize", default=True, help="Whether to save assets.")
flags.DEFINE_integer("num_epochs", default=1, help="Number of training epochs.")

class AnimeDS(Dataset):
    def __init__(self, path_to_ds: str,
                 augs: transforms.Compose) -> None:
        
        super().__init__()
        self.dataset = os.listdir(path_to_ds)
        self.augs = augs
        self.path_to_ds = path_to_ds

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index) -> tp.Any:
        cur_image = Image.open(os.path.join(self.path_to_ds, self.dataset[index]))
        if self.augs is not None:
            cur_image = self.augs(cur_image)
        return cur_image

class GanFactory:
    def __init__(self) -> None:
        self.generator = Generator(latent_dim=FLAGS.latent_dim).to("cuda:0")
        self.noise_vector = torch.randn((FLAGS.batch_size, FLAGS.latent_dim)).to("cuda:0")

        self.discriminator = Discriminator(input_size=FLAGS.image_size).to("cuda:0")
        self.criterion = torch.nn.BCELoss().to("cuda:0")

        self._build_factory()

    def _build_factory(self):
        self.opt_discr = torch.optim.AdamW(self.discriminator.parameters(), lr=3e-4)
        self.opt_gen = torch.optim.AdamW(self.generator.parameters(), lr=3e-4)

def main(_):
    augmentations = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(FLAGS.image_size),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    generative_factory = GanFactory()
    animedataset = torchvision.datasets.MNIST(root="dataset/", transform=augmentations, download=True)
    animeloader = torch.utils.data.DataLoader(animedataset, batch_size=FLAGS.batch_size, shuffle=True)

    #animedataset = AnimeDS(FLAGS.dataset_path, augs=augmentations)
    #animeloader = DataLoader(animedataset, batch_size=FLAGS.batch_size, shuffle=True, pin_memory=True)

    if FLAGS.visualize:
        example = next(iter(animeloader))[0]
        grid = torchvision.utils.make_grid(example, normalize=True).cpu().numpy()
        plt.imshow(np.transpose(grid, (1, 2, 0)), interpolation='nearest')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig("/home/bobrin_m_s/Projects/GansInAction/PyTorch/GAN/assets/grid.jpg")

    trainer = GanTrainer(factory=generative_factory)
    for epoch in range(FLAGS.num_epochs):
        for num, (image_sample, _) in enumerate(animeloader):
            trainer._train_step(image_sample, num=num, epoch=epoch)

    

if __name__ == "__main__":
    app.run(main)