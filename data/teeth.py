from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from torchvision import transforms
import glob, os
import torch
import pytorch_lightning as pl


class ImageDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        self.files = glob.glob(os.path.join(path, '*.pth'))
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            # transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return self.preprocess(torch.load(self.files[idx]))


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)


class ImageDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        # self.content_images = ImageDataset(os.path.join(self.data_dir, 'content'))
        # self.content_images, _ = random_split(self.content_images, [12, len(self.content_images)-12])
        self.style_images = ImageDataset(os.path.join(self.data_dir, 'style'))
        # self.style_images, _ = random_split(self.style_images, [12, len(self.style_images)-12])

    def train_dataloader(self):
        return DataLoader(self.style_images, batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers,
                          pin_memory=True)
