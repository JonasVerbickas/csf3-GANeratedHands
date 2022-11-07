import torch
import torchvision.transforms as T
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

class CenterCropImageDataset(Dataset):
    def __init__(self, dataset_dir):
        self.image_filenames = [os.path.join(dataset_dir, x) for x in os.listdir(dataset_dir) if '.png' in x]

    def __getitem__(self, index):
        image = Image.open(self.image_filenames[index])
        image_as_tensor = T.ToTensor()(image)
        resized_tensor = T.Resize((286, 286), interpolation=T.InterpolationMode.BICUBIC, antialias=True)(
            image_as_tensor)
        resized_and_cropped = T.CenterCrop((256, 256))(resized_tensor)
        return resized_and_cropped

    def __len__(self):
        return len(self.image_filenames)


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)


class HandsDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def setup(self, stage=None):
        synth_hands_dataset = CenterCropImageDataset("./synth_all_male_noobject_with_white_bg")
        real_hands_dataset = CenterCropImageDataset("./RealHandsForKagglev2")
        concat_dataset = ConcatDataset([synth_hands_dataset, real_hands_dataset])
        # Assign train/val datasets for use in dataloaders
        train_size = round(len(concat_dataset) * self.config['TRAIN_SPLIT'])
        validation_size = len(concat_dataset) - train_size
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(concat_dataset,
                                                                             [train_size, validation_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.config["BATCH_SIZE"], shuffle=True,
                          num_workers=self.config["CPU_COUNT"], pin_memory=self.config["PIN_MEMORY"], drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.config["BATCH_SIZE"], shuffle=False,
                          num_workers=self.config["CPU_COUNT"], pin_memory=self.config["PIN_MEMORY"], drop_last=True)
