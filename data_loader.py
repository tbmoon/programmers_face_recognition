import os
import pandas as pd
import torch
import torch.utils.data as data
from skimage import io
from torchvision import transforms


class FaceDataset(data.Dataset):
    def __init__(self, input_dir, phase, transform=None):
        self.input_dir = input_dir
        self.df = pd.read_csv(os.path.join(input_dir, phase + '.csv'))
        self.load_label = True if phase is not 'test' else False
        self.transform = transform

    def __getitem__(self, idx):
        image_path = os.path.join(self.input_dir, 'faces_images', self.df['filename'][idx])
        image = io.imread(image_path)

        if self.transform:
            image = self.transform(image)

        if self.load_label:
            label = self.df['label'][idx] - 1

        return {'image': image, 'label': label}

    def __len__(self):
        return len(self.df)


def get_dataloader(
    input_dir,
    phases,
    batch_size,
    num_workers):

    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(128),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation((-20, 20)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    face_datasets = {
        phase: FaceDataset(
            input_dir,
            phase,
            data_transforms)
        for phase in phases}

    data_loaders = {
        phase: torch.utils.data.DataLoader(
            dataset=face_datasets[phase],
            batch_size=batch_size,
            shuffle=False if phase is not 'test' else False,
            num_workers=num_workers,
            pin_memory=False)
        for phase in phases}

    data_sizes = {
        phase: len(face_datasets[phase])
        for phase in phases}

    return data_loaders, data_sizes
