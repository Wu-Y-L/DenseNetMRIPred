
import torch
import torchvision
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'
def create_dataloaders(dataset_dir : str,transforms : torchvision.transforms, BATCH_SIZE : int = 32 ):
    """_summary_

    turns data in DenseNetMRIPred/data to dataloaders

    Args:
        dataset_dir (str):
        transforms (torchvision.transforms: desired transformation
        BATCH_SIZE : batchsize of dataloaders, defaults to 32

    Returns:
        train_dataloader, test_dataloader, class_names

    sets up data in desired form to pass to model
    """

    train_dataset = torchvision.datasets.ImageFolder(root = dataset_dir / "train", transform = transforms)
    test_dataset = torchvision.datasets.ImageFolder(root = dataset_dir / "test", transform = transforms)

    train_dataloader = DataLoader(train_dataset, shuffle = True, num_workers = os.cpu_count(), batch_size = BATCH_SIZE)
    test_dataloader = DataLoader(test_dataset, shuffle = False, num_workers = os.cpu_count(), batch_size = BATCH_SIZE)

    class_names = train_dataset.classes

    return train_dataloader, test_dataloader, class_names
