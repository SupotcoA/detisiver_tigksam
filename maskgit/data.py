import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
# print("torchvision.transoforms.v2 not available, using v1 instead")
import cv2
import numpy as np
import os


class ImageDataset(Dataset):
    def __init__(self, data_dir, class_names, transform=None, base_class_label=0):
        self.data_dir = data_dir
        self.class_names = class_names
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Gather image paths and labels for all classes
        for i, class_name in enumerate(class_names):
            class_dir = os.path.join(self.data_dir, class_name)
            for filename in os.listdir(class_dir):
                self.image_paths.append(os.path.join(class_dir, filename))
                self.labels.append(i + base_class_label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = np.ascontiguousarray(cv2.imread(image_path)[:, :, ::-1])  # Ensure RGB format
        label = self.labels[idx]

        image = self.transform(image)

        return image, label


class TensorDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return image, label


class InfiniteDataLoader:

    def __init__(self, *args, **kwargs):
        self.loader = DataLoader(*args, **kwargs)

    def __iter__(self):
        while True:
            for data in self.loader:
                yield data


@torch.no_grad()
def build_dataset_img(model, data_config):  # only for afhq
    # Define image directory paths
    data_dir = data_config['afhq_root']  # Replace with the actual path to your AFHQ dataset
    class_names = ["cat", "dog", "wild"]

    # Define the image transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(data_config['image_size'], antialias=True),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    # Create dataset instance
    dataset = ImageDataset(data_dir,
                           class_names,
                           transform=transform,
                           )
    # Create data loader
    data_loader = DataLoader(dataset,
                             batch_size=data_config['batch_size'],
                             shuffle=False,
                             num_workers=4)

    x, cls = None, None
    for images, labels in data_loader:
        images = images.to(model.device)
        x_ = model.encode(images)
        if x is None:
            x = x_.cpu()
            cls = labels
        else:
            x = torch.cat((x, x_.cpu()), dim=0)
            cls = torch.cat((cls, labels), dim=0)
    print(f"x shape: {x.shape}, cls shape: {cls.shape}")
    torch.save(x, data_config['x_path'])
    torch.save(cls, data_config['cls_path'])


@torch.no_grad()
def build_dataset_img_places(model, data_config):
    # Define image directory paths
    data_dir = data_config['places_root']  # Replace with the actual path to your AFHQ dataset
    class_names = ['art_studio', 'attic', 'balcony', 'bedroom', 'childs_room',
                   'closet', 'conference_room', 'corridor', 'dining_room', 'hospital_room',
                   'kitchen', 'laboratory', 'lecture_room', 'library', 'living_room',
                   'music_studio', 'office', 'television_studio', 'yard']

    # Define the image transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Resize(data_config['image_size'], antialias=True),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    # Create dataset instance
    dataset = ImageDataset(data_dir,
                           class_names,
                           transform=transform,
                           base_class_label=3
                           )
    # Create data loader
    data_loader = DataLoader(dataset,
                             batch_size=data_config['batch_size'],
                             shuffle=False,
                             num_workers=4)

    c = 0
    x, cls = None, None
    for images, labels in data_loader:
        c += 1
        if c % 1000 == 0:
            print(f"encoding {c}th batch.")
        images = images.to(model.device)
        x_ = model.encode(images)
        if x is None:
            x = x_.cpu()
            cls = labels
        else:
            x = torch.cat((x, x_.cpu()), dim=0)
            cls = torch.cat((cls, labels), dim=0)
    print(f"x shape: {x.shape}, cls shape: {cls.shape}")
    torch.save(x, data_config['x_path'])
    torch.save(cls, data_config['cls_path'])


@torch.no_grad()
def build_cached_dataset(data_config):
    x = torch.load(data_config['x_path'])
    cls = torch.load(data_config['cls_path'])
    print(f"x shape: {x.shape}, cls shape: {cls.shape}")
    # assert x.shape[0] == 15000
    s = x.shape[0]
    split = int(s * data_config['split'])
    perm_idx = torch.randperm(x.shape[0])
    train_idx = perm_idx[:split]
    test_idx = perm_idx[split:]
    train_data = TensorDataset(x[train_idx], cls[train_idx])
    test_data = TensorDataset(x[test_idx], cls[test_idx])
    train_data_loader = InfiniteDataLoader(train_data,
                                           batch_size=data_config['batch_size'],
                                           shuffle=True,
                                           num_workers=4)
    test_data_loader = DataLoader(test_data,
                                  batch_size=data_config['batch_size'],
                                  shuffle=True,
                                  num_workers=4)
    return train_data_loader, test_data_loader
