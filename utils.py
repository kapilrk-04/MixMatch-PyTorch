'''
Function for generating augmented images

(Originally used code, not used anymore due to use of torchvision.dataset classes.
Use these functions only for creating own dataset objects of non torchvision class
image dataset.)
'''

import numpy as np
import torch
import torchvision.transforms as transforms
import torch.utils.data as data

# Function to generate K augnmentations for an image dataset
def generate_augmentations(X, k):
    ssl_transforms = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.9, 1.0)),  # Random crop, slightly smaller than the original size
        transforms.RandomHorizontalFlip(),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        # transforms.RandomGrayscale(p=0.2),
        transforms.GaussianNoise(0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    augmented_images = []

    for _ in range(k):
        # Apply the same set of transformations to the input image k times
        augmented_img = ssl_transforms(X)
        augmented_images.append(augmented_img)

    return augmented_images

# Dataloader of augmented images created using image dataset of non torchvision class images
def augmented_data_loader(original_dataset, batch_size, k):
    augmented_dataset = []

    for i in range(len(original_dataset)):
        # Generate k augmentations for each image in the original dataset
        x, label = original_dataset[i]
        augmented_images = generate_augmentations(x, k)

        for augmented_img in augmented_images:
            augmented_dataset.append((augmented_img, label))

    augmented_loader = data.DataLoader(augmented_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

    return augmented_loader
