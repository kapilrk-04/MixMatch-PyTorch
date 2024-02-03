'''
Main function to run MixMatch with data loading, model training, etc.
'''

import numpy as np
import torch
import torchvision.transforms as transforms
import argparse
import os
# import sys
# sys.path.append("/dataset_loaders/")
import dataloader
import model as Model
from losses import MixMatchLoss
from optimization import WeightEMA
from training_utils import train_model
from evaluation import test_model
from utils import augmented_data_loader
import dataset_loaders.loader as dataset
import dataset_loaders.utils as data_utils
import torch.utils.data as data


def parse_args():
    '''
    arguments for the main function
    1. dataset: dataset to use for training
    2. seed: seed for random number generator
    3. labeled_n: number of labelled examples
    4. temperature: temperature for sharpening
    5. K: number of augmentations for each image
    6. alpha: alpha for beta distribution
    7. lambda_u: lambda_u for training
    8. ema_decay: ema decay for training
    9. batch_size: batch size for training
    10. epochs: number of epochs for training
    11. iterations: number of iterations for training
    12. lr: learning rate for training
    13. gpu: GPU to use for training
    '''
    parser = argparse.ArgumentParser(description='MixMatch')
    parser.add_argument('--dataset', type=str, default='cifar10', metavar='DS',
                        help='dataset to use for training (default: cifar10)')
    parser.add_argument('--seed', type=int, default=0, metavar='S',
                        help='seed for random number generator (default: 0)')
    parser.add_argument('--labeled_n', type=int, default=250, metavar='LN',
                        help='number of labelled examples (default: 250)')
    parser.add_argument('--temperature', type=float, default=0.5, metavar='T',
                        help='temperature for sharpening (default: 0.5)')
    parser.add_argument('--K', type=int, default=2, metavar='K',
                        help='number of augmentations for each image (default: 2)')
    parser.add_argument('--alpha', type=float, default=0.75, metavar='A',
                        help='alpha for beta distribution (default: 0.75)')
    parser.add_argument('--lambda_u', type=float, default=75, metavar='LU',
                        help='lambda_u for training (default: 75)')
    parser.add_argument('--ema_decay', type=float, default=0.999, metavar='ED',
                        help='ema decay for training (default: 0.999)')
    parser.add_argument('--batch_size', type=int, default=64, metavar='BS',
                        help='batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=1000, metavar='E',
                        help='number of epochs for training (default: 1000)')
    parser.add_argument('--iterations', type=int, default=1000, metavar='I',
                        help='number of iterations for training (default: 1000)')
    parser.add_argument('--lr', type=float, default=0.002, metavar='LR',
                        help='learning rate for training (default: 0.002)')
    parser.add_argument('--gpu', type=str, default='0', metavar='GPU',
                        help='GPU to use for training (default: 0)')
    args = parser.parse_args()

    return args


def main(args = None):

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    cuda_available = torch.cuda.is_available()
    print('CUDA Available: {}'.format(cuda_available))

    '''
    Augmentation transformations for the dataset
    Since most image datasets tested are B&W, we combine all possible augmentations into Compose and apply them to the image data loaders
    '''
    transform_train = transforms.Compose([
        data_utils.RandomPadandCrop(32),
        data_utils.RandomFlip(),
        data_utils.ToTensor(),
    ])
    transform_val = transforms.Compose([
        data_utils.ToTensor(),
    ])

    # Creating the data loaders for each split of the dataset (refer get_data command to change loading parameters)
    train_labeled_set, train_unlabeled_set, val_set, test_set = dataset.get_data('./data', args.dataset, args.labeled_n, args.K, transform_train=transform_train, transform_val=transform_val) # command to load dataset, specified in dataset_loaders
    labeled_trainloader = data.DataLoader(train_labeled_set, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    unlabeled_trainloader = data.DataLoader(train_unlabeled_set, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Number of samples in each split of the dataset
    print('Number of labelled examples: {}'.format(len(train_labeled_set)))
    print('Number of unlabelled examples: {}'.format(len(train_unlabeled_set)))
    print('Number of validation examples: {}'.format(len(val_set)))
    print('Number of test examples: {}'.format(len(test_set)))

    num_classes = len(train_labeled_set.classes)

    model = Model.WideResNet(num_classes=10)
    ema_model = Model.WideResNet(num_classes=10)

    for param in ema_model.parameters():
        param.detach_()
    if cuda_available:
        model.cuda()
        ema_model.cuda()

    # Declaring required optimizers and loss functions for the model to train and evaluate on (refer to docs for more details)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    ema_optimizer = WeightEMA(model, ema_model, alpha=args.ema_decay, weight_decay=args.lr*0.02)
    training_criteria = MixMatchLoss(alpha=args.alpha, lambda_u=args.lambda_u, num_classes=num_classes, temperature=args.temperature, total_epochs=args.epochs)
    val_criteria = torch.nn.CrossEntropyLoss()

    ema_optimizer.apply()

    # each epoch consists of 1000 iterations, each iteration consists of entire mixmatch step
    for epoch in range(args.epochs):

        print('Epoch: [{}/{}]'.format(epoch+1, args.epochs))

        # training function followed by evaluation after each epoch, to track evaluation and performance over time
        loss = train_model(args, num_classes, model, ema_model, optimizer, ema_optimizer, training_criteria, labeled_trainloader, unlabeled_trainloader, cuda_available, epoch)
        self_accuraacy = test_model(ema_model, labeled_trainloader, val_criteria, cuda_available)
        accuracy = test_model(ema_model, val_loader, val_criteria, cuda_available)

        print('Epoch: [{}/{}]\tLoss: {}\tSelf Accuracy: {}\tAccuracy: {}\n'.format(epoch+1, args.epochs, loss, self_accuraacy, accuracy))


if __name__ == '__main__':
    args = parse_args()
    main(args)
