'''
Function for training the model
'''

import numpy as np
import torch
import time
from progress.bar import Bar

from mixup import mixup_data

'''
Adapted and refined from pytorch/examples/imagenet/main.py
We initialise directly to calculate the average loss and accuracy, in place of initialising name, fmt etc.
'''
class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# Returns a list of offsets, which represent the starting indices for each group in an interleaved sequence
def calculate_interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    return offsets

# Returns a list of tensors interleaved based on the calculated offsets
def interleave_tensors_by_offsets(tensor_list, batch_size):
    nu = len(tensor_list) - 1
    offsets = calculate_interleave_offsets(batch_size, nu)
    tensor_list = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in tensor_list]
    for i in range(1, nu + 1):
        tensor_list[0][i], tensor_list[i][i] = tensor_list[i][i], tensor_list[0][i]
    return [torch.cat(v, dim=0) for v in tensor_list]


'''
Training loop of each epoch, consisting of mixup on the labelled and unlabelled data, and then training the model on the output of the mixed data
'''
def train_model(args, num_classes, model, ema_model, optimizer, ema_optimizer, training_criteria, train_labeled_loader, train_unlabeled_loader, cuda_available, epoch):

    # progress bar maintained for each epoch for monitoring progress
    progress_bar = Bar('Training', max=args.iterations)
    train_labeled_iter = iter(train_labeled_loader)
    train_unlabeled_iter = iter(train_unlabeled_loader)

    len_train_labeled_iter = len(train_labeled_iter)
    len_train_unlabeled_iter = len(train_unlabeled_iter)

    model.train()

    start_time = time.time()

    # for batch_idx, train in enumerate(zip(train_labeled_iter, train_unlabeled_iter)):
    for batch_idx in range(args.iterations):
        '''
        the following code is used to iterate over the labelled and unlabelled data loaders, and then mixup the data
        train_labeled_x and train_labeled_y are the labelled data
        train_unlabeled_x and train_unlabeled_y, train_unlabeled_x2 and train_unlabeled_y2 are the unlabelled data
        We can change the ratio of labelled to unlabelled data in the mixup algorithm to change the K value in the mixmatch step
        Default is 2:1 in favour of unlabelled data
        '''
        if batch_idx % len_train_labeled_iter == 0:
            train_labeled_iter = iter(train_labeled_loader)
        if batch_idx % len_train_unlabeled_iter == 0:
            train_unlabeled_iter = iter(train_unlabeled_loader)

        train_labeled = next(train_labeled_iter)
        train_unlabeled = next(train_unlabeled_iter)

        train_labeled_x, train_labeled_y = train_labeled
        train_unlabeled_x, train_unlabeled_y = train_unlabeled

        #print(num_classes)
        train_labeled_y = torch.zeros(args.batch_size, num_classes).scatter_(1, train_labeled_y.view(-1,1).long(), 1)

        if cuda_available:
            train_labeled_x, train_labeled_y = train_labeled_x.cuda(), train_labeled_y.cuda(non_blocking=True)
            # for every element in train_unlabeled_x apply cuda
            train_unlabeled_x = [train_unlabeled_x[i].cuda() for i in range(len(train_unlabeled_x))]

        # calculate the output of the model on the unlabeled data on each augmented version of the unlabled input
        with torch.no_grad():
            train_unlabeled_y = [model(train_unlabeled_x[i]) for i in range(len(train_unlabeled_x))]
            prob_unlabeled_y = sum([torch.softmax(train_unlabeled_y[i], dim=1) for i in range(len(train_unlabeled_y))]) / len(train_unlabeled_y)
            prob = prob_unlabeled_y**(1/args.temperature)
            prob = prob / prob.sum(dim=1, keepdim=True)
            train_targets = prob.detach()

        # mixup the labelled and unlabelled data (standard algorithm, previous and pre-requisite to mixmatch)
        mixed_input, mixed_target = mixup_data(train_labeled_x, train_labeled_y, train_unlabeled_x, train_targets, alpha=args.alpha)

        mixed_input = list(torch.split(mixed_input, args.batch_size))
        mixed_input = interleave_tensors_by_offsets(mixed_input, args.batch_size)

        logits = [model(mixed_input[0])]
        for input in mixed_input[1:]:
            logits.append(model(input))

        # interleave the logits of the labelled and unlabelled data
        logits = interleave_tensors_by_offsets(logits, args.batch_size)
        logits_x = logits[0]
        logits_u = torch.cat(logits[1:], dim=0)

        # calculate the loss on the mixed data, for training the model
        loss = training_criteria.forward(logits_x, mixed_target[:args.batch_size], logits_u, mixed_target[args.batch_size:], epoch + batch_idx / args.iterations)

        # update the model parameters (end of learning in each iteration)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema_optimizer.step()

        curr_time = time.time()

        time_elapsed = curr_time - start_time
        e_time_left = time_elapsed / (batch_idx + 1) * (args.iterations - batch_idx - 1)

        progress_bar.suffix = '({batch}/{size}) | Time: {time}s | ETA: {eta}s | Loss: {loss}'.format(batch=batch_idx + 1, size=args.iterations, time=round(time_elapsed, 2), eta=round(e_time_left, 2), loss=round(loss.item(), 4))
        progress_bar.next()

    progress_bar.finish()

    return loss

