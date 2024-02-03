import numpy as np
import torch
from progress.bar import Bar


def test_model(model, val_loader, val_criteria, cuda_available):

    val_iter = iter(val_loader)
    len_val_iter = len(val_iter)

    progress_bar = Bar('Testing', max=len_val_iter)

    model.eval()

    total = 0
    correct = 0

    with torch.no_grad():
        for i in range(len_val_iter):

            inputs, targets = next(val_iter)

            if cuda_available:
                inputs, targets = inputs.cuda(), targets.cuda()

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += targets.size(0)
            correct += (predicted == targets).sum().item()

            progress_bar.next()
    progress_bar.finish()

    return 100.*correct/total
