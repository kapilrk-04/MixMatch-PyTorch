'''
Implementation of the MixUp algorithm

(Standard implementation of the algorithm, part off and pre-requisite to the mixmatch algorithm)
Refer documentation for algorithm details
'''
import numpy as np
import torch


def mixup_data(inputs_x, targets_x, inputs_u, targets_u, alpha=1.0):

    # combine the labelled and unlabelled data
    # combined_inputs = torch.cat([inputs_x, inputs_u], dim=0)
    # combined_targets = torch.cat([targets_x, targets_u], dim=0)
    combined_inputs = torch.cat([inputs_x, inputs_u[0]], dim=0)
    combined_targets = torch.cat([targets_x, targets_u], dim=0)
    for i in range(1, len(inputs_u)):
        combined_inputs = torch.cat([combined_inputs, inputs_u[i]], dim=0)
        combined_targets = torch.cat([combined_targets, targets_u], dim=0)

    mixing_ratio = np.random.beta(alpha, alpha)
    mixing_ratio = max(mixing_ratio, 1 - mixing_ratio)

    shuffled_indices = torch.randperm(combined_inputs.size(0))

    input_a, input_b = combined_inputs, combined_inputs[shuffled_indices]
    target_a, target_b = combined_targets, combined_targets[shuffled_indices]

    # combine the two datapoints in the mixup ratio
    mixed_input = mixing_ratio * input_a + (1 - mixing_ratio) * input_b
    mixed_target = mixing_ratio * target_a + (1 - mixing_ratio) * target_b

    return mixed_input, mixed_target
