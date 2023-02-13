import torch
import torch.nn as nn

def forward(input):
    # Initialize output
    output = torch.zeros_like(input)

    # Get height
    height = input.size(2)

    # Copy the last column
    input_temp = input[:, :, height-1]
    output_temp = output[:, :, height-1]
    output_temp.copy_(input_temp)

    max_temp = None
    for ind in range(1, height):
        input_temp = input[:, :, height-ind-1]
        output_temp = output[:, :, height-ind]
        max_temp = output[:, :, height-ind-1]

        torch.max(input_temp, output_temp, out=max_temp)

    return output

# not tested!
def backward(input, grad_output):
    output = torch.zeros_like(input)

    batch, channel, height, width = input.size()

    max_val = torch.zeros(input.shape[0:3] + (width,), dtype=torch.float32, device=input.device)
    max_ind = torch.zeros(input.shape[0:3] + (width,), dtype=torch.int64, device=input.device)

    input_temp = input[:, :, height-1]
    max_val.copy_(input_temp)

    max_ind.fill_(height-1)

    output_temp = output[:, :, height-1]
    grad_output_temp = grad_output[:, :, height-1]
    output_temp.copy_(grad_output_temp)

    un_max_ind = max_ind.unsqueeze(2)
    gt_mask = torch.zeros(input.shape[0:3] + (width,), dtype=torch.bool, device=input.device)
    max_temp = torch.zeros(input.shape[0:3] + (width,), dtype=torch.float32, device=input.device)
    for ind in range(1, height):
        input_temp = input[:, :, height-ind-1]
        gt_mask = (input_temp > max_val)
        max_temp = torch.where(gt_mask, input_temp, max_temp)
        max_val = torch.where(gt_mask, max_temp, max_val)
        max_ind = torch.where(gt_mask, height-ind-1, max_ind)
        grad_output_temp = grad_output[:, :, height-ind-1].unsqueeze(2)
        output.scatter_add_(2, un_max_ind, grad_output_temp)

    return output
