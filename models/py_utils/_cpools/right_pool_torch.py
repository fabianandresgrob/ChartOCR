import torch
import torch.nn as nn

def forward(input):
    # Initialize output
    output = torch.zeros_like(input)

    # Get width
    width = input.shape[3]

    # Copy the last column
    input_temp = input[:, :, :, 0]
    output_temp = output[:, :, :, 0]
    output_temp.copy_(input_temp)

    max_temp = None
    for ind in range(width - 1):
        input_temp = input[:, :, :, ind + 1]
        output_temp = output[:, :, :, ind]
        max_temp = output[:, :, :, ind + 1]

        torch.max(input_temp, output_temp, out=max_temp)

    return output

# not tested!
def backward(grad_output):
    # Initialize output
    output = torch.zeros_like(input)

    batch, channel, height, width = input.shape

    max_val = torch.zeros((batch, channel, height), dtype=torch.float32, device=input.device)
    max_ind = torch.zeros((batch, channel, height), dtype=torch.long, device=input.device)

    input_temp = input[:, :, :, 0]
    max_val.copy_(input_temp)

    max_ind.fill_(0)

    output_temp = output[:, :, :, 0]
    grad_output_temp = grad_output[:, :, :, 0]
    output_temp.copy_(grad_output_temp)

    un_max_ind = max_ind.unsqueeze(3)
    gt_mask = torch.zeros((batch, channel, height), dtype=torch.uint8, device=input.device)
    max_temp = torch.zeros((batch, channel, height), dtype=torch.float32, device=input.device)
    for ind in range(width - 1):
        input_temp = input[:, :, :, ind + 1]
        gt_mask = (input_temp > max_val)

        max_temp = torch.masked_select(input_temp, gt_mask)
        max_val.masked_scatter_(gt_mask, max_temp)
        max_ind.masked_fill_(gt_mask, ind + 1)

        grad_output_temp = grad_output[:, :, :, ind + 1].unsqueeze(3)
        output.scatter_add_(3, un_max_ind, grad_output_temp)

    return output
