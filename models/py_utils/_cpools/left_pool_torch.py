import torch
import torch.nn as nn

def forward(input):
    # Initialize output
    output = torch.zeros_like(input)

    # Get width
    width = input.size(3)

    # Copy the last column
    input_temp  = input[:, :, :, width - 1]
    output_temp = output[:, :, :, width - 1]
    output_temp.copy_(input_temp)

    for ind in range(1, width):
        input_temp  = input[:, :, :, width - ind - 1]
        output_temp = output[:, :, :, width - ind]
        max_temp    = output[:, :, :, width - ind - 1]

        torch.max(input_temp, output_temp, out=max_temp)

    return output

# not tested!
def backward(input, grad_output):
    output = torch.zeros_like(input)

    batch, channel, height, width = input.size()

    max_val = torch.zeros(batch, channel, height, device=input.device, dtype=torch.float32)
    max_ind = torch.zeros(batch, channel, height, device=input.device, dtype=torch.int64)

    max_val = input[:, :, :, width - 1].clone()
    max_ind.fill_(width - 1)

    output[:, :, :, width - 1] = grad_output[:, :, :, width - 1].clone()

    for ind in range(1, width):
        input_temp = input[:, :, :, width - ind - 1]
        gt_mask = input_temp > max_val
        max_temp = torch.zeros_like(max_val)
        torch.masked_select(input_temp, gt_mask, out=max_temp)
        max_val[gt_mask] = max_temp
        max_ind[gt_mask] = width - ind - 1
        output[:, :, :, max_ind] += grad_output[:, :, :, width - ind - 1].unsqueeze(3)

    return output
