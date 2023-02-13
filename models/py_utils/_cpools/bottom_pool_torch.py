import torch
import torch.nn as nn

def forward(input):
    # Initialize output
    output = torch.zeros_like(input)

    # Get height
    height = input.size(2)

    # Copy the last column
    input_temp = input[:, :, 0].clone()
    output[:, :, 0] = input_temp

    # Loop through the rest of the columns
    for ind in range(height - 1):
        # Get the current and next column
        input_temp = input[:, :, ind + 1]
        output_temp = output[:, :, ind]

        max_temp = output[:, :, ind + 1]
        # Get max of the two columns and store in output
        torch.max(input_temp, output_temp, out=max_temp)

    return output

# not tested!
def backward(grad_output):
    # Initialize output
    output = torch.zeros_like(input)

    batch, channel, height, width = input.size()

    max_val = torch.zeros((batch, channel, width), device=input.device, dtype=torch.float32)
    max_ind = torch.zeros((batch, channel, width), device=input.device, dtype=torch.int64)

    max_val[:, :, :] = input[:, :, 0, :]
    max_ind.fill_(0)

    output[:, :, 0, :] = grad_output[:, :, 0, :]

    un_max_ind = max_ind.unsqueeze(2)
    gt_mask = torch.zeros((batch, channel, width), device=input.device, dtype=torch.uint8)
    max_temp = torch.zeros((batch, channel, width), device=input.device, dtype=torch.float32)
    for ind in range(height - 1):
        input_temp = input[:, :, ind + 1, :]
        gt_mask = input_temp > max_val
        max_temp[gt_mask] = input_temp[gt_mask]
        max_val[gt_mask] = max_temp[gt_mask]
        max_ind[gt_mask] = ind + 1
        output[:, :, un_max_ind[gt_mask].squeeze(2), :] += grad_output[:, :, ind + 1, :][gt_mask].unsqueeze(2)

    return output
