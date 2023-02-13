import right_pool_torch
import left_pool_torch
import bottom_pool_torch
import top_pool_torch

import bottom_pool, top_pool, left_pool, right_pool
import torch




# Create a sample input tensor
input_tensor = torch.randn(10, 5, 7, 7)

# Pass the input tensor through the layer
print("BOTTOM POOL")
bottom_pool_output = bottom_pool.forward(input_tensor)[0]

# The shape of the output tensor should be (10, 5, 1, 1)
print(bottom_pool_output.shape)

# the output should be the same as the BottomPoolTorch layer
bottom_pool_torch_output = bottom_pool_torch.forward(input_tensor)
print(bottom_pool_torch_output.shape)
print(torch.allclose(bottom_pool_output, bottom_pool_torch_output))

# now for the top pool
print("\n\nTOP POOL")
top_pool_output = top_pool.forward(input_tensor)[0]
print(top_pool_output.shape)

top_pool_torch_output = top_pool_torch.forward(input_tensor)
print(top_pool_torch_output.shape)
print(torch.allclose(top_pool_output, top_pool_torch_output))

# now for the left pool
print("\nLEFT POOL")
left_pool_output = left_pool.forward(input_tensor)[0]
print(left_pool_output.shape)

left_pool_torch_output = left_pool_torch.forward(input_tensor)
print(left_pool_torch_output.shape)
print(torch.allclose(left_pool_output, left_pool_torch_output))

# now for the right pool
print("\n\nRIGHT POOL")
right_pool_output = right_pool.forward(input_tensor)[0]
print(right_pool_output.shape)

right_pool_torch_output = right_pool_torch.forward(input_tensor)
print(right_pool_torch_output.shape)
print(torch.allclose(right_pool_output, right_pool_torch_output))
