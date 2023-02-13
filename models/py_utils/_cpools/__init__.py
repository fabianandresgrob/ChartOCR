import torch

from torch import nn
from torch.autograd import Function

# import top_pool, bottom_pool, left_pool, right_pool
from . import top_pool_torch, bottom_pool_torch, left_pool_torch, right_pool_torch

class TopPoolFunction(Function):
    @staticmethod
    def forward(ctx, input):
        # output = top_pool.forward(input)[0]
        output = top_pool_torch.forward(input)
        ctx.save_for_backward(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input  = ctx.saved_variables[0]
        # output = top_pool.backward(input, grad_output)[0]
        output = top_pool_torch.backward(input, grad_output)
        return output

class BottomPoolFunction(Function):
    @staticmethod
    def forward(ctx, input):
        # output = bottom_pool.forward(input)[0]
        output = bottom_pool_torch.forward(input)
        ctx.save_for_backward(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input  = ctx.saved_variables[0]
        # output = bottom_pool.backward(input, grad_output)[0]
        output = bottom_pool_torch.backward(input, grad_output)
        return output

class LeftPoolFunction(Function):
    @staticmethod
    def forward(ctx, input):
        # output = left_pool.forward(input)[0]
        output = left_pool_torch.forward(input)
        ctx.save_for_backward(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input  = ctx.saved_variables[0]
        # output = left_pool.backward(input, grad_output)[0]
        output = left_pool_torch.backward(input, grad_output)
        return output

class RightPoolFunction(Function):
    @staticmethod
    def forward(ctx, input):
        # output = right_pool.forward(input)[0]
        output = right_pool_torch.forward(input)
        ctx.save_for_backward(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input  = ctx.saved_variables[0]
        # output = right_pool.backward(input, grad_output)[0]
        output = right_pool_torch.backward(input, grad_output)
        return output

class TopPool(nn.Module):
    def forward(self, x):
        return TopPoolFunction.apply(x)

class BottomPool(nn.Module):
    def forward(self, x):
        return BottomPoolFunction.apply(x)

class LeftPool(nn.Module):
    def forward(self, x):
        return LeftPoolFunction.apply(x)

class RightPool(nn.Module):
    def forward(self, x):
        return RightPoolFunction.apply(x)


# # Now for the torch version
# class TopPoolTorch(nn.Module):
#     def forward(self, x):
#         return TopPoolFunctionTorch.apply(x)

# class BottomPoolTorch(nn.Module):
#     def forward(self, x):
#         return BottomPoolFunctionTorch.apply(x)

# class LeftPoolTorch(nn.Module):
#     def forward(self, x):
#         return LeftPoolFunctionTorch.apply(x)

# class RightPoolTorch(nn.Module):
#     def forward(self, x):
#         return RightPoolFunctionTorch.apply(x)


# # now functions
# class TopPoolFunctionTorch(Function):
#     @staticmethod
#     def forward(ctx, input):
#         output = top_pool_torch.forward(input)
#         ctx.save_for_backward(input)
#         return output

#     @staticmethod
#     def backward(ctx, grad_output):
#         input  = ctx.saved_variables[0]
#         output = top_pool_torch.backward(input, grad_output)
#         return output

# class BottomPoolFunctionTorch(Function):
#     @staticmethod
#     def forward(ctx, input):
#         output = bottom_pool_torch.forward(input)
#         ctx.save_for_backward(input)
#         return output

#     @staticmethod
#     def backward(ctx, grad_output):
#         input  = ctx.saved_variables[0]
#         output = bottom_pool_torch.backward(input, grad_output)
#         return output

# class LeftPoolFunctionTorch(Function):
#     @staticmethod
#     def forward(ctx, input):
#         output = left_pool_torch.forward(input)
#         ctx.save_for_backward(input)
#         return output

#     @staticmethod
#     def backward(ctx, grad_output):
#         input  = ctx.saved_variables[0]
#         output = left_pool_torch.backward(input, grad_output)
#         return output

# class RightPoolFunctionTorch(Function):
#     @staticmethod
#     def forward(ctx, input):
#         output = right_pool_torch.forward(input)
#         ctx.save_for_backward(input)
#         return output

#     @staticmethod
#     def backward(ctx, grad_output):
#         input  = ctx.saved_variables[0]
#         output = right_pool_torch.backward(input, grad_output)
#         return output