import torch
import torch.nn as nn
import torch.nn.functional as F


class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class MemoryEfficientMish(nn.Module):
    class MishFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return x * torch.tanh(F.softplus(x))

        @staticmethod
        def backward(ctx, grad_output):
            x = ctx.saved_tensors[0]
            sx = torch.sigmoid(x)
            fx = F.softplus(x)
            tanh_fx = torch.tanh(fx)
            return grad_output * (tanh_fx + x * sx * (1 - tanh_fx * tanh_fx))

    def forward(self, x):
        return self.MishFunction.apply(x)
