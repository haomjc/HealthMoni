import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveBN:
    def __init__(self, model, use_moving_average=False, ma_decay=0.1):
        self.model = model
        self.use_moving_average = use_moving_average
        self.ma_decay = ma_decay
        self.original_stats = {}

    def apply(self, fn):
        for module in self.model.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                fn(module)

    def reset_bn(self, module):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)
        module.num_batches_tracked.zero_()

    def update_bn_stats(self, inputs, module):
        if self.use_moving_average:
            with torch.no_grad():
                batch_mean = inputs.mean([0, 2, 3])
                batch_var = inputs.var([0, 2, 3], unbiased=False)
                module.running_mean = (1 - self.ma_decay) * module.running_mean + self.ma_decay * batch_mean
                module.running_var = (1 - self.ma_decay) * module.running_var + self.ma_decay * batch_var
        else:
            module(inputs)

    def update_stats_with_target_data(self, target_loader):
        self.apply(self.reset_bn)
        self.model.train()
        device = next(self.model.parameters()).device  # 获取模型所在的设备
        with torch.no_grad():
            for inputs, _ in target_loader:
                inputs = inputs.to(device)  # 确保输入数据在正确的设备上
                for module in self.model.modules():
                    if isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                        self.update_bn_stats(inputs, module)

    def save_original_stats(self):
        # 保存原始BN层统计数据
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                self.original_stats[name] = (module.running_mean.clone(), module.running_var.clone(), module.num_batches_tracked.clone())

    def restore_original_stats(self):
        # 恢复原始BN层统计数据
        for name, module in self.model.named_modules():
            if name in self.original_stats and isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                module.running_mean, module.running_var, module.num_batches_tracked = self.original_stats[name]
