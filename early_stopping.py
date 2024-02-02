import numpy as np
import torch


class EarlyStopping:
    """如果验证集损失在一定的周期内不再改善，则提前停止训练。"""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print,
                 save_full_model=False):
        """
        参数:
            patience (int): 在停止训练前，验证损失不再改善的最大周期数。默认为7。
            verbose (bool): 如果为 True，则在每次验证损失改善时打印消息。默认为 False。
            delta (float): 视为损失改善的最小变化量。默认为0。
            path (str): 保存模型检查点的路径。默认为 'checkpoint.pt'。
            trace_func (function): 用于日志记录的函数。默认为 print 函数。
            save_full_model (bool): 是否保存完整的模型而不仅是状态字典。默认为 False。
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.save_full_model = save_full_model

    def __call__(self, val_loss, model):
        """在每个训练周期后调用，以检查验证损失是否改善。"""
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'早停计数器: {self.counter} / {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """当验证损失减少时保存模型。"""
        if self.verbose:
            self.trace_func(
                f'验证损失减少 ({self.val_loss_min:.6f} --> {val_loss:.6f})。 正在保存模型...')
        if self.save_full_model:
            torch.save(model, self.path)  # 保存完整的模型
        else:
            # torch.save(model.state_dict(), self.path)  # 仅保存模型状态字典
            pass
        self.val_loss_min = val_loss

    def load_checkpoint(self, model):
        """加载保存的最佳模型检查点。"""
        if self.save_full_model:
            model = torch.load(self.path)
        else:
            model.load_state_dict(torch.load(self.path))
        return model
