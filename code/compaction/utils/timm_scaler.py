import torch

def adaptive_clip_grad(parameters, clip_factor=0.01, eps=1e-3, norm_type=2.0):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    for p in parameters:
        if p.grad is None:
            continue
        p_data = p.detach()
        g_data = p.grad.detach()
        max_norm = unitwise_norm(p_data, norm_type=norm_type).clamp_(min=eps).mul_(clip_factor)
        grad_norm = unitwise_norm(g_data, norm_type=norm_type)
        clipped_grad = g_data * (max_norm / grad_norm.clamp(min=1e-6))
        new_grads = torch.where(grad_norm < max_norm, g_data, clipped_grad)
        p.grad.detach().copy_(new_grads)        


def dispatch_clip_grad(parameters, value: float, mode: str = 'norm', norm_type: float = 2.0):
    """ Dispatch to gradient clipping method
    Args:
        parameters (Iterable): model parameters to clip
        value (float): clipping value/factor/norm, mode dependant
        mode (str): clipping mode, one of 'norm', 'value', 'agc'
        norm_type (float): p-norm, default 2.0
    """
    if mode == 'norm':
        torch.nn.utils.clip_grad_norm_(parameters, value, norm_type=norm_type)
    elif mode == 'value':
        torch.nn.utils.clip_grad_value_(parameters, value)
    elif mode == 'agc':
        adaptive_clip_grad(parameters, value, norm_type=norm_type)
    else:
        assert False, f"Unknown clip mode ({mode})."        

# 对 timm 的改进，可以支持多个 optimizer 更新
class NativeScaler:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, clip_mode='norm', parameters=None, create_graph=False):
        if isinstance(loss, list):
            for l in loss:
                self._scaler.scale(l).backward()
        else:
            self._scaler.scale(loss).backward(create_graph=create_graph)
        if clip_grad is not None:
            assert parameters is not None
            if isinstance(optimizer, list):
                self._scaler.unscale_(optimizer[0])  # unscale the gradients of optimizer's assigned params in-place
                self._scaler.unscale_(optimizer[1])
            else:
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
            dispatch_clip_grad(parameters, clip_grad, mode=clip_mode)
        if isinstance(optimizer, list):
            for opt in optimizer:
                self._scaler.step(opt)
        else:
            self._scaler.step(optimizer)
        self._scaler.update()

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)

