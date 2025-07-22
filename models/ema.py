import torch


class SimpleEMAModel:
    """simple exponential moving average model"""

    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.ema_params = {}
        self.temp_stored_params = {}
        self.decay = decay

        # initialize EMA parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.ema_params[name] = param.clone().detach()
            else:
                self.ema_params[name] = param

    @torch.inference_mode()
    def step(self, model: torch.nn.Module):
        """update EMA parameters with current model parameters."""
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.ema_params[name].mul_(self.decay).add_(param, alpha=1 - self.decay)
            else:
                self.ema_params[name].copy_(param)

    def copy_to(self, model: torch.nn.Module) -> None:
        """copy current averaged parameters into given model."""
        for name, param in model.named_parameters():
            param.data.copy_(self.ema_params[name].to(param.device).data)

    def to(self, device=None, dtype=None) -> None:
        """move internal buffers to specified device."""
        # .to() on the tensors handles None correctly
        for name, param in self.ema_params.items():
            self.ema_params[name] = (
                self.ema_params[name].to(device=device, dtype=dtype)
                if self.ema_params[name].is_floating_point()
                else self.ema_params[name].to(device=device)
            )

    def store(self, model: torch.nn.Module) -> None:
        """store current model parameters temporarily."""
        for name, param in model.named_parameters():
            self.temp_stored_params[name] = param.detach().cpu().clone()

    def restore(self, model: torch.nn.Module) -> None:
        """restore parameters stored with the store method."""
        if self.temp_stored_params is None:
            raise RuntimeError("This ExponentialMovingAverage has no `store()`ed weights to `restore()`")

        for name, param in model.named_parameters():
            assert name in self.temp_stored_params, f"{name} not found in temp_stored_params"
            param.data.copy_(self.temp_stored_params[name].data)
        self.temp_stored_params = {}

    def load_state_dict(self, state_dict: dict | list) -> None:
        """load EMA state from state dict."""
        if isinstance(state_dict, dict):
            for name, param in self.ema_params.items():
                param.data.copy_(state_dict[name].to(param.device).data)
        elif isinstance(state_dict, list):
            i = 0
            for name, param in self.ema_params.items():
                param.data.copy_(state_dict[i].to(param.device).data)
                i += 1
        else:
            raise ValueError("state_dict must be a dict or list")

    def state_dict(self) -> dict:
        """return EMA parameters as state dict."""
        return self.ema_params
