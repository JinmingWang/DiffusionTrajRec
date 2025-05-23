# encoding: utf-8

import torch
from typing import *
from tqdm import tqdm
from math import log

Tensor = torch.Tensor

class DDIM:
    """
    DDIM (Denoising Diffusion Implicit Models) class for diffusion-based generative modeling.

    Attributes:
        sample_shape (list[int]): Shape of the samples to be generated.
        min_beta (float): Minimum beta value for the diffusion process.
        max_beta (float): Maximum beta value for the diffusion process.
        max_diffusion_step (int): Total number of diffusion steps.
        device (str): Device to perform computations on ('cuda' or 'cpu').
        scale_mode (Literal["linear", "quadratic", "log"]): Mode for scaling beta values.
        skip_step (int): Number of steps to skip during denoising.
    """
    def __init__(self,
                 min_beta: float = 0.0001,
                 max_beta: float = 0.002,
                 max_diffusion_step: int = 100,
                 device: str = 'cuda',
                 scale_mode: Literal["linear", "quadratic", "log"] = "linear",
                 skip_step=1):
        """
        Initializes the DDIM model with the given parameters.

        :param sample_shape: Shape of the samples to be generated.
        :param min_beta: Minimum beta value for the diffusion process.
        :param max_beta: Maximum beta value for the diffusion process.
        :param max_diffusion_step: Total number of diffusion steps.
        :param device: Device to perform computations on ('cuda' or 'cpu').
        :param scale_mode: Mode for scaling beta values.
        :param skip_step: Number of steps to skip during denoising.
        """
        if scale_mode == "quadratic":
            betas = torch.linspace(min_beta ** 0.5, max_beta ** 0.5, max_diffusion_step).to(device) ** 2
        elif scale_mode == "log":
            betas = torch.exp(torch.linspace(log(min_beta), log(max_beta), max_diffusion_step).to(device))
        else:
            betas = torch.linspace(min_beta, max_beta, max_diffusion_step).to(device)

        self.skip_step = skip_step
        self.device = device

        alphas = 1 - betas
        alpha_bars = torch.empty_like(alphas)
        product = 1
        for i, alpha in enumerate(alphas):
            product *= alpha
            alpha_bars[i] = product
        self.T = max_diffusion_step

        # Shapes: (T,)
        self.beta = betas.view(-1, 1)
        self.alpha = alphas.view(-1, 1)
        self.αbar = alpha_bars.view(-1, 1)
        self.sqrt_αbar = torch.sqrt(alpha_bars).view(-1, 1)
        self.sqrt_1_m_αbar = torch.sqrt(1 - alpha_bars).view(-1, 1)


    def diffuseStep(self, x_t: Tensor, t: int, epsilon_t_to_tp1: Tensor) -> Tensor:
        """
        Performs a single diffusion step.

        :param x_t: Current state of the sample.
        :param t: Current time step.
        :param epsilon_t_to_tp1: Noise to be added during the step.
        :return: The next state after the diffusion step.
        """
        return torch.sqrt(self.alpha[t]) * x_t + torch.sqrt(1 - self.alpha[t]) * epsilon_t_to_tp1


    def diffuse(self, x_0: Tensor, t: int, epsilon: Tensor) -> Tensor:
        """
        Diffuses the initial sample x_0 to a given time step t.

        :param x_0: Initial sample.
        :param t: Target time step.
        :param epsilon: Noise to be added.
        :return: The diffused sample at time step t.
        """
        original_shape = x_0.shape
        x_t = self.sqrt_αbar[t] * x_0.flatten(1) + self.sqrt_1_m_αbar[t] * epsilon.flatten(1)
        return x_t.view(*original_shape)


    def denoiseStep(self, x0_pred: Tensor, epsilon_pred: Tensor, x_tp1: Tensor, t: Tensor, next_t: Tensor) -> Tensor:
        """
        Performs a single denoising step.

        :param x0_pred: Predicted x_0 (original sample).
        :param epsilon_pred: Predicted noise.
        :param x_tp1: Current state of the sample.
        :param t: Current time step.
        :param next_t: Next time step.
        :return: The denoised sample at the next time step.
        """
        original_shape = epsilon_pred.shape
        if x0_pred is None:
            x0_pred = (x_tp1.flatten(1) - self.sqrt_1_m_αbar[t] * epsilon_pred.flatten(1)) / self.sqrt_αbar[t]
        else:
            x0_pred = x0_pred.flatten(1)
        epsilon_pred = epsilon_pred.flatten(1)

        # if t <= self.skip_step, then mask is 1, which means return pred_x0
        # otherwise, mask is 0, which means return diffuse
        mask = (t == 0).to(x0_pred.dtype).view(-1, 1)
        return (x0_pred * mask + self.diffuse(x0_pred, next_t, epsilon_pred) * (1 - mask)).view(*original_shape)


    @torch.no_grad()
    def denoise(self,
                x_T: Tensor,
                pred_func: Callable[[Tensor, Tensor, Any], Tuple[Tensor, Tensor]],
                verbose: bool = False,
                **pred_func_args) -> Tensor:
        """
        Denoises a sample from the final time step T to the initial time step 0.

        :param x_T: Sample at the final time step T.
        :param pred_func: Function to predict x_0 and noise.
        :param verbose: Whether to display a progress bar.
        :param pred_func_args: Additional arguments for the prediction function.
        :return: The denoised sample at time step 0.
        """
        x_t = x_T.clone()
        all_t = torch.arange(self.T, dtype=torch.long, device=self.device).repeat(x_T.shape[0], 1)  # (B, T)

        # [T, T-s, T-2s, ..., k], k >= 0
        t_schedule = list(range(self.T - 1, -1, -self.skip_step))
        if t_schedule[-1] != 0:
            t_schedule.append(0)
        # [T, T-s, T-2s, ..., 0]

        pbar = tqdm(t_schedule) if verbose else t_schedule
        for ti, t in enumerate(pbar):
            x0_pred, epsilon_pred = pred_func(x_t, all_t[:, t], **pred_func_args)
            t_next = 0 if ti + 1 == len(t_schedule) else t_schedule[ti + 1]
            x_t = self.denoiseStep(x0_pred, epsilon_pred, x_t, all_t[:, t], all_t[:, t_next])

        return x_t


    def combineNoise(self, eps_0_to_t, eps_t_to_tp1, t):
        """

        :param eps_0_to_t: Combined noise,  (B, 2, L)
        :param eps_t_to_tp1: Noise for step, (B, 2, L)
        :param t: t int {0, 1, 2, ... T-1}
        :return: eps_0_to_tp1
        """
        if t == 0:
            return eps_t_to_tp1

        original_shape = eps_0_to_t.shape

        term_1 = torch.sqrt(self.alpha[t]) * self.sqrt_1_m_αbar[t - 1] * eps_0_to_t.flatten(1)

        term_2 = torch.sqrt(1 - self.alpha[t]) * eps_t_to_tp1.flatten(1)

        return ((term_1 + term_2) / self.sqrt_1_m_αbar[t]).view(original_shape)
