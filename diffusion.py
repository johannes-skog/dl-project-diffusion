from signal import raise_signal
import time
import torch

def extract(a, t, x_shape):

    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())

    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

class Diffusion(object):

    def __init__(self, timesteps: int, schedular: str = "linear", **kwargs) -> None:

        self._timesteps = timesteps

        self._beta_schedule = None

        if schedular == "linear":

            self._beta_schedule = self._setup_linear_beta_schedule(**kwargs)

        elif schedular == "cosine":

            self._beta_schedule = self._setup_cosine_beta_schedule(**kwargs)

        else:

            raise NotImplementedError

        self._alphas = 1. - self._beta_schedule

        self._alphas_cumprod = torch.cumprod(self._alphas, axis=0)

        # Pad the left end by 1, do not pad the right end
        self._alphas_cumprod_prev = torch.nn.functional.pad(
            input=self._alphas_cumprod[:-1],
            pad=(1, 0),
            value=1.0
        )

        self._sqrt_recip_alphas = torch.sqrt(1.0 / self._alphas)

        self._sqrt_alphas_cumprod = torch.sqrt(self._alphas_cumprod)

        self._sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self._alphas_cumprod)

        self._posterior_variance = self._beta_schedule * (1. - self._alphas_cumprod_prev) / (1. - self._alphas_cumprod)

    def _setup_linear_beta_schedule(self, start: float = 0.0001, end: float = 0.02):

        return torch.linspace(start, end, self._timesteps)

    def q_mean_variance(self, x0, t):

        mean = extract(self._sqrt_alphas_cumprod, t, x0.shape) * x0
        variance = extract(1. - self._alphas_cumprod, t, x0.shape)
        log_variance = torch.log(variance)

        return mean, variance, log_variance

    def _setup_cosine_beta_schedule(self, s: float = 0.0008):
        """
        cosine schedule as proposed in https://arxiv.org/abs/2102.09672
        """

        steps = self._timesteps + 1
        x = torch.linspace(0, self._timesteps, steps)
        alphas_cumprod = torch.cos(((x / self._timesteps) + s) / (1 + s) * (torch.pi * 0.5)) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 1)

    def _generate_random_timesteps(self, x: torch.Tensor):

        t = torch.randint(0, self._timesteps, (x.shape[0],), device=x.device).long()

        return t

    def forward(self, x: torch.Tensor, t: torch.Tensor = None):

        t = t if t is not None else self._generate_random_timesteps(x=x)

        noise = torch.randn_like(x)

        # fetch the data from the pre-calculated vector, grabbing the entries
        # specified by the indicies in t
        sqrt_alphas_cumprod_t = extract(self._sqrt_alphas_cumprod, t, x.shape)

        sqrt_one_minus_alphas_cumprod_t = extract(
            self._sqrt_one_minus_alphas_cumprod, t, x.shape
        )

        xt = sqrt_alphas_cumprod_t * x + sqrt_one_minus_alphas_cumprod_t * noise

        return xt, noise, t

    def loss(self, noise: torch.Tensor, predicted_noise: torch.Tensor):

        loss = torch.nn.functional.mse_loss(noise, predicted_noise)  # + 0.1 * predicted_noise.mean() # torch.nn.functional.smooth_l1_loss(noise, predicted_noise)

        return loss

    def backward(self, x: torch.Tensor, predicted_noise: torch.Tensor, t: torch.Tensor):

        beta = extract(self._beta_schedule, t, x.shape)

        sqrt_one_minus_alphas_cumprod_t = extract(
            self._sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = extract(self._sqrt_recip_alphas, t, x.shape)

        model_mean = sqrt_recip_alphas_t * (
            x - beta * predicted_noise / sqrt_one_minus_alphas_cumprod_t
        )

        posterior_variance_t = extract(self._posterior_variance, t, x.shape)

        noise = torch.randn_like(x) if t > 0 else 0

        output = model_mean + torch.sqrt(posterior_variance_t) * noise

        # output = torch.clip(output, -1, 1)

        return output
