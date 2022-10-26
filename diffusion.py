from signal import raise_signal
import time
import torch

def extract(a, t, x_shape):

    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())

    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

class Diffusion(object):

    def __init__(self, timesteps: int, schedular: str = "linear") -> None:

        self._timesteps = timesteps

        self._beta_schedule = None

        if schedular == "linear":

            self._beta_schedule = self._setup_linear_beta_schedule()

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

    def _setup_linear_beta_schedule(self, start: float = 0.0001, end: float = 0.03):

        return torch.linspace(start, end, self._timesteps)

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

        return sqrt_alphas_cumprod_t * x + sqrt_one_minus_alphas_cumprod_t * noise, noise, t

    def loss(self, noise: torch.Tensor, predicted_noise: torch.Tensor):

        loss = torch.nn.functional.mse_loss(noise, predicted_noise) # torch.nn.functional.smooth_l1_loss(noise, predicted_noise)

        return loss

    def backward(self, x: torch.Tensor, predicted_noise: torch.Tensor, t: torch.Tensor):

        betas = extract(self._beta_schedule, t, x.shape)

        sqrt_one_minus_alphas_cumprod_t = extract(
            self._sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = extract(self._sqrt_recip_alphas, t, x.shape)

        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        model_mean = sqrt_recip_alphas_t * (
            x - betas * predicted_noise / sqrt_one_minus_alphas_cumprod_t
        )

        posterior_variance_t = extract(self._posterior_variance, t, x.shape)

        noise = torch.randn_like(x) if t > 0 else 0

        return model_mean + torch.sqrt(posterior_variance_t) * noise
