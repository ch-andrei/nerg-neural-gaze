import math

import torch
import torch.nn.functional as F


def logsinh_stable(kappa: torch.Tensor) -> torch.Tensor:
    """
    log(sinh kappa) with stable handling at small and large kappa (kappa >= 0).
    """
    # thresholds by dtype
    log2 = kappa.new_tensor(math.log(2.0))

    # large-κ branch: log(sinh κ) = κ - log 2 + log1p(-exp(-2κ))
    large = kappa >= 1e-4
    out_large = kappa - log2 + torch.log1p(-torch.exp(-2.0 * kappa))

    # small-κ series: sinh κ ≈ κ + κ^3/6  ⇒ log(sinh κ) ≈ log κ + κ^2/6
    k = kappa.clamp_min(1e-9)
    out_small = torch.log(k) + (kappa * kappa) / 6.0

    return torch.where(large, out_large, out_small)


def logC_vmf(kappa):
    # log C(kappa) on S^2: log(kappa) - log(4π) - log(sinh kappa)
    log4pi = torch.log(torch.tensor(4.0 * torch.pi, dtype=kappa.dtype, device=kappa.device))
    small = (kappa < 1e-4)
    # series near 0: -log(4π) - kappa^2/6
    approx_small = -log4pi - (kappa * kappa) / 6.0
    exact = torch.log(kappa.clamp_min(1e-9)) - log4pi - logsinh_stable(kappa)
    return torch.where(small, approx_small, exact)


def vmf_log_density_mixture(d, mu, kappa, mix_logits):
    """
    d: [N,3] unit directions
    mu: [N,K,3] unit means
    kappa: [N,K] concentrations >= 0
    mix_logits: [N,K] unnormalized log-weights
    returns: log density per direction, shape [N]
    """
    # ensure shapes [N,K,3], [N,K], [N,K]
    # N = d.shape[0]
    # K = mu.shape[-2]
    d_exp = d[:, None, :]                          # [N,1,3]
    # mu = mu.expand(N, K, 3).contiguous()
    # kappa = kappa.expand(N, K).contiguous()
    # mix_logits = mix_logits.expand(N, K).contiguous()
    dot = (d_exp * mu).sum(dim=-1)                 # [N,K]
    comp_log = logC_vmf(kappa) + kappa * dot       # [N,K]
    log_pi = F.log_softmax(mix_logits, dim=-1)     # [N,K]
    return torch.logsumexp(log_pi + comp_log, dim=-1, keepdim=True)  # [N]


class VMFHead(torch.nn.Module):
    def __init__(
            self,
            dim_in,
            dim_hidden=32,
            num_mu=16,
            share_kappa=False
    ):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim_in, dim_hidden),
            torch.nn.ReLU(),
        )
        self.mu_raw = torch.nn.Linear(dim_hidden, num_mu * 3)
        self.kappa_raw = torch.nn.Linear(dim_hidden, 1 if share_kappa else num_mu)
        self.mix_logits = torch.nn.Linear(dim_hidden, num_mu)
        self.num_mu = num_mu

    @torch.amp.autocast('cuda', dtype=torch.float32)
    def forward(self, h):
        h = self.mlp(h)
        mu = self.mu_raw(h).view(-1, self.num_mu, 3)
        mu = mu / (mu.norm(dim=-1, keepdim=True) + 1e-8)  # unit
        kappa = F.softplus(self.kappa_raw(h)) + 1e-6       # >=0
        mix_logits = self.mix_logits(h)
        return mu, kappa, mix_logits
