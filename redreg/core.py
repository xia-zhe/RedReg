import math
from collections import deque
import torch
from torch import nn
from torch.nn import functional as F
from models.backbone import resnet18
from models.basic_model import AVClassifier
from utils.utils import weight_init


def gaussian_rate(mu, logvar):
    """Eq. 13: sum over latent dimensions, never divide by latent dimension."""
    return 0.5 * (mu.square() + torch.expm1(logvar) - logvar).sum(dim=-1)


def gated_compression_loss(raw_rates, gate, beta, latent_dim, reduction='sum'):
    """Keep raw KL for monitoring; explicitly choose compression-loss units.

    ``sum`` is literal manuscript Eq. 24. ``mean_dimension`` is an engineering
    scale calibration, equivalent to beta/latent_dim on the raw KL objective.
    It is intentionally explicit rather than silently redefining Gaussian KL.
    """
    if latent_dim <= 0 or reduction not in ['sum', 'mean_dimension']:
        raise ValueError('Invalid compression reduction or latent dimension')
    mask = torch.as_tensor(gate, dtype=raw_rates.dtype, device=raw_rates.device).detach()
    if raw_rates.ndim != 1 or mask.shape != raw_rates.shape:
        raise ValueError('Compression expects globally batch-averaged rates per modality')
    divisor = latent_dim if reduction == 'mean_dimension' else 1
    return beta * (mask * raw_rates).sum() / divisor


def align_global_readouts(out, module):
    """The official baseline computes unimodal readouts AFTER DP gathering.

    Returning the two feature tensors separately keeps the gather/backward
    layout compatible with its AVClassifier, rather than slicing a stacked z.
    """
    if getattr(module, 'readout_operator', 'fused_linear') == 'baseline_mm':
        a, v, d = out['audio_features'], out['video_features'], module.latent_dim
        out['video_logits'] = torch.mm(v, module.head.weight[:, d:].t()) + module.head.bias / 2
        out['audio_logits'] = torch.mm(a, module.head.weight[:, :d].t()) + module.head.bias / 2
        if hasattr(module, 'task_heads'):
            out['audio_task_logits'] = module.task_heads[0](a)
            out['video_task_logits'] = module.task_heads[1](v)
    return out


class StochasticBottleneck(nn.Module):
    def __init__(self, latent_dim=512, initial_logvar=-6.0, initial_mean_scale=1.0):
        super().__init__()
        self.mu_head = nn.Linear(512, latent_dim)
        self.logvar_head = nn.Linear(512, latent_dim)
        if latent_dim == 512:
            nn.init.eye_(self.mu_head.weight)
        else:
            nn.init.xavier_normal_(self.mu_head.weight)
        if not math.isfinite(initial_mean_scale) or initial_mean_scale < 0:
            raise ValueError('Initial mean scale must be finite and nonnegative')
        with torch.no_grad():
            self.mu_head.weight.mul_(initial_mean_scale)
        nn.init.zeros_(self.mu_head.bias)
        nn.init.zeros_(self.logvar_head.weight)
        nn.init.constant_(self.logvar_head.bias, initial_logvar)

    def forward(self, h):
        mu = self.mu_head(h)
        logvar = self.logvar_head(h).clamp(-12.0, 8.0)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu) if self.training else mu
        return z, mu, gaussian_rate(mu, logvar)


class RedReg(nn.Module):
    def __init__(self, latent_dim=512, initial_logvar=-6.0, initial_mean_scale=1.0, bottleneck_type='gaussian', auxiliary_heads='shared', readout_operator='fused_linear'):
        super().__init__()
        self.latent_dim = latent_dim
        self.bottleneck_type = bottleneck_type
        self.readout_operator = readout_operator
        if readout_operator not in ['fused_linear', 'baseline_mm']:
            raise ValueError('Unknown readout operator')
        if bottleneck_type not in ['gaussian', 'none'] or (bottleneck_type == 'none' and latent_dim != 512):
            raise ValueError('Plain control requires dimension 512; unknown bottleneck type is invalid')
        self.audio_net = resnet18(modality='audio')
        self.visual_net = resnet18(modality='visual')
        self.bottlenecks = nn.ModuleList([
            StochasticBottleneck(latent_dim, initial_logvar, initial_mean_scale) for _ in range(2)])
        self.head = nn.Linear(2 * latent_dim, 6)
        nn.init.xavier_normal_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        if auxiliary_heads not in ['shared', 'independent']:
            raise ValueError('Unknown auxiliary head policy')
        if auxiliary_heads == 'independent':
            self.task_heads = nn.ModuleList([nn.Linear(latent_dim, 6) for _ in range(2)])

    def forward(self, audio, visual):
        a = F.adaptive_avg_pool2d(self.audio_net(audio), 1).flatten(1)
        v = self.visual_net(visual)
        _, c, h, w = v.shape
        v = v.view(a.shape[0], -1, c, h, w).permute(0, 2, 1, 3, 4)
        v = F.adaptive_avg_pool3d(v, 1).flatten(1)
        if self.bottleneck_type == 'gaussian':
            za, mua, ra = self.bottlenecks[0](a)
            zv, muv, rv = self.bottlenecks[1](v)
        else:
            za = mua = a
            zv = muv = v
            ra, rv = a.new_zeros(a.shape[0]), v.new_zeros(v.shape[0])
        logits = self.head(torch.cat((za, zv), dim=1))
        if self.readout_operator == 'baseline_mm':
            video_logits = torch.mm(zv, self.head.weight[:, self.latent_dim:].t()) + self.head.bias / 2
            audio_logits = torch.mm(za, self.head.weight[:, :self.latent_dim].t()) + self.head.bias / 2
        else:
            audio_logits = F.linear(za, self.head.weight[:, :self.latent_dim], self.head.bias / 2)
            video_logits = F.linear(zv, self.head.weight[:, self.latent_dim:], self.head.bias / 2)
        # Per-sample rates are gathered across GPUs before their global batch mean.
        result = {'logits': logits, 'audio_logits': audio_logits, 'video_logits': video_logits,
                'z': torch.stack((za, zv), dim=1), 'mu': torch.stack((mua, muv), dim=1),
                'rates': torch.stack((ra, rv), dim=1)}
        if hasattr(self, 'task_heads'):
            result['audio_task_logits'] = self.task_heads[0](za)
            result['video_task_logits'] = self.task_heads[1](zv)
        if self.readout_operator == 'baseline_mm':
            result['audio_features'], result['video_features'] = za, zv
        return result


class ProbeBank(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.heads = nn.ModuleList([nn.Linear(latent_dim, 6) for _ in range(2)])

    def forward(self, representations):
        # Detachment is mandatory even when callers forget it.
        return torch.stack([head(representations[:, m].detach())
                            for m, head in enumerate(self.heads)], dim=1)


def build_from_baseline_initialization(args):
    """Fresh initialization, not the trained InfoReg checkpoint.

    Reuse identical initial encoders/classifier and restore the post-baseline-init
    CPU RNG state so worker augmentation seeds are not shifted by new head creation.
    """
    baseline = AVClassifier(args)
    baseline.apply(weight_init)
    rng = torch.get_rng_state()
    model = RedReg(args.latent_dim, args.initial_logvar,
                       getattr(args, 'initial_mean_scale', 1.0), getattr(args, 'bottleneck_type', 'gaussian'),
                       getattr(args, 'auxiliary_heads', 'shared'), getattr(args, 'readout_operator', 'fused_linear'))
    model.audio_net.load_state_dict(baseline.audio_net.state_dict())
    model.visual_net.load_state_dict(baseline.visual_net.state_dict())
    if args.latent_dim == 512:
        model.head.load_state_dict(baseline.head.state_dict())
    if hasattr(model, 'task_heads'):
        with torch.no_grad():
            for m, head in enumerate(model.task_heads):
                head.weight.copy_(model.head.weight[:, m * args.latent_dim:(m + 1) * args.latent_dim])
                head.bias.copy_(model.head.bias / 2)
    probes = ProbeBank(args.latent_dim)
    with torch.no_grad():
        for m, head in enumerate(probes.heads):
            head.weight.copy_(model.head.weight[:, m * args.latent_dim:(m + 1) * args.latent_dim])
            head.bias.copy_(model.head.bias / 2)
    torch.set_rng_state(rng)
    return model, probes


def prediction_consistency(probe_logits):
    """Eqs. 19-20 for audio/video, natural-log JSD normalized by log(2)."""
    with torch.no_grad():
        logp = F.log_softmax(probe_logits.detach(), dim=-1)
        p = logp.exp()
        logmixture = ((p[:, 0] + p[:, 1]) / 2).clamp_min(1e-12).log()
        js = 0.5 * ((p[:, 0] * (logp[:, 0] - logmixture)).sum(-1)
                    + (p[:, 1] * (logp[:, 1] - logmixture)).sum(-1)).mean()
        return float((1.0 - js / math.log(2.0)).clamp(0.0, 1.0))


class RedundancyMonitor:
    def __init__(self, window=10, gamma=0.5, threshold=0.90, agreement=0.5, epsilon=1e-8):
        self.window, self.gamma = window, gamma
        self.threshold, self.agreement, self.epsilon = threshold, agreement, epsilon
        self.steps = 0
        self.losses = [deque(maxlen=window) for _ in range(2)]
        self.rates = [deque(maxlen=window) for _ in range(2)]
        self.gains = [deque(maxlen=window) for _ in range(2)]
        self.previous_smooth = [None, None]

    def observe(self, branch_losses, rates, similarity):
        smooth, gains, snorm, rnorm, redundancy = [], [], [], [], []
        for m in range(2):
            loss, rate = float(branch_losses[m]), float(rates[m])
            if not math.isfinite(loss) or not math.isfinite(rate):
                raise FloatingPointError('Nonfinite RedReg monitor input')
            self.losses[m].append(loss)
            mean_loss = sum(self.losses[m]) / len(self.losses[m])
            previous = self.previous_smooth[m]
            gain = 0.0 if previous is None else (previous - mean_loss) / (abs(previous) + self.epsilon)
            self.previous_smooth[m] = mean_loss
            self.gains[m].append(gain)
            self.rates[m].append(rate)
            norm_gain = gain / (sum(abs(x) for x in self.gains[m]) / len(self.gains[m]) + self.epsilon)
            norm_rate = rate / (sum(abs(x) for x in self.rates[m]) / len(self.rates[m]) + self.epsilon)
            smooth.append(mean_loss); gains.append(gain)
            snorm.append(norm_gain); rnorm.append(norm_rate)
            redundancy.append(norm_rate - self.gamma * max(norm_gain, 0.0))
        dominant = min(range(2), key=lambda m: smooth[m])
        ready = self.steps >= self.window  # Algorithm 1: t < L gets no compression.
        gate = [float(ready and m == dominant and redundancy[m] > self.threshold
                      and similarity >= self.agreement) for m in range(2)]
        self.steps += 1
        return {'smooth_loss': smooth, 'task_gain': gains, 'normalized_gain': snorm,
                'normalized_rate': rnorm, 'redundancy': redundancy, 'dominant': dominant,
                'similarity': similarity, 'gate': gate, 'ready': ready}

    def state_dict(self):
        return {'steps': self.steps, 'losses': [list(x) for x in self.losses],
                'rates': [list(x) for x in self.rates], 'gains': [list(x) for x in self.gains],
                'previous_smooth': list(self.previous_smooth)}

    def load_state_dict(self, state):
        self.steps = state['steps']
        self.previous_smooth = list(state['previous_smooth'])
        for key in ['losses', 'rates', 'gains']:
            setattr(self, key, [deque(x, maxlen=self.window) for x in state[key]])
