"""
# L1 + Multi-resolution spectrogram Loss

Mixed time-domain L1 + STFT magnitude loss with multiple window sizes.

Adapted from ESPnet2:
https://github.com/espnet/espnet/blob/master/espnet2/enh/loss/criterions/time_domain.py

License Apache 2.0
https://github.com/espnet/espnet?tab=Apache-2.0-1-ov-file#readme

2024: Modified by Robin Scheibler as follows
- Simplified torch version compatibility code
- Replaced ESPnet STFT by torchaudio STFT
"""
import torch
import torchaudio


class MultiResL1SpecLoss(torch.nn.Module):
    """Multi-Resolution L1 time-domain + STFT mag loss

    Reference:
    Lu, Y. J., Cornell, S., Chang, X., Zhang, W., Li, C., Ni, Z., ... & Watanabe, S.
    Towards Low-Distortion Multi-Channel Speech Enhancement:
    The ESPNET-Se Submission to the L3DAS22 Challenge. ICASSP 2022 p. 9201-9205.

    Parameters
    ----------
    window_sz: (list)
        list of STFT window sizes.
    hop_sz: (list, optional)
        list of hop_sizes, default is each window_sz // 2.
    eps: (float)
        stability epsilon
    time_domain_weight: (float)
        weight for time domain loss.
    """

    def __init__(
        self,
        window_sz=[512],
        hop_sz=None,
        eps=1e-8,
        time_domain_weight=0.5,
        scale_invariant=False,
    ):
        super().__init__()

        assert all([x % 2 == 0 for x in window_sz])
        self.window_sz = window_sz

        if hop_sz is None:
            self.hop_sz = [x // 2 for x in window_sz]
        else:
            self.hop_sz = hop_sz

        self.time_domain_weight = time_domain_weight
        self.eps = eps
        self.scale_invariant = scale_invariant
        self.stft_encoders = torch.nn.ModuleList([])

        for w, h in zip(self.window_sz, self.hop_sz):
            # with power=1, magnitude is provided directly
            stft_enc = torchaudio.transforms.Spectrogram(
                n_fft=w,
                win_length=w,
                hop_length=h,
                window_fn=torch.hann_window,
                center=True,
                normalized=False,
                onesided=True,
                pad_mode="constant",
                power=None,
            )
            self.stft_encoders.append(stft_enc)

    def forward(
        self,
        target: torch.Tensor,
        estimate: torch.Tensor,
    ):
        """forward.

        Args:
            target: (Batch, T)
            estimate: (Batch, T)
        Returns:
            loss: (Batch,)
        """
        assert target.shape == estimate.shape, (target.shape, estimate.shape)
        half_precision = (torch.float16, torch.bfloat16)
        if target.dtype in half_precision or estimate.dtype in half_precision:
            target = target.float()
            estimate = estimate.float()
        # shape bsz, samples
        if self.scale_invariant:
            scaling_factor = torch.sum(estimate * target, -1, keepdim=True) / (
                torch.sum(estimate**2, -1, keepdim=True) + self.eps
            )
        else:
            scaling_factor = 1.0

        dims = tuple(range(1, len(target.shape)))
        time_domain_loss = torch.mean(
            (estimate * scaling_factor - target).abs(), dim=dims
        )

        if len(self.stft_encoders) == 0:
            return time_domain_loss.mean()
        else:
            spectral_loss = torch.zeros_like(time_domain_loss)
            for stft_enc in self.stft_encoders:
                target_spec = stft_enc(target)
                estimate_spec = stft_enc(estimate * scaling_factor)

                dims = tuple(range(1, len(target_spec.shape)))  # all dims but batch

                # magnitutde
                target_mag = abs(target_spec)
                estimate_mag = abs(estimate_spec)
                c_loss = [torch.mean((estimate_mag - target_mag).abs(), dim=dims)]

                spectral_loss += torch.stack(c_loss, dim=0).mean(dim=0)

            total_loss = time_domain_loss * self.time_domain_weight + (
                1 - self.time_domain_weight
            ) * spectral_loss / len(self.stft_encoders)
            return total_loss.mean()
