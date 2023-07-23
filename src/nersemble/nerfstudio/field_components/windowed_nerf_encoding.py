from typing import Optional

import torch
from jaxtyping import Shaped
from nerfstudio.field_components.encodings import NeRFEncoding
from nerfstudio.utils.math import expected_sin
from torch import Tensor


class WindowedNeRFEncoding(NeRFEncoding):
    """Multi-scale sinousoidal encodings. Support ``integrated positional encodings`` if covariances are provided.
    Each axis is encoded with frequencies ranging from 2^min_freq_exp to 2^max_freq_exp.

    Args:
        in_dim: Input dimension of tensor
        num_frequencies: Number of encoded frequencies per axis
        min_freq_exp: Minimum frequency exponent
        max_freq_exp: Maximum frequency exponent
        include_input: Append the input coordinate to the encoding
    """

    def __init__(
            self, in_dim: int, num_frequencies: int, min_freq_exp: float, max_freq_exp: float,
            include_input: bool = False
    ) -> None:
        super().__init__(in_dim, num_frequencies, min_freq_exp, max_freq_exp, include_input)

        self.num_frequencies = num_frequencies
        self.min_freq = min_freq_exp
        self.max_freq = max_freq_exp
        self.include_input = include_input

    def forward(
            self,
            in_tensor: Shaped[Tensor, "B D"],
            covs: Optional[Tensor] = None,
            windows_param: Optional[float] = None,
    ) -> Tensor:
        """Calculates NeRF encoding. If covariances are provided the encodings will be integrated as proposed
            in mip-NeRF.

        Args:
            in_tensor: For best performance, the input tensor should be between 0 and 1.
            covs: Covariances of input points.
        Returns:
            Output values will be between -1 and 1
        """
        in_tensor = 2 * torch.pi * in_tensor  # scale to [0, 2pi]
        freqs = 2 ** torch.linspace(self.min_freq, self.max_freq, self.num_frequencies).to(in_tensor.device)
        scaled_inputs = in_tensor[..., None] * freqs  # [..., "input_dim", "num_scales"]
        scaled_inputs = scaled_inputs.view(*scaled_inputs.shape[:-2], -1)  # [..., "input_dim" * "num_scales"]

        if covs is None:
            encoded_inputs = torch.sin(torch.cat([scaled_inputs, scaled_inputs + torch.pi / 2.0], dim=-1))
        else:
            input_var = torch.diagonal(covs, dim1=-2, dim2=-1)[..., :, None] * freqs[None, :] ** 2
            input_var = input_var.reshape((*input_var.shape[:-2], -1))
            encoded_inputs = expected_sin(
                torch.cat([scaled_inputs, scaled_inputs + torch.pi / 2.0], dim=-1), torch.cat(2 * [input_var], dim=-1)
            )

        if windows_param is not None:
            window = (
                self.posenc_window(windows_param)
                    .to(in_tensor.device)[None, :]
                    .repeat(in_tensor.shape[-1], 1)
                    .reshape(-1)
                    .repeat(2)
            )
            encoded_inputs = window * encoded_inputs

        if self.include_input:
            encoded_inputs = torch.cat([encoded_inputs, in_tensor], dim=-1)
        return encoded_inputs

    def posenc_window(self, windows_param):
        """Windows a the encoding using a cosiney window.

        This is equivalent to taking a truncated Hann window and sliding it to the
        right along the frequency spectrum.

        Args:
            min_deg: the lower frequency band.
            max_deg: the upper frequency band.
            windows_param: will ease in each frequency as windows_param goes from 0.0 to num_freqs.

        Returns:
            A 1-d numpy array with num_sample elements containing the window.
        """
        bands = torch.linspace(self.min_freq, self.max_freq, self.num_frequencies)
        x = torch.clamp(windows_param - bands, 0, 1)
        return 0.5 * (1 - torch.cos(torch.pi * x))
