"""Real inverse FFT via the standard complex inverse DFT."""

import torch
from torch import Tensor, nn


class WebIRFFT(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.even_size = size % 2 == 0
        self.register_buffer("mirror", torch.arange((size - 1) // 2, 0, -1), persistent=False)

    def forward(self, spectrum: Tensor) -> Tensor:
        real = spectrum[:, 0]
        imag = spectrum[:, 1]
        imag = torch.cat((torch.zeros_like(imag[:1]), imag[1:]))
        if self.even_size:
            imag = torch.cat((imag[:-1], torch.zeros_like(imag[-1:])))
        # Reconstruct negative frequencies for runtimes that require a full inverse DFT.
        full_spectrum = torch.stack(
            (
                torch.cat((real, real.index_select(0, self.mirror))),
                torch.cat((imag, -imag.index_select(0, self.mirror))),
            ),
            dim=-1,
        )
        signal = torch.fft.ifft(torch.view_as_complex(full_spectrum))
        return torch.view_as_real(signal)[:, 0]
