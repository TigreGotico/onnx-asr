"""Base Speaker Embedding classes."""

from typing import Literal, Protocol

import numpy as np
import numpy.typing as npt


class SpeakerEmbedding(Protocol):
    """Speaker Embedding protocol."""

    _supports_fetcher: bool = False
    """The constructor takes a `fetcher` keyword for on-demand file download."""

    @staticmethod
    def _get_sample_rate() -> Literal[8_000, 16_000]:
        return 16_000

    def embedding(
        self, waveforms: npt.NDArray[np.float32], waveforms_len: npt.NDArray[np.int64]
    ) -> npt.NDArray[np.float32]:
        """Compute speaker embedding."""
        ...
