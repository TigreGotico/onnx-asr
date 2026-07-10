"""Wav2Vec2 CTC model implementation."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import numpy.typing as npt
import onnxruntime as rt

from onnx_asr.asr import Preprocessor, _AsrWithCtcDecoding
from onnx_asr.onnx import OnnxSessionOptions
from onnx_asr.utils import is_float32_array, is_int64_array


class Wav2Vec2Ctc(_AsrWithCtcDecoding):
    """Wav2Vec2 CTC model implementation (HuggingFace wav2vec2 / XLS-R fine-tunes)."""

    def __init__(  # noqa: D107
        self,
        model_files: dict[str, Path],
        preprocessor_factory: Callable[[str], Preprocessor],
        onnx_options: OnnxSessionOptions,
    ):
        super().__init__(model_files, preprocessor_factory, onnx_options)
        self._model = rt.InferenceSession(model_files["model"], **onnx_options)

    @staticmethod
    def _get_model_files(quantization: str | None = None) -> dict[str, str]:
        """
        Builds the model and vocabulary filenames for the selected quantization format.
        
        Parameters:
            quantization (str | None): Optional quantization identifier appended to the model filename.
        
        Returns:
            dict[str, str]: Mapping containing the model and vocabulary filenames.
        """
        suffix = "?" + quantization if quantization else ""
        return {"model": f"model{suffix}.onnx", "vocab": "vocab.txt"}

    @property
    def _preprocessor_name(self) -> str:
        """Return the name of the preprocessor used by the model."""
        return "identity"

    @property
    def _subsampling_factor(self) -> int:
        """
        Provide the number of input samples represented by each output timestep.
        
        Returns:
        	int: The configured subsampling factor, or 320 when unspecified.
        """
        return int(self.config.get("subsampling_factor", 320))

    def _encode(
        self, waveforms: npt.NDArray[np.float32], waveforms_len: npt.NDArray[np.int64]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
        """
        Encode audio waveforms into CTC log-probabilities and output sequence lengths.
        
        Parameters:
            waveforms: Input audio waveforms.
            waveforms_len: Length of each input waveform.
        
        Returns:
            A tuple containing the CTC log-probabilities and corresponding output sequence lengths.
        """
        (logprobs,) = self._model.run(
            ["logprobs"],
            {"input_values": waveforms, "input_lengths": waveforms_len.astype(np.int64)},
        )
        assert is_float32_array(logprobs)
        out_lens = waveforms_len // self._subsampling_factor + 1
        out_lens = np.minimum(out_lens, logprobs.shape[1]).astype(np.int64)
        assert is_int64_array(out_lens)
        return logprobs, out_lens
