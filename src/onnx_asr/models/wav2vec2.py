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
        suffix = "?" + quantization if quantization else ""
        return {"model": f"model{suffix}.onnx", "vocab": "vocab.txt"}

    @property
    def _preprocessor_name(self) -> str:
        return "identity"

    @property
    def _subsampling_factor(self) -> int:
        return int(self.config.get("subsampling_factor", 320))

    def _encode(
        self, waveforms: npt.NDArray[np.float32], waveforms_len: npt.NDArray[np.int64]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
        (logprobs,) = self._model.run(
            ["logprobs"],
            {"input_values": waveforms, "input_lengths": waveforms_len.astype(np.int64)},
        )
        assert is_float32_array(logprobs)
        out_lens = waveforms_len // self._subsampling_factor + 1
        out_lens = np.minimum(out_lens, logprobs.shape[1]).astype(np.int64)
        assert is_int64_array(out_lens)
        return logprobs, out_lens


class Wav2Vec2CtcLogits(_AsrWithCtcDecoding):
    """Wav2Vec2 CTC model exported with raw logits and a sherpa-onnx tokens file.

    This is the export convention used by the fairseq2 and sherpa-onnx tool chains:
    one input `x` with the raw waveform, one output `logits` without log_softmax,
    and a `tokens.txt` vocabulary in which the CTC blank is `<pad>`. The class
    applies log_softmax and, unless the config turns it off, the per-utterance
    normalisation that these models expect.
    """

    def __init__(  # noqa: D107
        self,
        model_files: dict[str, Path],
        preprocessor_factory: Callable[[str], Preprocessor],
        onnx_options: OnnxSessionOptions,
    ):
        super().__init__(model_files, preprocessor_factory, onnx_options)
        self._model = rt.InferenceSession(model_files["model"], **onnx_options)

        blank_token = str(self.config.get("blank_token", "<pad>"))
        self._blank_idx = next((id for id, token in self._vocab.items() if token == blank_token), 0)

    @staticmethod
    def _get_model_files(quantization: str | None = None) -> dict[str, str]:
        suffix = "?" + quantization if quantization else ""
        return {"model": f"**/model{suffix}.onnx", "vocab": "**/tokens.txt"}

    @property
    def _preprocessor_name(self) -> str:
        return "identity"

    @property
    def _subsampling_factor(self) -> int:
        return int(self.config.get("subsampling_factor", 320))

    @property
    def _normalize_audio(self) -> bool:
        return bool(self.config.get("normalize_audio", True))

    def _encode(
        self, waveforms: npt.NDArray[np.float32], waveforms_len: npt.NDArray[np.int64]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
        if self._normalize_audio:
            mask = np.arange(waveforms.shape[1])[None, :] < waveforms_len[:, None]
            count = np.maximum(waveforms_len, 1).astype(np.float32)[:, None]
            mean = (waveforms * mask).sum(axis=1, keepdims=True) / count
            var = (((waveforms - mean) * mask) ** 2).sum(axis=1, keepdims=True) / count
            waveforms = np.where(mask, (waveforms - mean) / np.sqrt(var + 1e-7), waveforms).astype(np.float32)

        (logits,) = self._model.run(["logits"], {"x": waveforms})
        assert is_float32_array(logits)
        shifted = logits - logits.max(axis=-1, keepdims=True)
        logprobs = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
        logprobs = logprobs.astype(np.float32)
        assert is_float32_array(logprobs)

        out_lens = waveforms_len // self._subsampling_factor + 1
        out_lens = np.minimum(out_lens, logprobs.shape[1]).astype(np.int64)
        assert is_int64_array(out_lens)
        return logprobs, out_lens
