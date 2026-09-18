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


class Wav2Vec2BertCtc(_AsrWithCtcDecoding):
    """w2v-BERT 2.0 CTC model (HuggingFace ``Wav2Vec2BertForCTC`` exports).

    The graph takes ``input_features`` (fbank frames stacked in pairs, 160 values
    per step) and an ``attention_mask``, and returns ``logits``. The features
    come from the ``seamless`` preprocessor, which reproduces the HuggingFace
    ``SeamlessM4TFeatureExtractor``; an int8 graph decodes the same text as the
    PyTorch pipeline only when the two are identical.
    """

    def __init__(  # noqa: D107
        self,
        model_files: dict[str, Path],
        preprocessor_factory: Callable[[str], Preprocessor],
        onnx_options: OnnxSessionOptions,
    ):
        super().__init__(model_files, preprocessor_factory, onnx_options)
        self._model = rt.InferenceSession(model_files["model"], **onnx_options)
        mask_type = next(i.type for i in self._model.get_inputs() if i.name == "attention_mask")
        self._mask_dtype = np.int32 if mask_type == "tensor(int32)" else np.int64

    @staticmethod
    def _get_model_files(quantization: str | None = None) -> dict[str, str]:
        suffix = "?" + quantization if quantization else ""
        return {"model": f"model{suffix}.onnx", "vocab": "vocab.txt"}

    @property
    def _preprocessor_name(self) -> str:
        return "seamless"

    @property
    def _subsampling_factor(self) -> int:
        # 10 ms fbank frames, stacked in pairs, then the adapter's stride of 2
        return int(self.config.get("subsampling_factor", 640))

    def _encode(
        self, features: npt.NDArray[np.float32], features_lens: npt.NDArray[np.int64]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
        mask = (np.arange(features.shape[1])[None, :] < features_lens[:, None]).astype(self._mask_dtype)
        (logits,) = self._model.run(["logits"], {"input_features": features, "attention_mask": mask})
        assert is_float32_array(logits)
        logprobs = logits - np.logaddexp.reduce(logits, axis=-1, keepdims=True)
        out_lens = np.minimum((features_lens - 1) // 2 + 1, logprobs.shape[1]).astype(np.int64)
        assert is_int64_array(out_lens)
        return logprobs.astype(np.float32), out_lens
