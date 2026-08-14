"""ESPnet E-Branchformer model implementations."""

from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

import numpy as np
import numpy.typing as npt
import onnxruntime as rt

from onnx_asr.asr import Preprocessor, _AsrWithCtcDecoding, _AsrWithDecoding
from onnx_asr.onnx import OnnxSessionOptions
from onnx_asr.utils import is_float32_array, is_int64_array


class _Espnet(_AsrWithDecoding):
    """Common parts of the ESPnet ASR model implementations."""

    @property
    def _preprocessor_name(self) -> str:
        return "w2vbert"

    @property
    def _subsampling_factor(self) -> int:
        return int(self.config.get("subsampling_factor", 8))


class EspnetCtc(_Espnet, _AsrWithCtcDecoding):
    """ESPnet E-Branchformer CTC model implementation."""

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
        return {"model": f"model{suffix}.onnx", "vocab": "vocab.txt", "config": "config.json"}

    def _encode(
        self, features: npt.NDArray[np.float32], features_lens: npt.NDArray[np.int64]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
        logprobs, logprobs_lens = self._model.run(
            ["logprobs", "logprobs_lens"], {"features": features, "features_lens": features_lens.astype(np.int64)}
        )
        assert is_float32_array(logprobs)
        assert is_int64_array(logprobs_lens)
        return logprobs, np.minimum(logprobs_lens, logprobs.shape[1])


class EspnetAED(_Espnet):
    """ESPnet E-Branchformer attention decoder model implementation.

    The encoder graph and the transformer decoder graph are separate files. Decoding is
    greedy and recomputes the decoder over the whole prefix at every step, so the decoder
    graph needs no key-value cache inputs.
    """

    def __init__(  # noqa: D107
        self,
        model_files: dict[str, Path],
        preprocessor_factory: Callable[[str], Preprocessor],
        onnx_options: OnnxSessionOptions,
    ):
        super().__init__(model_files, preprocessor_factory, onnx_options)
        self._encoder = rt.InferenceSession(model_files["encoder"], **onnx_options)
        self._decoder = rt.InferenceSession(model_files["decoder"], **onnx_options)

        tokens = {token: id for id, token in self._vocab.items()}
        self._eos_token_id = tokens["<sos/eos>"]

    @staticmethod
    def _get_model_files(quantization: str | None = None) -> dict[str, str]:
        suffix = "?" + quantization if quantization else ""
        return {
            "encoder": f"encoder{suffix}.onnx",
            "decoder": f"decoder{suffix}.onnx",
            "vocab": "vocab.txt",
            "config": "config.json",
        }

    @property
    def _max_sequence_length(self) -> int:
        return int(self.config.get("max_sequence_length", 200))

    def _encode(
        self, features: npt.NDArray[np.float32], features_lens: npt.NDArray[np.int64]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
        encoder_out, encoder_out_lens = self._encoder.run(
            ["encoder_out", "encoder_out_lens"], {"features": features, "features_lens": features_lens.astype(np.int64)}
        )
        assert is_float32_array(encoder_out)
        assert is_int64_array(encoder_out_lens)
        return encoder_out, np.minimum(encoder_out_lens, encoder_out.shape[1])

    def _decode(
        self,
        tokens: npt.NDArray[np.int64],
        encoder_out: npt.NDArray[np.float32],
        encoder_out_lens: npt.NDArray[np.int64],
    ) -> npt.NDArray[np.float32]:
        (logprobs,) = self._decoder.run(
            ["logprobs"],
            {"tokens": tokens, "encoder_out": encoder_out, "encoder_out_lens": encoder_out_lens},
        )
        assert is_float32_array(logprobs)
        return logprobs

    def _decoding(
        self, encoder_out: npt.NDArray[np.float32], encoder_out_lens: npt.NDArray[np.int64], /, **kwargs: object | None
    ) -> Iterator[tuple[Iterable[int], None, Iterable[float]]]:
        batch_size = encoder_out.shape[0]
        batch_tokens = np.full((batch_size, 1), self._eos_token_id, dtype=np.int64)
        batch_logprobs = np.zeros((batch_size, 0), dtype=np.float32)
        finished = np.zeros(batch_size, dtype=bool)
        max_length = min(self._max_sequence_length, int(max(encoder_out_lens)))

        while batch_tokens.shape[1] <= max_length:
            logprobs = self._decode(batch_tokens, encoder_out, encoder_out_lens)

            next_tokens = np.argmax(logprobs[:, -1], axis=-1)
            next_tokens[finished] = self._eos_token_id
            finished |= next_tokens == self._eos_token_id
            if finished.all():
                break

            next_logprobs = np.take_along_axis(logprobs[:, -1], next_tokens[:, None], axis=-1).squeeze(axis=-1)
            batch_tokens = np.concatenate((batch_tokens, next_tokens[:, None]), axis=-1)
            batch_logprobs = np.concatenate((batch_logprobs, next_logprobs[:, None]), axis=-1)

        for tokens, logprobs in zip(batch_tokens[:, 1:], batch_logprobs, strict=True):
            mask = tokens != self._eos_token_id
            yield tokens[mask], None, logprobs[mask]
