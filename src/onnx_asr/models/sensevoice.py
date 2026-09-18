"""SenseVoice model implementation (FunASR non-autoregressive CTC with rich tokens).

The family covers `FunAudioLLM/SenseVoiceSmall`: a SANM encoder with a CTC head that
runs in one forward pass. Four prompt frames go in front of the speech frames, and the
CTC output over those frames carries the detected language, the emotion, the audio
event and the inverse-text-normalization mode. The rest of the CTC output is the
transcript, so decoding is plain CTC greedy collapse with no decoder loop.

`model.onnx`
    Kaldi fbank features + lengths + language id + textnorm id -> CTC log-probs and
    their lengths.

The FunASR frontend does more than the fbank: it stacks 7 frames with a hop of 6 (LFR)
and applies the `am.mvn` mean-variance statistics. Both steps are folded into the
graph, so the only external feature step is the plain kaldi fbank that onnx-asr already
computes for `wespeaker`.

The first four decoded tokens are rich tokens such as `<|en|>`, `<|NEUTRAL|>`,
`<|Speech|>` and `<|woitn|>`. They stay in `TimestampedResult.tokens` but are kept out
of `TimestampedResult.text`.
"""

import re
import typing
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import numpy.typing as npt
import onnxruntime as rt

from onnx_asr.asr import Preprocessor, TimestampedResult, _AsrWithCtcDecoding
from onnx_asr.onnx import OnnxSessionOptions
from onnx_asr.utils import is_float32_array, is_int64_array

RICH_TOKEN_PATTERN = re.compile(r"<\|[^|]*\|>")


class UnknownPromptValueError(ValueError):
    """Unknown SenseVoice prompt value error."""

    def __init__(self, kind: str, name: str, known: Iterable[str]) -> None:
        """Create error."""
        super().__init__(f"Unknown {kind} '{name}', expected one of {sorted(known)}.")


class SenseVoice(_AsrWithCtcDecoding):
    """SenseVoice model implementation."""

    def __init__(  # noqa: D107
        self,
        model_files: dict[str, Path],
        preprocessor_factory: Callable[[str], Preprocessor],
        onnx_options: OnnxSessionOptions,
    ):
        super().__init__(model_files, preprocessor_factory, onnx_options)
        config = typing.cast(dict[str, typing.Any], self.config)

        self._model = rt.InferenceSession(model_files["model"], **onnx_options)
        self._languages: dict[str, int] = config.get("languages", {"auto": 0})
        self._textnorm: dict[str, int] = config.get("textnorm", {"withitn": 14, "woitn": 15})
        self._default_language = str(config.get("default_language", "auto"))
        self._default_textnorm = str(config.get("default_textnorm", "woitn"))
        # The FunASR frontend runs the fbank on an int16 scaled waveform. The scale
        # matters because the log floor and the baked-in CMVN both depend on it.
        self._waveform_scale = float(config.get("waveform_scale", 1 << 15))

    def _features(
        self, waveforms: npt.NDArray[np.float32], waveforms_len: npt.NDArray[np.int64]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
        return self._preprocessor((waveforms * self._waveform_scale).astype(np.float32), waveforms_len)

    @staticmethod
    def _get_model_files(quantization: str | None = None) -> dict[str, str]:
        suffix = "?" + quantization if quantization else ""
        return {"model": f"model{suffix}.onnx", "vocab": "vocab.txt", "config": "config.json"}

    @property
    def _preprocessor_name(self) -> str:
        config = typing.cast(dict[str, typing.Any], self.config)
        return str(config.get("preprocessor", "wespeaker"))

    @property
    def _subsampling_factor(self) -> int:
        return int(self.config.get("subsampling_factor", 6))

    def _lookup(self, table: dict[str, int], name: str, kind: str) -> int:
        if name not in table:
            raise UnknownPromptValueError(kind, name, table)
        return table[name]

    def _encode_with_prompt(
        self,
        features: npt.NDArray[np.float32],
        features_lens: npt.NDArray[np.int64],
        language: int,
        textnorm: int,
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
        batch_size = features.shape[0]
        logprobs, logprobs_lens = self._model.run(
            ["logprobs", "logprobs_lens"],
            {
                "features": features,
                "features_lens": features_lens.astype(np.int64),
                "language": np.full(batch_size, language, dtype=np.int64),
                "textnorm": np.full(batch_size, textnorm, dtype=np.int64),
            },
        )
        assert is_float32_array(logprobs)
        assert is_int64_array(logprobs_lens)
        return logprobs, np.minimum(logprobs_lens, logprobs.shape[1])

    def _encode(
        self, features: npt.NDArray[np.float32], features_lens: npt.NDArray[np.int64]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
        return self._encode_with_prompt(
            features,
            features_lens,
            self._languages[self._default_language],
            self._textnorm[self._default_textnorm],
        )

    def _decode_tokens(
        self, ids: Iterable[int], indices: Iterable[int] | None, logprobs: Iterable[float] | None
    ) -> TimestampedResult:
        result = super()._decode_tokens(ids, indices, logprobs)
        result.text = re.sub(RICH_TOKEN_PATTERN, "", result.text).strip()
        return result

    def recognize_batch(
        self, waveforms: npt.NDArray[np.float32], waveforms_len: npt.NDArray[np.int64], /, **kwargs: object | None
    ) -> typing.Iterator[TimestampedResult]:
        """Recognize waveforms batch.

        Args:
            waveforms: Waveforms batch.
            waveforms_len: Waveform lengths.
            language: Language name, one of the keys of `languages` in the model config.
            use_itn: Apply inverse text normalization (punctuation and digits).
            **kwargs: Passed on to the CTC decoding.

        """
        language = self._lookup(
            self._languages, str(kwargs.pop("language", None) or self._default_language), "language"
        )
        use_itn = kwargs.pop("use_itn", None)
        name = self._default_textnorm if use_itn is None else ("withitn" if use_itn else "woitn")
        textnorm = self._lookup(self._textnorm, name, "textnorm mode")

        encoder_out, encoder_out_lens = self._encode_with_prompt(
            *self._features(waveforms, waveforms_len), language, textnorm
        )
        return map(self._decode_tokens, *zip(*self._decoding(encoder_out, encoder_out_lens, **kwargs), strict=False))
