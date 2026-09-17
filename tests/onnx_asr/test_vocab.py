from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from onnx_asr.asr import Preprocessor, _AsrWithDecoding


class _VocabOnlyAsr(_AsrWithDecoding):
    """Reads the vocabulary and nothing else, so the parser can be tested alone."""

    @staticmethod
    def _get_model_files(quantization: str | None = None) -> dict[str, str]:
        return {"vocab": "tokens.txt"}

    @property
    def _preprocessor_name(self) -> str:
        return "identity"

    @property
    def _subsampling_factor(self) -> int:
        return 320

    def _encode(
        self, features: npt.NDArray[np.float32], features_lens: npt.NDArray[np.int64]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
        raise NotImplementedError

    def _decoding(
        self, encoder_out: npt.NDArray[np.float32], encoder_out_lens: npt.NDArray[np.int64], /, **kwargs: object | None
    ) -> Iterator[tuple[Iterable[int], Iterable[int] | None, Iterable[float] | None]]:
        raise NotImplementedError


def _no_preprocessor(_name: str) -> Preprocessor:
    def preprocessor(
        waveforms: npt.NDArray[np.float32], waveforms_lens: npt.NDArray[np.int64]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
        return waveforms, waveforms_lens

    return preprocessor


def _load_vocab(tmp_path: Path, lines: list[str]) -> dict[int, str]:
    file = tmp_path.joinpath("tokens.txt")
    file.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    model = _VocabOnlyAsr({"vocab": file}, _no_preprocessor, {})
    return model._vocab


def test_vocab_reads_plain_tokens(tmp_path: Path) -> None:
    vocab = _load_vocab(tmp_path, ["<blk> 0", "a 1", "b 2"])
    assert vocab == {0: "<blk>", 1: "a", 2: "b"}


def test_vocab_converts_word_delimiter(tmp_path: Path) -> None:
    vocab = _load_vocab(tmp_path, ["<blk> 0", "▁ 1"])
    assert vocab[1] == " "


def test_vocab_reads_a_space_token(tmp_path: Path) -> None:
    """A sherpa-onnx tokens.txt can hold a literal space as a token."""
    vocab = _load_vocab(tmp_path, ["<s> 0", "<pad> 1", "  2", "a 3"])
    assert vocab == {0: "<s>", 1: "<pad>", 2: " ", 3: "a"}


def test_vocab_ignores_a_trailing_blank_line(tmp_path: Path) -> None:
    file = tmp_path.joinpath("tokens.txt")
    file.write_text("<blk> 0\na 1\n\n", encoding="utf-8")
    model = _VocabOnlyAsr({"vocab": file}, _no_preprocessor, {})
    assert model._vocab == {0: "<blk>", 1: "a"}


def test_vocab_finds_the_blank_token(tmp_path: Path) -> None:
    file = tmp_path.joinpath("tokens.txt")
    file.write_text("a 0\n<blk> 1\n", encoding="utf-8")
    model = _VocabOnlyAsr({"vocab": file}, _no_preprocessor, {})
    assert model._blank_idx == 1


def test_vocab_rejects_a_line_without_an_id(tmp_path: Path) -> None:
    file = tmp_path.joinpath("tokens.txt")
    file.write_text("<blk> 0\nbroken\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"not enough values to unpack|invalid literal"):
        _VocabOnlyAsr({"vocab": file}, _no_preprocessor, {})
