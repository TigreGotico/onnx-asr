"""Tests for the omnilingual-ctc model type with a tiny generated ONNX graph.

The graph stands in for the Omnilingual ASR CTC export: one raw-waveform input `x`,
one `logits` output, no length input and no language selection. Each group of
`SUBSAMPLING` samples becomes one frame, and the frame values are the logits, so a
test waveform names the tokens it wants directly.
"""

import json
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

import onnx_asr
from onnx_asr.models.omnilingual import OmnilingualCtc

SUBSAMPLING = 4
# Index 0 is the blank, index 2 is a real space, exactly as the released vocabulary.
VOCAB = ["<s>", "a", " ", "b"]
BLANK, A, SPACE, B = range(4)


def _make_graph(path: Path) -> None:
    nodes = [
        helper.make_node(
            "Constant", [], ["shape"], value=helper.make_tensor("v", TensorProto.INT64, [3], [0, -1, len(VOCAB)])
        ),
        helper.make_node("Reshape", ["x", "shape"], ["logits"]),
    ]
    graph = helper.make_graph(
        nodes,
        "omnilingual",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", "num_samples"])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", "num_frames", len(VOCAB)])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def _waveform(*token_ids: int) -> np.ndarray:
    """One frame per token id, with that id winning the argmax of the frame."""
    frames = np.zeros((len(token_ids), len(VOCAB)), dtype=np.float32)
    frames[np.arange(len(token_ids)), list(token_ids)] = 10.0
    return frames.reshape(1, -1)


@pytest.fixture
def model_dir(tmp_path: Path) -> Path:
    """An omnilingual-ctc model on disk."""
    _make_graph(tmp_path / "model.onnx")

    with (tmp_path / "tokens.txt").open("wt", encoding="utf-8") as f:
        for index, token in enumerate(VOCAB):
            f.write(f"{token} {index}\n")

    with (tmp_path / "config.json").open("wt", encoding="utf-8") as f:
        json.dump({"model_type": "omnilingual-ctc", "subsampling_factor": SUBSAMPLING}, f)
    return tmp_path


@pytest.fixture
def model(model_dir: Path) -> onnx_asr.adapters.TextResultsAsrAdapter:
    """Loaded model."""
    return onnx_asr.load_model("omnilingual-ctc", model_dir)


def test_loads_registered_model_type(model_dir: Path) -> None:
    assert isinstance(onnx_asr.load_model("omnilingual-ctc", model_dir).asr, OmnilingualCtc)


def test_config_alone_selects_the_model_type(model_dir: Path) -> None:
    assert isinstance(onnx_asr.load_model(str(model_dir), model_dir).asr, OmnilingualCtc)


def test_reads_a_space_token_from_tokens_txt(model: onnx_asr.adapters.TextResultsAsrAdapter) -> None:
    assert isinstance(model.asr, OmnilingualCtc)
    assert model.asr._vocab[SPACE] == " "
    assert model.asr._vocab_size == len(VOCAB)


def test_index_zero_is_the_blank(model: onnx_asr.adapters.TextResultsAsrAdapter) -> None:
    assert isinstance(model.asr, OmnilingualCtc)
    assert model.asr._blank_idx == BLANK


def test_ctc_collapse_keeps_the_space_as_word_separator(
    model: onnx_asr.adapters.TextResultsAsrAdapter,
) -> None:
    waveform = _waveform(A, A, BLANK, SPACE, B, B)
    assert model.recognize(waveform, sample_rate=16_000) == "a b"


def test_repeated_token_needs_a_blank_between(model: onnx_asr.adapters.TextResultsAsrAdapter) -> None:
    assert model.recognize(_waveform(A, BLANK, A), sample_rate=16_000) == "aa"


def test_leading_and_trailing_spaces_are_stripped(model: onnx_asr.adapters.TextResultsAsrAdapter) -> None:
    assert model.recognize(_waveform(SPACE, A, BLANK, SPACE), sample_rate=16_000) == "a"


def test_blank_only_audio_is_empty(model: onnx_asr.adapters.TextResultsAsrAdapter) -> None:
    assert model.recognize(_waveform(BLANK, BLANK), sample_rate=16_000) == ""


def test_batch_of_two_is_padded_and_trimmed(model: onnx_asr.adapters.TextResultsAsrAdapter) -> None:
    short = _waveform(A)[0]
    long = _waveform(B, BLANK, B)[0]
    assert model.recognize([short, long], sample_rate=16_000) == ["a", "bb"]


def test_timestamps_follow_the_subsampling_factor(
    model: onnx_asr.adapters.TextResultsAsrAdapter,
) -> None:
    result = model.with_timestamps().recognize(_waveform(A, BLANK, B), sample_rate=16_000)
    assert result.timestamps == [0.0, pytest.approx(0.08)]


def test_logprobs_are_normalised(model: onnx_asr.adapters.TextResultsAsrAdapter) -> None:
    """Raw graph logits must go through a log-softmax before they are reported."""
    result = model.with_timestamps().recognize(_waveform(A), sample_rate=16_000)
    assert result.logprobs is not None
    assert all(value < 0 for value in result.logprobs)
