"""Tests for the ESPnet model classes with small generated ONNX graphs."""

import json
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from onnx_asr.models.espnet import EspnetAED, EspnetCtc
from onnx_asr.preprocessors.preprocessor import IdentityPreprocessor

VOCAB = ["<blk>", "<unk>", "▁ola", "▁mundo", "s", "▁e", "▁tu", "<sos/eos>"]
VOCAB_SIZE = len(VOCAB)
FEATURES_SIZE = 160
ENCODER_SIZE = 16


def write_model_files(path: Path, **files: onnx.ModelProto) -> dict[str, Path]:
    (path / "vocab.txt").write_text("".join(f"{token} {id}\n" for id, token in enumerate(VOCAB)), encoding="utf-8")
    (path / "config.json").write_text(
        json.dumps({"model_type": "espnet-ctc", "subsampling_factor": 8}), encoding="utf-8"
    )

    model_files = {"vocab": path / "vocab.txt", "config": path / "config.json"}
    for name, model in files.items():
        onnx.save_model(model, path / f"{name}.onnx")
        model_files[name] = path / f"{name}.onnx"
    return model_files


def make_slice_model(output_name: str, size: int, lens_name: str) -> onnx.ModelProto:
    """Graph that returns the first `size` channels of the features and passes lengths through."""
    nodes = [
        helper.make_node("Slice", ["features", "starts", "ends", "axes"], [output_name]),
        helper.make_node("Identity", ["features_lens"], [lens_name]),
    ]
    initializers = [
        numpy_helper.from_array(np.array([0], dtype=np.int64), "starts"),
        numpy_helper.from_array(np.array([size], dtype=np.int64), "ends"),
        numpy_helper.from_array(np.array([2], dtype=np.int64), "axes"),
    ]
    graph = helper.make_graph(
        nodes,
        "test",
        [
            helper.make_tensor_value_info("features", TensorProto.FLOAT, ["b", "t", FEATURES_SIZE]),
            helper.make_tensor_value_info("features_lens", TensorProto.INT64, ["b"]),
        ],
        [
            helper.make_tensor_value_info(output_name, TensorProto.FLOAT, ["b", "t", size]),
            helper.make_tensor_value_info(lens_name, TensorProto.INT64, ["b"]),
        ],
        initializers,
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def make_decoder_model(table: np.ndarray) -> onnx.ModelProto:
    """Graph whose next token distribution depends only on the last token, via a lookup table."""
    nodes = [helper.make_node("Gather", ["table", "tokens"], ["logprobs"], axis=0)]
    graph = helper.make_graph(
        nodes,
        "test",
        [
            helper.make_tensor_value_info("tokens", TensorProto.INT64, ["b", "l"]),
            helper.make_tensor_value_info("encoder_out", TensorProto.FLOAT, ["b", "t", ENCODER_SIZE]),
            helper.make_tensor_value_info("encoder_out_lens", TensorProto.INT64, ["b"]),
        ],
        [helper.make_tensor_value_info("logprobs", TensorProto.FLOAT, ["b", "l", VOCAB_SIZE])],
        [numpy_helper.from_array(table, "table")],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def one_hot_features(token_ids: list[int]) -> np.ndarray:
    features = np.zeros((1, len(token_ids), FEATURES_SIZE), dtype=np.float32)
    features[0, np.arange(len(token_ids)), token_ids] = 1.0
    return features


@pytest.fixture
def ctc_model(tmp_path: Path) -> EspnetCtc:
    model_files = write_model_files(tmp_path, model=make_slice_model("logprobs", VOCAB_SIZE, "logprobs_lens"))
    return EspnetCtc(model_files, lambda _: IdentityPreprocessor(), {})


@pytest.fixture
def aed_model(tmp_path: Path) -> EspnetAED:
    table = np.full((VOCAB_SIZE, VOCAB_SIZE), -10.0, dtype=np.float32)
    # <sos/eos> -> "▁ola" -> "▁mundo" -> <sos/eos>
    table[VOCAB.index("<sos/eos>"), VOCAB.index("▁ola")] = 0.0
    table[VOCAB.index("▁ola"), VOCAB.index("▁mundo")] = 0.0
    table[VOCAB.index("▁mundo"), VOCAB.index("<sos/eos>")] = 0.0

    model_files = write_model_files(
        tmp_path,
        encoder=make_slice_model("encoder_out", ENCODER_SIZE, "encoder_out_lens"),
        decoder=make_decoder_model(table),
    )
    return EspnetAED(model_files, lambda _: IdentityPreprocessor(), {})


def test_get_model_files() -> None:
    assert EspnetCtc._get_model_files() == {
        "model": "model.onnx",
        "vocab": "vocab.txt",
        "config": "config.json",
    }
    assert EspnetCtc._get_model_files("int8")["model"] == "model?int8.onnx"
    assert EspnetAED._get_model_files("int8")["encoder"] == "encoder?int8.onnx"


def test_preprocessor_name(ctc_model: EspnetCtc, aed_model: EspnetAED) -> None:
    assert ctc_model._preprocessor_name == "w2vbert"
    assert aed_model._preprocessor_name == "w2vbert"
    assert ctc_model._subsampling_factor == 8


def test_ctc_greedy_decoding(ctc_model: EspnetCtc) -> None:
    blank = VOCAB.index("<blk>")
    token_ids = [blank, 2, 2, blank, 3, 3, 4, blank]
    features = one_hot_features(token_ids)
    features_lens = np.array([len(token_ids)], dtype=np.int64)

    logprobs, logprobs_lens = ctc_model._encode(features, features_lens)
    assert logprobs.shape == (1, len(token_ids), VOCAB_SIZE)
    np.testing.assert_equal(logprobs_lens, features_lens)

    (result,) = ctc_model.recognize_batch(features, features_lens)
    assert result.text == "ola mundos"
    assert result.tokens == [" ola", " mundo", "s"]
    assert result.timestamps == [0.08 * 1, 0.08 * 4, 0.08 * 6]


def test_ctc_greedy_decoding_respects_lengths(ctc_model: EspnetCtc) -> None:
    features = one_hot_features([2, 0, 3, 0, 6])
    (result,) = ctc_model.recognize_batch(features, np.array([3], dtype=np.int64))
    assert result.text == "ola mundo"


def test_aed_greedy_decoding(aed_model: EspnetAED) -> None:
    features = np.zeros((1, 6, FEATURES_SIZE), dtype=np.float32)
    features_lens = np.array([6], dtype=np.int64)

    encoder_out, encoder_out_lens = aed_model._encode(features, features_lens)
    assert encoder_out.shape == (1, 6, ENCODER_SIZE)
    np.testing.assert_equal(encoder_out_lens, features_lens)

    (result,) = aed_model.recognize_batch(features, features_lens)
    assert result.text == "ola mundo"
    assert result.tokens == [" ola", " mundo"]
    assert result.timestamps is None


def test_aed_greedy_decoding_stops_at_max_length(tmp_path: Path) -> None:
    # A table that never emits <sos/eos> must still stop at the encoder length.
    table = np.full((VOCAB_SIZE, VOCAB_SIZE), -10.0, dtype=np.float32)
    table[:, VOCAB.index("s")] = 0.0
    model_files = write_model_files(
        tmp_path,
        encoder=make_slice_model("encoder_out", ENCODER_SIZE, "encoder_out_lens"),
        decoder=make_decoder_model(table),
    )
    model = EspnetAED(model_files, lambda _: IdentityPreprocessor(), {})

    features = np.zeros((1, 4, FEATURES_SIZE), dtype=np.float32)
    (result,) = model.recognize_batch(features, np.array([4], dtype=np.int64))
    assert result.text == "ssss"
