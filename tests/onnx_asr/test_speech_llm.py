"""Tests for the speech-llm model family with tiny generated ONNX graphs."""

import json
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from onnx_asr.asr import Preprocessor
from onnx_asr.models.speech_llm import SpeechLlm, _post_cnn_length

HIDDEN = 4
VOCAB = 8
NUM_LAYERS = 1
NUM_KV = 1
HEAD_DIM = 2
EOS_ID = 7
FIRST_ID = 5
MEL_BINS = 128

VOCAB_JSON = {
    "<|endoftext|>": 0,
    "a": 1,
    "b": 2,
    "c": 3,
    "d": 4,
    "Ġhello": FIRST_ID,
    "Ġworld": FIRST_ID + 1,
    "<|im_end|>": EOS_ID,
}


def _const(name: str, array: np.ndarray) -> onnx.NodeProto:
    return helper.make_node("Constant", [], [name], value=numpy_helper.from_array(array, name + "_value"))


def _save(graph: onnx.GraphProto, path: Path) -> None:
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def _make_encoder(path: Path) -> None:
    """Audio embeddings are zeros, but the graph checks the input contract."""
    nodes = [
        _const("hidden", np.array([HIDDEN], dtype=np.int64)),
        _const("one", np.array([1], dtype=np.int64)),
        helper.make_node("Shape", ["valid_indices"], ["seq"]),
        helper.make_node("Concat", ["one", "seq", "hidden"], ["out_shape"], axis=0),
        helper.make_node(
            "ConstantOfShape",
            ["out_shape"],
            ["audio_embeds"],
            value=numpy_helper.from_array(np.array([0.0], dtype=np.float32), "zero"),
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "encoder",
        [
            helper.make_tensor_value_info("input_features", TensorProto.FLOAT, [1, MEL_BINS, "padded_frames"]),
            helper.make_tensor_value_info("valid_indices", TensorProto.INT64, ["audio_seq"]),
            helper.make_tensor_value_info("attn_bias", TensorProto.FLOAT, [1, 1, "audio_seq", "audio_seq"]),
        ],
        [helper.make_tensor_value_info("audio_embeds", TensorProto.FLOAT, [1, "audio_seq", HIDDEN])],
    )
    _save(graph, path)


def _make_fixed_encoder(path: Path, audio_length: int) -> None:
    """A Whisper style encoder: only input_features, and a fixed number of audio frames."""
    nodes = [
        _const("out_shape", np.array([1, audio_length, HIDDEN], dtype=np.int64)),
        helper.make_node(
            "ConstantOfShape",
            ["out_shape"],
            ["audio_embeds"],
            value=numpy_helper.from_array(np.array([0.0], dtype=np.float32), "zero"),
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "encoder",
        [helper.make_tensor_value_info("input_features", TensorProto.FLOAT, [1, MEL_BINS, 3000])],
        [helper.make_tensor_value_info("audio_embeds", TensorProto.FLOAT, [1, audio_length, HIDDEN])],
    )
    _save(graph, path)


def _make_embed_tokens(path: Path) -> None:
    """Every token embeds to its own id repeated over the hidden dimension."""
    nodes = [
        _const("axis2", np.array([2], dtype=np.int64)),
        _const("hidden", np.array([HIDDEN], dtype=np.int64)),
        helper.make_node("Cast", ["input_ids"], ["ids_f"], to=TensorProto.FLOAT),
        helper.make_node("Unsqueeze", ["ids_f", "axis2"], ["ids_3d"]),
        helper.make_node("Shape", ["ids_3d"], ["ids_shape"]),
        _const("starts", np.array([0], dtype=np.int64)),
        _const("ends", np.array([2], dtype=np.int64)),
        helper.make_node("Slice", ["ids_shape", "starts", "ends"], ["batch_seq"]),
        helper.make_node("Concat", ["batch_seq", "hidden"], ["target"], axis=0),
        helper.make_node("Expand", ["ids_3d", "target"], ["inputs_embeds"]),
    ]
    graph = helper.make_graph(
        nodes,
        "embed_tokens",
        [helper.make_tensor_value_info("input_ids", TensorProto.INT64, [1, "seq"])],
        [helper.make_tensor_value_info("inputs_embeds", TensorProto.FLOAT, [1, "seq", HIDDEN])],
    )
    _save(graph, path)


def _make_decoder(path: Path, prefill_length: int) -> None:
    """Emit FIRST_ID, FIRST_ID + 1, then EOS, driven only by the KV cache length."""
    nodes = [
        _const("axis2", np.array([2], dtype=np.int64)),
        _const("kv_shape_tail", np.array([1, NUM_KV], dtype=np.int64)),
        _const("head_dim", np.array([HEAD_DIM], dtype=np.int64)),
        _const("zero_i", np.array([0], dtype=np.int64)),
        _const("two_i", np.array([2], dtype=np.int64)),
        _const("three_i", np.array([3], dtype=np.int64)),
        _const("vocab_i", np.array([VOCAB], dtype=np.int64)),
        _const("one_i", np.array([1], dtype=np.int64)),
        _const("bias", np.array(FIRST_ID - prefill_length, dtype=np.float32)),
        _const("targets", np.arange(VOCAB, dtype=np.float32)),
        # new (all-zero) cache entries for the current step
        helper.make_node("Shape", ["inputs_embeds"], ["embeds_shape"]),
        helper.make_node("Slice", ["embeds_shape", "one_i", "two_i"], ["seq_len"]),
        helper.make_node("Concat", ["kv_shape_tail", "seq_len", "head_dim"], ["new_kv_shape"], axis=0),
        helper.make_node(
            "ConstantOfShape",
            ["new_kv_shape"],
            ["new_kv"],
            value=numpy_helper.from_array(np.array([0.0], dtype=np.float32), "kv_zero"),
        ),
        helper.make_node("Concat", ["past_key_values.0.key", "new_kv"], ["present.0.key"], axis=2),
        helper.make_node("Concat", ["past_key_values.0.value", "new_kv"], ["present.0.value"], axis=2),
        # logits peak at (total_length - prefill_length + FIRST_ID)
        helper.make_node("Shape", ["present.0.key"], ["present_shape"]),
        helper.make_node("Slice", ["present_shape", "two_i", "three_i"], ["total"]),
        helper.make_node("Cast", ["total"], ["total_f"], to=TensorProto.FLOAT),
        helper.make_node("Add", ["total_f", "bias"], ["peak"]),
        helper.make_node("Sub", ["targets", "peak"], ["delta"]),
        helper.make_node("Mul", ["delta", "delta"], ["delta2"]),
        helper.make_node("Neg", ["delta2"], ["row"]),
        helper.make_node("Unsqueeze", ["row", "zero_i"], ["row_2d"]),
        helper.make_node("Unsqueeze", ["row_2d", "zero_i"], ["row_3d"]),
        helper.make_node("Concat", ["one_i", "seq_len", "vocab_i"], ["logits_shape"], axis=0),
        helper.make_node("Expand", ["row_3d", "logits_shape"], ["logits"]),
    ]
    graph = helper.make_graph(
        nodes,
        "decoder",
        [
            helper.make_tensor_value_info("inputs_embeds", TensorProto.FLOAT, [1, "seq", HIDDEN]),
            helper.make_tensor_value_info("attn_bias", TensorProto.FLOAT, [1, 1, "seq", "total"]),
            helper.make_tensor_value_info("position_ids", TensorProto.INT64, [1, "seq"]),
            helper.make_tensor_value_info("past_key_values.0.key", TensorProto.FLOAT, [1, NUM_KV, "past", HEAD_DIM]),
            helper.make_tensor_value_info("past_key_values.0.value", TensorProto.FLOAT, [1, NUM_KV, "past", HEAD_DIM]),
        ],
        [
            helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, "seq", VOCAB]),
            helper.make_tensor_value_info("present.0.key", TensorProto.FLOAT, [1, NUM_KV, "total", HEAD_DIM]),
            helper.make_tensor_value_info("present.0.value", TensorProto.FLOAT, [1, NUM_KV, "total", HEAD_DIM]),
        ],
    )
    _save(graph, path)


def _fake_preprocessor(_name: str) -> Preprocessor:
    def preprocessor(waveforms: np.ndarray, waveforms_lens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        frames = np.zeros((waveforms.shape[0], MEL_BINS, 3000), dtype=np.float32)
        return frames, waveforms_lens

    return preprocessor


PREFIX_IDS = [1, 2]
SUFFIX_IDS = [3, 4]
AUDIO_SECONDS = 2.0


@pytest.fixture
def model_dir(tmp_path: Path) -> Path:
    feature_len = int(AUDIO_SECONDS * 16_000) // 160
    audio_length = int(_post_cnn_length(np.clip(feature_len - np.arange(0, feature_len + 100, 100), 0, 100)).sum())
    prefill = len(PREFIX_IDS) + audio_length + len(SUFFIX_IDS)

    _make_encoder(tmp_path / "encoder.onnx")
    _make_embed_tokens(tmp_path / "embed_tokens.onnx")
    _make_decoder(tmp_path / "decoder.onnx", prefill)

    with (tmp_path / "vocab.json").open("wt", encoding="utf-8") as f:
        json.dump(VOCAB_JSON, f)

    config = {
        "model_type": "speech-llm",
        "features_size": MEL_BINS,
        "n_window": 50,
        "n_window_infer": 800,
        "eos_token_ids": [EOS_ID],
        "max_sequence_length": 16,
        "prompt_prefix_ids": PREFIX_IDS,
        "prompt_suffix_ids": SUFFIX_IDS,
        "language_prompt_ids": {"": PREFIX_IDS, "English": [1, 1]},
    }
    with (tmp_path / "config.json").open("wt", encoding="utf-8") as f:
        json.dump(config, f)
    return tmp_path


@pytest.fixture
def model(model_dir: Path) -> SpeechLlm:
    files = {name: model_dir / f"{name}.onnx" for name in ("encoder", "embed_tokens", "decoder")} | {
        "vocab": model_dir / "vocab.json",
        "config": model_dir / "config.json",
    }
    return SpeechLlm(files, _fake_preprocessor, {})


def test_post_cnn_length() -> None:
    lengths = np.array([0, 1, 50, 100], dtype=np.int64)
    assert _post_cnn_length(lengths).tolist() == [0, 1, 7, 13]


@pytest.mark.parametrize(
    ("feature_len", "expected_frames", "expected_seq", "expected_blocks"),
    [(250, 300, 13 + 13 + 7, [33]), (100, 100, 13, [13]), (1000, 1000, 130, [104, 26])],
)
def test_audio_windows(
    model: SpeechLlm, feature_len: int, expected_frames: int, expected_seq: int, expected_blocks: list[int]
) -> None:
    padded_frames, valid_indices, bias = model._audio_windows(feature_len)

    assert padded_frames == expected_frames
    assert len(valid_indices) == expected_seq
    assert bias.shape == (1, 1, expected_seq, expected_seq)

    # the bias must be block diagonal with the expected block sizes
    allowed = bias[0, 0] == 0.0
    offset = 0
    for size in expected_blocks:
        assert allowed[offset : offset + size, offset : offset + size].all()
        offset += size
    assert offset == expected_seq
    assert allowed.sum() == sum(size * size for size in expected_blocks)


def test_recognize_batch(model: SpeechLlm) -> None:
    samples = int(AUDIO_SECONDS * 16_000)
    waveforms = np.zeros((1, samples), dtype=np.float32)
    waveforms_len = np.array([samples], dtype=np.int64)

    results = list(model.recognize_batch(waveforms, waveforms_len))

    assert len(results) == 1
    assert results[0].text == "hello world"


def test_recognize_batch_with_language(model: SpeechLlm) -> None:
    samples = int(AUDIO_SECONDS * 16_000)
    waveforms = np.zeros((2, samples), dtype=np.float32)
    waveforms_len = np.array([samples, samples], dtype=np.int64)

    results = list(model.recognize_batch(waveforms, waveforms_len, language="English"))

    assert [result.text for result in results] == ["hello world", "hello world"]


SP_VOCAB = {
    "<s>": 0,
    "</s>": 1,
    "▁olá": FIRST_ID,
    "<0xC3>": FIRST_ID + 1,
    "<|im_end|>": EOS_ID,
}


@pytest.fixture
def slam_model(tmp_path: Path) -> SpeechLlm:
    """A SLAM-ASR shaped model: fixed encoder, no prompt prefix, SentencePiece vocabulary."""
    audio_length = 6
    prefill = audio_length + len(SUFFIX_IDS)

    _make_fixed_encoder(tmp_path / "encoder.onnx", audio_length)
    _make_embed_tokens(tmp_path / "embed_tokens.onnx")
    _make_decoder(tmp_path / "decoder.onnx", prefill)

    with (tmp_path / "vocab.json").open("wt", encoding="utf-8") as f:
        json.dump(SP_VOCAB, f)

    config = {
        "model_type": "speech-llm",
        "features_size": MEL_BINS,
        "tokenizer_type": "sentencepiece",
        "normalize_waveform": True,
        "eos_token_ids": [EOS_ID],
        "max_sequence_length": 16,
        "prompt_prefix_ids": [],
        "prompt_suffix_ids": SUFFIX_IDS,
    }
    with (tmp_path / "config.json").open("wt", encoding="utf-8") as f:
        json.dump(config, f)

    files = {name: tmp_path / f"{name}.onnx" for name in ("encoder", "embed_tokens", "decoder")} | {
        "vocab": tmp_path / "vocab.json",
        "config": tmp_path / "config.json",
    }
    return SpeechLlm(files, _fake_preprocessor, {})


def test_slam_layout_recognize(slam_model: SpeechLlm) -> None:
    samples = int(AUDIO_SECONDS * 16_000)
    rng = np.random.default_rng(0)
    waveforms = rng.standard_normal((1, samples), dtype=np.float32) * 1e-3
    waveforms_len = np.array([samples], dtype=np.int64)

    results = list(slam_model.recognize_batch(waveforms, waveforms_len))

    # "▁olá" + "<0xC3>" is the space marker, the word, and a dangling byte-fallback token.
    assert results[0].text == "olá�"


def test_waveform_normalization_is_applied(slam_model: SpeechLlm) -> None:
    seen: list[np.ndarray] = []
    preprocessor = slam_model._preprocessor

    def _tracking_preprocessor(
        waveforms: npt.NDArray[np.float32], lens: npt.NDArray[np.int64]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
        seen.append(waveforms)
        return preprocessor(waveforms, lens)

    slam_model._preprocessor = cast(Preprocessor, _tracking_preprocessor)

    samples = int(AUDIO_SECONDS * 16_000)
    waveforms = np.full((1, samples), 3.0, dtype=np.float32)
    waveforms[0, ::2] = 1.0
    list(slam_model.recognize_batch(waveforms, np.array([samples], dtype=np.int64)))

    assert seen[0].mean() == pytest.approx(0.0, abs=1e-5)
    assert seen[0].std() == pytest.approx(1.0, abs=1e-4)


LANGUAGE_SUFFIX_IDS = [3, 5]


@pytest.fixture
def voxtral_model(tmp_path: Path) -> SpeechLlm:
    """A Voxtral shaped model: fixed encoder and the language marker after the audio."""
    audio_length = 6
    prefill = len(PREFIX_IDS) + audio_length + len(SUFFIX_IDS)

    _make_fixed_encoder(tmp_path / "encoder.onnx", audio_length)
    _make_embed_tokens(tmp_path / "embed_tokens.onnx")
    _make_decoder(tmp_path / "decoder.onnx", prefill)

    with (tmp_path / "vocab.json").open("wt", encoding="utf-8") as f:
        json.dump(VOCAB_JSON, f)

    config = {
        "model_type": "speech-llm",
        "features_size": MEL_BINS,
        "eos_token_ids": [EOS_ID],
        "max_sequence_length": 16,
        "prompt_prefix_ids": PREFIX_IDS,
        "prompt_suffix_ids": SUFFIX_IDS,
        "language_suffix_ids": {"en": LANGUAGE_SUFFIX_IDS, "English": LANGUAGE_SUFFIX_IDS},
    }
    with (tmp_path / "config.json").open("wt", encoding="utf-8") as f:
        json.dump(config, f)

    files = {name: tmp_path / f"{name}.onnx" for name in ("encoder", "embed_tokens", "decoder")} | {
        "vocab": tmp_path / "vocab.json",
        "config": tmp_path / "config.json",
    }
    return SpeechLlm(files, _fake_preprocessor, {})


@pytest.mark.parametrize(
    ("language", "expected_suffix"), [(None, SUFFIX_IDS), ("en", LANGUAGE_SUFFIX_IDS), ("english", LANGUAGE_SUFFIX_IDS)]
)
def test_language_suffix_ids(voxtral_model: SpeechLlm, language: str | None, expected_suffix: list[int]) -> None:
    seen: list[list[int]] = []
    embed = voxtral_model._embed

    def _tracking_embed(ids: list[int]) -> npt.NDArray[np.float32]:
        seen.append(ids)
        return embed(ids)

    voxtral_model._embed = _tracking_embed  # type: ignore[method-assign,assignment]

    samples = int(AUDIO_SECONDS * 16_000)
    results = list(
        voxtral_model.recognize_batch(
            np.zeros((1, samples), dtype=np.float32), np.array([samples], dtype=np.int64), language=language
        )
    )

    assert seen[:2] == [PREFIX_IDS, expected_suffix]
    assert results[0].text == "hello world"
