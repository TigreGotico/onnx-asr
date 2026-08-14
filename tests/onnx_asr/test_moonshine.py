"""Moonshine tests against tiny hand-built ONNX graphs (no downloads)."""

import json
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto
from onnx import helper as h
from onnx import numpy_helper as nh

import onnx_asr
from onnx_asr.adapters import TextResultsAsrAdapter
from onnx_asr.asr import BaseAsr
from onnx_asr.models.moonshine import Moonshine
from onnx_asr.preprocessors.preprocessor import IdentityPreprocessor

HIDDEN = 8
HEADS = 2
HEAD_DIM = 4
VOCAB = 8
BOS, EOS = 1, 6

# id -> (token, special)
TOKENS = [
    ("<unk>", True),
    ("<s>", True),
    ("▁he", False),
    ("llo", False),
    ("<0x21>", False),
    ("▁w", False),
    ("</s>", True),
    ("orld", False),
]


def _const(name: str, array: np.ndarray) -> onnx.NodeProto:
    return h.make_node("Constant", [], [name], value=nh.from_array(array, name))


def _build_encoder(path: Path) -> None:
    """last_hidden_state[b, 1, HIDDEN] = mean(input_values[b])."""
    nodes = [
        _const("ones", np.ones((1, 1, HIDDEN), dtype=np.float32)),
        _const("axes1", np.array([1], dtype=np.int64)),
        _const("axes2", np.array([2], dtype=np.int64)),
        h.make_node("ReduceMean", ["input_values", "axes1"], ["mean"], keepdims=1),
        h.make_node("Unsqueeze", ["mean", "axes2"], ["mean3d"]),
        h.make_node("Mul", ["mean3d", "ones"], ["last_hidden_state"]),
    ]
    graph = h.make_graph(
        nodes,
        "encoder",
        [h.make_tensor_value_info("input_values", TensorProto.FLOAT, ["batch_size", "num_samples"])],
        [h.make_tensor_value_info("last_hidden_state", TensorProto.FLOAT, ["batch_size", 1, HIDDEN])],
    )
    onnx.save(h.make_model(graph, opset_imports=[h.make_opsetid("", 18)]), path)


def _build_decoder(path: Path) -> None:
    """logits peak at input_ids + 1, so greedy decoding walks BOS -> ... -> EOS.

    Logits are built as the negated square of (vocab_range - input_ids - 1).

    The KV cache is shape-only: `present.*` just grows by the current sequence
    length, which is all the runtime cache bookkeeping actually reads.
    """
    kv_shape: list[str | int] = ["batch_size", HEADS, "past_decoder_sequence_length", HEAD_DIM]
    inputs = [
        h.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch_size", "decoder_sequence_length"]),
        h.make_tensor_value_info(
            "encoder_hidden_states", TensorProto.FLOAT, ["batch_size", "encoder_sequence_length", HIDDEN]
        ),
        h.make_tensor_value_info("past_key_values.0.decoder.key", TensorProto.FLOAT, kv_shape),
        h.make_tensor_value_info("past_key_values.0.decoder.value", TensorProto.FLOAT, kv_shape),
        h.make_tensor_value_info("past_key_values.0.encoder.key", TensorProto.FLOAT, kv_shape),
        h.make_tensor_value_info("past_key_values.0.encoder.value", TensorProto.FLOAT, kv_shape),
        h.make_tensor_value_info("use_cache_branch", TensorProto.BOOL, [1]),
    ]
    outputs = [
        h.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch_size", "decoder_sequence_length", VOCAB]),
        *(
            h.make_tensor_value_info(f"present.0.{part}.{kind}", TensorProto.FLOAT, kv_shape)
            for part in ("decoder", "encoder")
            for kind in ("key", "value")
        ),
    ]

    nodes = [
        _const("vocab_range", np.arange(VOCAB, dtype=np.float32).reshape(1, 1, VOCAB)),
        _const("one_f", np.array(1.0, dtype=np.float32)),
        _const("axes2", np.array([2], dtype=np.int64)),
        h.make_node("Cast", ["input_ids"], ["ids_f"], to=TensorProto.FLOAT),
        h.make_node("Unsqueeze", ["ids_f", "axes2"], ["ids_3d"]),
        h.make_node("Add", ["ids_3d", "one_f"], ["target"]),
        h.make_node("Sub", ["vocab_range", "target"], ["delta"]),
        h.make_node("Mul", ["delta", "delta"], ["delta_sq"]),
        h.make_node("Neg", ["delta_sq"], ["logits"]),
        # decoder cache grows by decoder_sequence_length, encoder cache sized by the encoder output
        _const("kv_mid", np.array([HEADS], dtype=np.int64)),
        _const("kv_last", np.array([HEAD_DIM], dtype=np.int64)),
        h.make_node("Shape", ["input_ids"], ["batch"], start=0, end=1),
        h.make_node("Shape", ["input_ids"], ["seq"], start=1, end=2),
        h.make_node("Shape", ["encoder_hidden_states"], ["enc_seq"], start=1, end=2),
        h.make_node("Concat", ["batch", "kv_mid", "seq", "kv_last"], ["new_shape"], axis=0),
        h.make_node("Concat", ["batch", "kv_mid", "enc_seq", "kv_last"], ["enc_shape"], axis=0),
        h.make_node("ConstantOfShape", ["new_shape"], ["new_kv"], value=nh.from_array(np.zeros(1, np.float32))),
        h.make_node("ConstantOfShape", ["enc_shape"], ["enc_kv"], value=nh.from_array(np.zeros(1, np.float32))),
        h.make_node("Concat", ["past_key_values.0.decoder.key", "new_kv"], ["present.0.decoder.key"], axis=2),
        h.make_node("Concat", ["past_key_values.0.decoder.value", "new_kv"], ["present.0.decoder.value"], axis=2),
        h.make_node("Identity", ["enc_kv"], ["present.0.encoder.key"]),
        h.make_node("Identity", ["enc_kv"], ["present.0.encoder.value"]),
    ]
    graph = h.make_graph(nodes, "decoder", inputs, outputs)
    onnx.save(h.make_model(graph, opset_imports=[h.make_opsetid("", 18)]), path)


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("moonshine")
    (path / "onnx").mkdir()
    _build_encoder(path / "onnx" / "encoder_model.onnx")
    _build_decoder(path / "onnx" / "decoder_model_merged.onnx")

    (path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "moonshine",
                "decoder_start_token_id": BOS,
                "eos_token_id": EOS,
                "max_position_embeddings": 512,
            }
        )
    )
    (path / "tokenizer.json").write_text(
        json.dumps(
            {
                "added_tokens": [{"id": id, "content": token} for id, (token, special) in enumerate(TOKENS) if special],
                "model": {"type": "BPE", "vocab": {token: id for id, (token, _) in enumerate(TOKENS)}},
            }
        )
    )
    return path


@pytest.fixture(scope="module")
def model(model_dir: Path) -> TextResultsAsrAdapter:
    return onnx_asr.load_model("moonshine", model_dir)


def test_model_type_resolved_from_config(model_dir: Path) -> None:
    resolver = onnx_asr.loader.create_asr_resolver(None, model_dir)
    assert resolver.model_type is Moonshine


def test_preprocessor_is_identity(model: TextResultsAsrAdapter) -> None:
    asr = model.asr
    assert isinstance(asr, BaseAsr)
    assert isinstance(asr._preprocessor, IdentityPreprocessor)


def test_recognize_walks_greedy_chain_to_eos(model: TextResultsAsrAdapter) -> None:
    rng = np.random.default_rng(0)
    waveform = rng.random(16_000, dtype=np.float32)

    # BOS(1) -> 2 -> 3 -> 4 -> 5 -> EOS(6); specials dropped, <0x21> is byte fallback "!"
    assert model.recognize(waveform) == "hello! w"


def test_recognize_batch_shares_result(model: TextResultsAsrAdapter) -> None:
    rng = np.random.default_rng(0)
    waveforms: list[str | Path | np.ndarray] = [
        rng.random(16_000, dtype=np.float32),
        rng.random(8_000, dtype=np.float32),
    ]

    results = model.recognize(waveforms)
    assert results == ["hello! w", "hello! w"]


def test_empty_recognize(model: TextResultsAsrAdapter) -> None:
    assert model.recognize([]) == []


def test_max_length_is_capped_by_audio_duration(model: TextResultsAsrAdapter) -> None:
    rng = np.random.default_rng(0)
    # 0.2 s of audio => 1 + int(0.2 * 6) = 2 tokens, so decoding stops before EOS
    assert model.recognize(rng.random(3_200, dtype=np.float32)) == "he"


def test_decode_tokens_handles_multibyte_fallback(model: TextResultsAsrAdapter) -> None:
    asr = model.asr
    assert isinstance(asr, Moonshine)
    asr._vocab |= {100: "<0xC3>", 101: "<0xA9>"}

    assert asr._decode_tokens(np.array([BOS, 100, 101, 3, EOS])).text == "éllo"


def test_decode_tokens_replaces_invalid_utf8(model: TextResultsAsrAdapter) -> None:
    asr = model.asr
    assert isinstance(asr, Moonshine)
    asr._vocab |= {102: "<0xFF>"}

    assert asr._decode_tokens(np.array([BOS, 102, EOS])).text == "�"


@pytest.mark.parametrize(
    ("quantization", "expected"),
    [(None, "**/encoder_model.onnx"), ("int8", "**/encoder_model?int8.onnx")],
)
def test_model_files_globs(quantization: str | None, expected: str) -> None:
    files = Moonshine._get_model_files(quantization)
    assert files["encoder"] == expected
    assert files["tokenizer"] == "tokenizer.json"
