"""Tests for the wav2vec2-adapters model type with a tiny generated ONNX graph.

The graph is a stand-in for the shared MMS base: the CTC head is a graph input, so
two fake languages with different vocabulary sizes run through one session.
"""

import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as rt
import pytest
from onnx import TensorProto, helper

import onnx_asr
from onnx_asr.models.wav2vec2_adapters import (
    LanguageNotFoundError,
    LanguageNotSpecifiedError,
    MissingAdapterInputsError,
    Wav2Vec2Adapters,
)

HIDDEN = 4
SUBSAMPLING = 4
# "aa" for the first language, "bb" for the second, both after CTC collapse.
XX_VOCAB = ["<blk>", "a", "▁"]
YY_VOCAB = ["<blk>", "b", "c", "d", "▁"]


def _savez(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with path.open("wb") as f:
        np.savez(f, **arrays)  # type: ignore[arg-type]


def _make_base(path: Path) -> None:
    """Frame features are `input_values` reshaped, then projected by the head input."""
    nodes = [
        helper.make_node(
            "Constant", [], ["shape"], value=helper.make_tensor("v", TensorProto.INT64, [3], [0, -1, HIDDEN])
        ),
        helper.make_node("Reshape", ["input_values", "shape"], ["frames"]),
        helper.make_node("MatMul", ["frames", "lm_head.weight"], ["projected"]),
        helper.make_node("Add", ["projected", "lm_head.bias"], ["biased"]),
        helper.make_node("Mul", ["biased", "adapter.0.scale"], ["logits"]),
        helper.make_node("LogSoftmax", ["logits"], ["logprobs"], axis=-1),
    ]
    graph = helper.make_graph(
        nodes,
        "base",
        [
            helper.make_tensor_value_info("input_values", TensorProto.FLOAT, ["batch", "time"]),
            helper.make_tensor_value_info("input_lengths", TensorProto.INT64, ["batch"]),
            helper.make_tensor_value_info("adapter.0.scale", TensorProto.FLOAT, [1]),
            helper.make_tensor_value_info("lm_head.weight", TensorProto.FLOAT, [HIDDEN, "vocab"]),
            helper.make_tensor_value_info("lm_head.bias", TensorProto.FLOAT, ["vocab"]),
        ],
        [helper.make_tensor_value_info("logprobs", TensorProto.FLOAT, ["batch", "frames", "vocab"])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def _write_pack(directory: Path, language: str, vocab: list[str], token: str) -> None:
    """A head that always argmaxes to `token`, whatever the audio is."""
    weight = np.zeros((HIDDEN, len(vocab)), dtype=np.float32)
    bias = np.zeros(len(vocab), dtype=np.float32)
    bias[vocab.index(token)] = 10.0
    _savez(
        directory / f"{language}.npz",
        {
            "adapter.0.scale": np.ones(1, dtype=np.float32),
            "lm_head.weight": weight,
            "lm_head.bias": bias,
            "vocab_size": np.array(len(vocab), dtype=np.int64),
        },
    )


@pytest.fixture
def model_dir(tmp_path: Path) -> Path:
    """A two-language adapter model on disk."""
    (tmp_path / "adapters").mkdir()
    (tmp_path / "vocabs").mkdir()
    _make_base(tmp_path / "model.onnx")

    for language, vocab, token in (("xx", XX_VOCAB, "a"), ("yy", YY_VOCAB, "b")):
        _write_pack(tmp_path / "adapters", language, vocab, token)
        with (tmp_path / "vocabs" / f"{language}.txt").open("wt", encoding="utf-8") as f:
            for idx, item in enumerate(vocab):
                f.write(f"{item} {idx}\n")

    with (tmp_path / "config.json").open("wt", encoding="utf-8") as f:
        json.dump(
            {
                "model_type": "wav2vec2-adapters",
                "subsampling_factor": SUBSAMPLING,
                "default_language": "xx",
                "language_aliases": {"zz": "yy"},
            },
            f,
        )
    return tmp_path


@pytest.fixture
def model(model_dir: Path) -> onnx_asr.adapters.TextResultsAsrAdapter:
    """Loaded model."""
    return onnx_asr.load_model("wav2vec2-adapters", model_dir)


def _waveform(frames: int = 3) -> np.ndarray:
    return np.arange(frames * HIDDEN * SUBSAMPLING // SUBSAMPLING, dtype=np.float32).reshape(1, -1)


def test_loads_registered_model_type(model_dir: Path) -> None:
    assert isinstance(onnx_asr.load_model("wav2vec2-adapters", model_dir).asr, Wav2Vec2Adapters)


def test_lists_languages(model: onnx_asr.adapters.TextResultsAsrAdapter) -> None:
    assert isinstance(model.asr, Wav2Vec2Adapters)
    assert model.asr.languages == ["xx", "yy"]


def test_default_language_is_used(model: onnx_asr.adapters.TextResultsAsrAdapter) -> None:
    assert model.recognize(_waveform(), sample_rate=16_000) == "a"


def test_second_language_has_its_own_vocab_size(model: onnx_asr.adapters.TextResultsAsrAdapter) -> None:
    assert model.recognize(_waveform(), sample_rate=16_000, language="yy") == "b"


def test_language_switches_back(model: onnx_asr.adapters.TextResultsAsrAdapter) -> None:
    model.recognize(_waveform(), sample_rate=16_000, language="yy")
    assert model.recognize(_waveform(), sample_rate=16_000, language="xx") == "a"


@pytest.mark.parametrize("language", ["yy", "yy-ZZ", "zz", "zz-Latn-ZZ"])
def test_bcp47_and_alias_resolution(model: onnx_asr.adapters.TextResultsAsrAdapter, language: str) -> None:
    assert model.recognize(_waveform(), sample_rate=16_000, language=language) == "b"


def test_unknown_language_lists_available(model: onnx_asr.adapters.TextResultsAsrAdapter) -> None:
    with pytest.raises(LanguageNotFoundError, match="xx, yy"):
        model.recognize(_waveform(), sample_rate=16_000, language="qq")


def test_one_session_for_all_languages(model: onnx_asr.adapters.TextResultsAsrAdapter) -> None:
    assert isinstance(model.asr, Wav2Vec2Adapters)
    asr = model.asr
    assert isinstance(asr._model, rt.InferenceSession)
    model.recognize(_waveform(), sample_rate=16_000, language="yy")
    assert asr._model is model.asr._model


def test_batch_of_two(model: onnx_asr.adapters.TextResultsAsrAdapter) -> None:
    waveforms = [_waveform()[0], _waveform()[0]]
    assert model.recognize(waveforms, sample_rate=16_000) == ["a", "a"]


def test_missing_default_language_raises_on_use(model_dir: Path) -> None:
    config = json.loads((model_dir / "config.json").read_text())
    del config["default_language"]
    (model_dir / "config.json").write_text(json.dumps(config))

    model = onnx_asr.load_model("wav2vec2-adapters", model_dir)
    with pytest.raises(LanguageNotSpecifiedError, match="xx, yy"):
        model.recognize(_waveform(), sample_rate=16_000)


def test_single_pack_becomes_the_default(model_dir: Path) -> None:
    config = json.loads((model_dir / "config.json").read_text())
    del config["default_language"]
    (model_dir / "config.json").write_text(json.dumps(config))
    (model_dir / "adapters" / "xx.npz").unlink()

    model = onnx_asr.load_model("wav2vec2-adapters", model_dir)
    assert model.recognize(_waveform(), sample_rate=16_000) == "b"


def test_incomplete_pack_is_rejected(model_dir: Path) -> None:
    _savez(model_dir / "adapters" / "yy.npz", {"lm_head.bias": np.zeros(len(YY_VOCAB), dtype=np.float32)})
    model = onnx_asr.load_model("wav2vec2-adapters", model_dir)
    with pytest.raises(MissingAdapterInputsError, match=r"adapter\.0\.scale"):
        model.recognize(_waveform(), sample_rate=16_000, language="yy")


def test_padded_vocab_is_trimmed(model_dir: Path) -> None:
    """A head padded past the true vocabulary must not emit the padding tokens."""
    padded = len(YY_VOCAB) + 3
    weight = np.zeros((HIDDEN, padded), dtype=np.float32)
    bias = np.zeros(padded, dtype=np.float32)
    bias[padded - 1] = 100.0  # padding token wins unless it is trimmed away
    bias[YY_VOCAB.index("c")] = 10.0
    _savez(
        model_dir / "adapters" / "yy.npz",
        {
            "adapter.0.scale": np.ones(1, dtype=np.float32),
            "lm_head.weight": weight,
            "lm_head.bias": bias,
            "vocab_size": np.array(len(YY_VOCAB), dtype=np.int64),
        },
    )
    model = onnx_asr.load_model("wav2vec2-adapters", model_dir)
    assert model.recognize(_waveform(), sample_rate=16_000, language="yy") == "c"


def test_omitting_language_returns_to_the_default(model: onnx_asr.adapters.TextResultsAsrAdapter) -> None:
    """A previous call must not decide the language of the next one."""
    model.recognize(_waveform(), sample_rate=16_000, language="yy")
    assert model.recognize(_waveform(), sample_rate=16_000) == "a"
