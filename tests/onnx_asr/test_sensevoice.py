"""Tests for the sensevoice model family with tiny generated ONNX graphs."""

import json
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from onnx_asr.models.sensevoice import SenseVoice

BLANK_ID = 0
VOCAB = 10
EVENT_ID = 3
EMOTION_ID = 4

VOCAB_TXT = {
    "<blk>": BLANK_ID,
    "▁hello": 1,
    "▁world": 2,
    "<|Speech|>": EVENT_ID,
    "<|NEUTRAL|>": EMOTION_ID,
    "<|en|>": 5,
    "<|zh|>": 6,
    "<|woitn|>": 7,
    "<|withitn|>": 8,
    "▁there": 9,
}
LANGUAGES = {"en": 5, "zh": 6}
TEXTNORM = {"woitn": 7, "withitn": 8}

# Speech frames that collapse to [hello, world].
SPEECH_IDS = [1, 1, BLANK_ID, 2]


def _const(name: str, array: np.ndarray) -> onnx.NodeProto:
    return helper.make_node("Constant", [], [name], value=numpy_helper.from_array(array, name + "_value"))


def _make_model(path: Path) -> None:
    """Echo the prompt ids back as the first four CTC frames, then a fixed transcript.

    The language and textnorm graph inputs become output tokens, so a runtime that
    fails to forward them shows up as a wrong rich token rather than silently passing.
    """
    nodes = [
        _const("eye", np.eye(VOCAB, dtype=np.float32)),
        _const("event_emotion", np.array([EVENT_ID, EMOTION_ID], dtype=np.int64)),
        _const("speech_ids", np.array(SPEECH_IDS, dtype=np.int64)),
        _const("zero", np.array(0.0, dtype=np.float32)),
        _const("batch_axis", np.array([0], dtype=np.int64)),
        helper.make_node("Concat", ["language", "event_emotion", "textnorm", "speech_ids"], ["ids"], axis=0),
        helper.make_node("Gather", ["eye", "ids"], ["frames"], axis=0),
        helper.make_node("Unsqueeze", ["frames", "batch_axis"], ["logprobs_clean"]),
        # touch the features so a wrong feature shape is still an error
        helper.make_node("ReduceSum", ["features"], ["features_sum"], keepdims=0),
        helper.make_node("Mul", ["features_sum", "zero"], ["features_bias"]),
        helper.make_node("Add", ["logprobs_clean", "features_bias"], ["logprobs"]),
        helper.make_node("Shape", ["ids"], ["logprobs_lens"]),
    ]
    graph = helper.make_graph(
        nodes,
        "sensevoice",
        [
            helper.make_tensor_value_info("features", TensorProto.FLOAT, [1, "time", 80]),
            helper.make_tensor_value_info("features_lens", TensorProto.INT64, [1]),
            helper.make_tensor_value_info("language", TensorProto.INT64, [1]),
            helper.make_tensor_value_info("textnorm", TensorProto.INT64, [1]),
        ],
        [
            helper.make_tensor_value_info("logprobs", TensorProto.FLOAT, [1, "time_out", VOCAB]),
            helper.make_tensor_value_info("logprobs_lens", TensorProto.INT64, [1]),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


SEEN_WAVEFORMS: list[np.ndarray] = []


def _identity_preprocessor(_name: str):
    def preprocessor(waveforms, waveforms_lens):
        SEEN_WAVEFORMS.append(waveforms)
        features = np.zeros((waveforms.shape[0], 20, 80), dtype=np.float32)
        return features, waveforms_lens

    return preprocessor


@pytest.fixture
def model(tmp_path: Path) -> SenseVoice:
    _make_model(tmp_path / "model.onnx")

    with (tmp_path / "vocab.txt").open("wt", encoding="utf-8") as f:
        for token, id in VOCAB_TXT.items():
            f.write(f"{token} {id}\n")
    config = {
        "model_type": "sensevoice",
        "preprocessor": "identity",
        "subsampling_factor": 6,
        "languages": LANGUAGES,
        "textnorm": TEXTNORM,
        "default_language": "en",
        "default_textnorm": "woitn",
    }
    with (tmp_path / "config.json").open("wt", encoding="utf-8") as f:
        json.dump(config, f)

    files = {
        "model": tmp_path / "model.onnx",
        "vocab": tmp_path / "vocab.txt",
        "config": tmp_path / "config.json",
    }
    return SenseVoice(files, _identity_preprocessor, {})


def test_the_waveform_is_scaled_to_the_int16_range_before_the_fbank(model: SenseVoice) -> None:
    # FunASR takes the fbank of an int16 scaled waveform. The log floor and the CMVN
    # baked into the graph both assume that scale, so a unit-scale waveform is wrong.
    SEEN_WAVEFORMS.clear()
    waveform = np.full((1, 16_000), 0.5, dtype=np.float32)
    list(model.recognize_batch(waveform, np.array([16_000], dtype=np.int64)))
    assert len(SEEN_WAVEFORMS) == 1
    assert SEEN_WAVEFORMS[0].dtype == np.float32
    np.testing.assert_allclose(SEEN_WAVEFORMS[0], 0.5 * 32768.0)


def _recognize(model: SenseVoice, **kwargs: object):
    waveform = np.zeros((1, 16_000), dtype=np.float32)
    (result,) = model.recognize_batch(waveform, np.array([16_000], dtype=np.int64), **kwargs)
    return result


def test_rich_tokens_are_kept_out_of_the_text(model: SenseVoice) -> None:
    assert _recognize(model).text == "hello world"


def test_rich_tokens_stay_in_the_token_list(model: SenseVoice) -> None:
    result = _recognize(model)
    assert result.tokens is not None
    assert result.tokens[:4] == ["<|en|>", "<|Speech|>", "<|NEUTRAL|>", "<|woitn|>"]


def test_language_reaches_the_graph(model: SenseVoice) -> None:
    result = _recognize(model, language="zh")
    assert result.tokens is not None
    assert result.tokens[0] == "<|zh|>"


def test_default_language_is_used_when_none_is_given(model: SenseVoice) -> None:
    result = _recognize(model)
    assert result.tokens is not None
    assert result.tokens[0] == "<|en|>"


@pytest.mark.parametrize(("use_itn", "expected"), [(True, "<|withitn|>"), (False, "<|woitn|>")])
def test_use_itn_selects_the_textnorm_prompt(model: SenseVoice, use_itn: bool, expected: str) -> None:
    result = _recognize(model, use_itn=use_itn)
    assert result.tokens is not None
    assert result.tokens[3] == expected


def test_unknown_language_is_rejected(model: SenseVoice) -> None:
    with pytest.raises(ValueError, match="Unknown language"):
        _recognize(model, language="klingon")


def test_timestamps_use_the_lfr_subsampling_factor(model: SenseVoice) -> None:
    assert model._subsampling_factor == 6
    result = _recognize(model)
    assert result.timestamps is not None
    # four prompt frames, then the first speech token
    assert result.timestamps[4] == pytest.approx(0.01 * 6 * 4)


def test_model_files() -> None:
    assert SenseVoice._get_model_files() == {
        "model": "model.onnx",
        "vocab": "vocab.txt",
        "config": "config.json",
    }
    assert SenseVoice._get_model_files("int8")["model"] == "model?int8.onnx"
