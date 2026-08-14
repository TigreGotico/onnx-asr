"""Tests for the granite-nar model family with tiny generated ONNX graphs."""

import json
from pathlib import Path

import numpy as np
import numpy.typing as npt
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from onnx_asr.asr import Preprocessor
from onnx_asr.models.granite_nar import GraniteNar, _ctc_collapse, _insertion_slots

HIDDEN = 4
VOCAB = 8
BLANK_ID = 0
MIN_EDIT_LENGTH = 8

# The encoder hypothesis collapses to [1, 2, 3]; the editor keeps 1 and 3 and
# deletes 2, so the transcript is "hello world".
CTC_ARGMAX = [1, 1, BLANK_ID, 2, 2, 2, 3]
HYPOTHESIS = [1, 2, 3]
EDIT_TABLE = {BLANK_ID: BLANK_ID, 1: 1, 2: BLANK_ID, 3: 3}

VOCAB_JSON = {
    "<|endoftext|>": BLANK_ID,
    "Ġhello": 1,
    "Ġthere": 2,
    "Ġworld": 3,
    "a": 4,
    "b": 5,
    "c": 6,
    "<|im_end|>": 7,
}


def _const(name: str, array: np.ndarray) -> onnx.NodeProto:
    return helper.make_node("Constant", [], [name], value=numpy_helper.from_array(array, name + "_value"))


def _save(graph: onnx.GraphProto, path: Path) -> None:
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def _make_encoder(path: Path, audio_length: int) -> None:
    """Emit a fixed CTC hypothesis and zero audio embeddings."""
    ctc_logits = np.zeros((1, len(CTC_ARGMAX), VOCAB), dtype=np.float32)
    ctc_logits[0, np.arange(len(CTC_ARGMAX)), CTC_ARGMAX] = 1.0

    nodes = [
        _const("ctc_logits", ctc_logits),
        _const("audio_embeds", np.zeros((1, audio_length, HIDDEN), dtype=np.float32)),
        _const("audio_embeds_lens", np.array([audio_length], dtype=np.int64)),
        _const("ctc_lens", np.array([len(CTC_ARGMAX)], dtype=np.int64)),
    ]
    graph = helper.make_graph(
        nodes,
        "encoder",
        [helper.make_tensor_value_info("input_features", TensorProto.FLOAT, [1, "samples"])],
        [
            helper.make_tensor_value_info("audio_embeds", TensorProto.FLOAT, [1, "audio_seq", HIDDEN]),
            helper.make_tensor_value_info("ctc_logits", TensorProto.FLOAT, [1, "ctc_seq", VOCAB]),
            helper.make_tensor_value_info("audio_embeds_lens", TensorProto.INT64, [1]),
            helper.make_tensor_value_info("ctc_lens", TensorProto.INT64, [1]),
        ],
    )
    _save(graph, path)


def _make_editor(path: Path) -> None:
    """Rewrite every slot through EDIT_TABLE, so the logits depend on `text_ids`."""
    table = np.zeros((VOCAB, VOCAB), dtype=np.float32)
    for source in range(VOCAB):
        table[source, EDIT_TABLE.get(source, source)] = 1.0

    nodes = [
        _const("table", table),
        # touch audio_embeds so a wrong audio input shape is still an error
        _const("scale", np.array(0.0, dtype=np.float32)),
        helper.make_node("Gather", ["table", "text_ids"], ["gathered"], axis=0),
        helper.make_node("ReduceSum", ["audio_embeds"], ["audio_sum"], keepdims=0),
        helper.make_node("Mul", ["audio_sum", "scale"], ["audio_bias"]),
        helper.make_node("Add", ["gathered", "audio_bias"], ["logits"]),
    ]
    graph = helper.make_graph(
        nodes,
        "editor",
        [
            helper.make_tensor_value_info("audio_embeds", TensorProto.FLOAT, [1, "audio_seq", HIDDEN]),
            helper.make_tensor_value_info("text_ids", TensorProto.INT64, [1, "text_seq"]),
        ],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, "text_seq", VOCAB])],
    )
    _save(graph, path)


def _identity_preprocessor(_name: str) -> Preprocessor:
    def preprocessor(
        waveforms: npt.NDArray[np.float32], waveforms_lens: npt.NDArray[np.int64]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
        return waveforms, waveforms_lens

    return preprocessor


AUDIO_LENGTH = 5


@pytest.fixture
def model(tmp_path: Path) -> GraniteNar:
    _make_encoder(tmp_path / "encoder.onnx", AUDIO_LENGTH)
    _make_editor(tmp_path / "editor.onnx")

    with (tmp_path / "vocab.json").open("wt", encoding="utf-8") as f:
        json.dump(VOCAB_JSON, f)
    config = {
        "model_type": "granite-nar",
        "preprocessor": "identity",
        "blank_token_id": BLANK_ID,
        "min_edit_sequence_length": MIN_EDIT_LENGTH,
    }
    with (tmp_path / "config.json").open("wt", encoding="utf-8") as f:
        json.dump(config, f)

    files = {name: tmp_path / f"{name}.onnx" for name in ("encoder", "editor")} | {
        "vocab": tmp_path / "vocab.json",
        "config": tmp_path / "config.json",
    }
    return GraniteNar(files, _identity_preprocessor, {})


@pytest.mark.parametrize(
    ("token_ids", "expected"),
    [
        ([], []),
        ([BLANK_ID, BLANK_ID], []),
        ([1, 1, BLANK_ID, 2, 2, 2, 3], [1, 2, 3]),
        ([1, BLANK_ID, 1], [1, 1]),
        ([1, 1, 1], [1]),
        ([BLANK_ID, 1, BLANK_ID, 2, BLANK_ID], [1, 2]),
    ],
)
def test_ctc_collapse(token_ids: list[int], expected: list[int]) -> None:
    assert _ctc_collapse(np.array(token_ids, dtype=np.int64), BLANK_ID).tolist() == expected


def test_ctc_collapse_keeps_a_repeat_split_by_a_blank() -> None:
    # the blank is what makes a doubled letter survive; dropping blanks first would lose it
    assert _ctc_collapse(np.array([5, BLANK_ID, 5], dtype=np.int64), BLANK_ID).tolist() == [5, 5]


@pytest.mark.parametrize(
    ("token_ids", "expected"),
    [
        ([1, 2, 3], [BLANK_ID, 1, BLANK_ID, 2, BLANK_ID, 3, BLANK_ID, BLANK_ID]),
        ([], [BLANK_ID] * MIN_EDIT_LENGTH),
        ([1, 2, 3, 4, 5], [BLANK_ID, 1, BLANK_ID, 2, BLANK_ID, 3, BLANK_ID, 4, BLANK_ID, 5, BLANK_ID]),
    ],
)
def test_insertion_slots(token_ids: list[int], expected: list[int]) -> None:
    slots = _insertion_slots(np.array(token_ids, dtype=np.int64), BLANK_ID, MIN_EDIT_LENGTH)
    assert slots.tolist() == expected


def test_insertion_slots_never_shorter_than_the_minimum() -> None:
    for size in range(6):
        slots = _insertion_slots(np.arange(1, size + 1, dtype=np.int64), BLANK_ID, MIN_EDIT_LENGTH)
        assert len(slots) >= MIN_EDIT_LENGTH
        assert len(slots) == max(2 * size + 1, MIN_EDIT_LENGTH)


def test_recognize(model: GraniteNar) -> None:
    waveform = np.zeros((1, 16_000), dtype=np.float32)
    (result,) = model.recognize_batch(waveform, np.array([16_000], dtype=np.int64))
    assert result.text == "hello world"


def test_recognize_batch_processes_each_waveform(model: GraniteNar) -> None:
    waveforms = np.zeros((3, 16_000), dtype=np.float32)
    results = list(model.recognize_batch(waveforms, np.array([16_000, 8_000, 4_000], dtype=np.int64)))
    assert [result.text for result in results] == ["hello world"] * 3


def test_editor_sees_the_interleaved_hypothesis(model: GraniteNar) -> None:
    seen: list[np.ndarray] = []
    edit = model._edit

    def spy(audio_embeds: npt.NDArray[np.float32], text_ids: npt.NDArray[np.int64]) -> npt.NDArray[np.float32]:
        seen.append(text_ids)
        return edit(audio_embeds, text_ids)

    model._edit = spy  # type: ignore[method-assign]
    list(model.recognize_batch(np.zeros((1, 16_000), dtype=np.float32), np.array([16_000], dtype=np.int64)))

    expected = _insertion_slots(np.array(HYPOTHESIS, dtype=np.int64), BLANK_ID, MIN_EDIT_LENGTH)
    assert len(seen) == 1
    assert seen[0].tolist() == expected.tolist()


def test_model_files() -> None:
    assert GraniteNar._get_model_files() == {
        "encoder": "**/encoder.onnx",
        "editor": "**/editor.onnx",
        "vocab": "vocab.json",
    }
    assert GraniteNar._get_model_files("int8")["encoder"] == "**/encoder?int8.onnx"
