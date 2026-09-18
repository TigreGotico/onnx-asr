"""Tests for the paraformer model family with tiny generated ONNX graphs."""

import json
from pathlib import Path

import numpy as np
import numpy.typing as npt
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from onnx_asr.asr import Preprocessor, TimestampedResult
from onnx_asr.models.paraformer import Paraformer

VOCAB = 10
EOS_ID = 2
LFR_M, LFR_N = 7, 6
FEAT_DIM = 80

VOCAB_TXT = {
    "<blank>": 0,
    "<s>": 1,
    "</s>": EOS_ID,
    "he@@": 3,
    "llo": 4,
    "world": 5,
    "中": 6,
    "文": 7,
    "a@@": 8,
    "b@@": 9,
}

# he@@ llo world 中 文, then </s>, then a token that must never be decoded.
TOKEN_IDS = [3, 4, 5, 6, 7, EOS_ID, 5]


def _const(name: str, array: np.ndarray) -> onnx.NodeProto:
    return helper.make_node("Constant", [], [name], value=numpy_helper.from_array(array, name + "_value"))


def _make_model(path: Path) -> None:
    """Emit a fixed token sequence and echo `speech_lengths` back as `token_num`.

    `token_num` comes from the input, so a runtime that ignores the CIF token count
    fails the truncation tests instead of silently passing.
    """
    nodes = [
        _const("eye", np.eye(VOCAB, dtype=np.float32)),
        _const("token_ids", np.array(TOKEN_IDS, dtype=np.int64)),
        _const("zero", np.array(0.0, dtype=np.float32)),
        _const("batch_axis", np.array([0], dtype=np.int64)),
        helper.make_node("Gather", ["eye", "token_ids"], ["rows"], axis=0),
        helper.make_node("Unsqueeze", ["rows", "batch_axis"], ["logits_clean"]),
        # touch the features so a wrong feature shape is still an error
        helper.make_node("ReduceSum", ["speech"], ["speech_sum"], keepdims=0),
        helper.make_node("Mul", ["speech_sum", "zero"], ["speech_bias"]),
        helper.make_node("Add", ["logits_clean", "speech_bias"], ["logits"]),
        helper.make_node("Identity", ["speech_lengths"], ["token_num"]),
    ]
    graph = helper.make_graph(
        nodes,
        "paraformer",
        [
            helper.make_tensor_value_info("speech", TensorProto.FLOAT, [1, "time", LFR_M * FEAT_DIM]),
            helper.make_tensor_value_info("speech_lengths", TensorProto.INT32, [1]),
        ],
        [
            helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, len(TOKEN_IDS), VOCAB]),
            helper.make_tensor_value_info("token_num", TensorProto.INT32, [1]),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


SEEN_WAVEFORMS: list[np.ndarray] = []
FRAMES = 36


def _identity_preprocessor(_name: str) -> Preprocessor:
    def preprocessor(
        waveforms: npt.NDArray[np.float32], waveforms_lens: npt.NDArray[np.int64]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
        SEEN_WAVEFORMS.append(waveforms)
        batch = waveforms_lens.shape[0]
        features = np.zeros((batch, FRAMES, FEAT_DIM), dtype=np.float32)
        return features, np.full(batch, FRAMES, dtype=np.int64)

    return preprocessor


@pytest.fixture
def model(tmp_path: Path) -> Paraformer:
    _make_model(tmp_path / "model.onnx")

    with (tmp_path / "vocab.txt").open("wt", encoding="utf-8") as f:
        for token, id in VOCAB_TXT.items():
            f.write(f"{token} {id}\n")
    config = {
        "model_type": "paraformer",
        "preprocessor": "identity",
        "subsampling_factor": LFR_N,
        "waveform_scale": 1 << 15,
        "lfr_window_size": LFR_M,
        "lfr_window_shift": LFR_N,
        "neg_mean": np.arange(LFR_M * FEAT_DIM, dtype=np.float32).tolist(),
        "inv_stddev": np.full(LFR_M * FEAT_DIM, 0.5, dtype=np.float32).tolist(),
    }
    with (tmp_path / "config.json").open("wt", encoding="utf-8") as f:
        json.dump(config, f)

    files = {
        "model": tmp_path / "model.onnx",
        "vocab": tmp_path / "vocab.txt",
        "config": tmp_path / "config.json",
    }
    return Paraformer(files, _identity_preprocessor, {})


def _recognize(model: Paraformer, samples: int = 16_000, **kwargs: object) -> TimestampedResult:
    waveform = np.zeros((1, samples), dtype=np.float32)
    (result,) = model.recognize_batch(waveform, np.array([samples], dtype=np.int64), **kwargs)
    return result


def test_the_waveform_is_scaled_to_the_int16_range_before_the_fbank(model: Paraformer) -> None:
    # FunASR takes the fbank of an int16 scaled waveform. The log floor and the CMVN
    # statistics both assume that scale, so a unit-scale waveform is wrong.
    SEEN_WAVEFORMS.clear()
    waveform = np.full((1, 16_000), 0.5, dtype=np.float32)
    list(model.recognize_batch(waveform, np.array([16_000], dtype=np.int64)))
    assert len(SEEN_WAVEFORMS) == 1
    np.testing.assert_allclose(SEEN_WAVEFORMS[0], 0.5 * 32768.0)


def test_decoding_stops_at_the_end_of_sequence_token(model: Paraformer) -> None:
    result = _recognize(model)
    assert result.tokens == ["he@@", "llo", "world", "中", "文"]


def test_subwords_merge_and_cjk_takes_no_spaces(model: Paraformer) -> None:
    assert _recognize(model).text == "hello world 中文"


def test_paraformer_has_no_timestamps(model: Paraformer) -> None:
    assert _recognize(model).timestamps is None


def test_logprobs_are_returned_when_asked(model: Paraformer) -> None:
    result = _recognize(model, need_logprobs=True)
    assert result.logprobs is not None
    assert len(result.logprobs) == 5
    assert all(value < 0 for value in result.logprobs)


@pytest.mark.parametrize(
    ("token_num", "expected"),
    [(1, ["he@@"]), (3, ["he@@", "llo", "world"]), (5, ["he@@", "llo", "world", "中", "文"])],
)
def test_token_num_truncates_the_argmax(model: Paraformer, token_num: int, expected: list[str]) -> None:
    # The tiny graph returns speech_lengths as token_num, and the LFR turns a frame
    # count into ceil(frames / 6), so pick the frame count that gives the token count.
    logits = np.zeros((1, len(TOKEN_IDS), VOCAB), dtype=np.float32)
    logits[0, np.arange(len(TOKEN_IDS)), TOKEN_IDS] = 1.0
    ((tokens, indices, _logprobs),) = model._decoding(logits, np.array([token_num], dtype=np.int64))
    assert [VOCAB_TXT_INV[i] for i in tokens] == expected
    assert indices is None


VOCAB_TXT_INV = {id: token for token, id in VOCAB_TXT.items()}


def test_a_token_num_past_the_logits_is_clipped(model: Paraformer) -> None:
    # A quantized graph can return a token count larger than the logits it produced.
    speech = np.zeros((1, 4, LFR_M * FEAT_DIM), dtype=np.float32)
    logits, token_num = model._encode(speech, np.array([len(TOKEN_IDS) + 100], dtype=np.int64))
    assert token_num.tolist() == [len(TOKEN_IDS)]
    ((tokens, _, _),) = model._decoding(logits, token_num)
    assert list(tokens) == TOKEN_IDS[:5]


def test_zero_token_num_gives_empty_text(model: Paraformer) -> None:
    logits = np.zeros((1, len(TOKEN_IDS), VOCAB), dtype=np.float32)
    ((tokens, _, _),) = model._decoding(logits, np.array([0], dtype=np.int64))
    assert list(tokens) == []
    assert model._decode_tokens([], None, None).text == ""


def _reference_lfr(features: np.ndarray, m: int = LFR_M, n: int = LFR_N) -> np.ndarray:
    """The FunASR `apply_lfr`, written straight from the FunASR source."""
    time = features.shape[0]
    time_lfr = int(np.ceil(time / n))
    padded = np.concatenate([np.repeat(features[:1], (m - 1) // 2, axis=0), features])
    total = padded.shape[0]
    out = []
    for i in range(time_lfr):
        if m <= total - i * n:
            out.append(padded[i * n : i * n + m].reshape(-1))
        else:
            frame = padded[i * n :].reshape(-1)
            frame = np.concatenate([frame] + [padded[-1]] * (m - (total - i * n)))
            out.append(frame)
    return np.stack(out)


@pytest.mark.parametrize("frames", [1, 5, 6, 7, 12, 13, 20])
def test_the_lfr_stack_matches_funasr(model: Paraformer, frames: int) -> None:
    rng = np.random.default_rng(frames)
    features = rng.standard_normal((1, frames, FEAT_DIM), dtype=np.float32)
    speech, speech_lens = model._lfr_cmvn(features, np.array([frames], dtype=np.int64))
    expected = (_reference_lfr(features[0]) + model._cmvn_add) * model._cmvn_mul
    np.testing.assert_allclose(speech[0], expected, rtol=1e-6, atol=1e-6)
    assert speech_lens.tolist() == [int(np.ceil(frames / LFR_N))]
    assert speech.shape[1] == int(np.ceil(frames / LFR_N))


def test_batch_padding_does_not_change_the_short_item(model: Paraformer) -> None:
    rng = np.random.default_rng(0)
    long, short = 20, 9
    features = rng.standard_normal((2, long, FEAT_DIM), dtype=np.float32)
    features[1, short:] = 1e6  # padding a runtime must not look at
    batched, lens = model._lfr_cmvn(features, np.array([long, short], dtype=np.int64))
    single, _ = model._lfr_cmvn(features[1:, :short].copy(), np.array([short], dtype=np.int64))
    assert lens.tolist() == [4, 2]
    np.testing.assert_allclose(batched[1, : single.shape[1]], single[0], rtol=1e-6, atol=1e-6)


def test_model_files() -> None:
    assert Paraformer._get_model_files() == {
        "model": "model.onnx",
        "vocab": "vocab.txt",
        "config": "config.json",
    }
    assert Paraformer._get_model_files("int8")["model"] == "model?int8.onnx"
