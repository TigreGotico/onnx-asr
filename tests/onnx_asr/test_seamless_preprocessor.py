"""The SeamlessM4T front end: NumPy and ONNX paths agree, frames are counted the reference's way."""

import numpy as np
import pytest

from onnx_asr.preprocessors.numpy_preprocessor import SeamlessPreprocessorNumpy
from onnx_asr.preprocessors.preprocessor import OnnxPreprocessor

WIN, HOP, STRIDE = 400, 160, 2


def _tone(samples: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(samples) / 16_000
    return (0.3 * np.sin(2 * np.pi * 220 * t) + 0.1 * rng.standard_normal(samples)).astype(np.float32)


@pytest.mark.parametrize("samples", [16_000, 16_000 + HOP, 16_000 + 2 * HOP])
def test_numpy_and_onnx_agree(samples: int) -> None:
    waveforms = _tone(samples)[None, :]
    lens = np.array([samples], dtype=np.int64)
    np_features, np_lens = SeamlessPreprocessorNumpy("seamless")(waveforms, lens)
    onnx_features, onnx_lens = OnnxPreprocessor("seamless", {})(waveforms, lens)
    assert np_features.shape == onnx_features.shape
    assert np_features.shape[-1] == 80 * STRIDE
    assert np.array_equal(np_lens, onnx_lens)
    # the ONNX path takes its DFT as a matmul and its sums in float64 where the
    # reference uses pocketfft and float32; the two agree to a few float32
    # units, not bit for bit. The NumPy path is the one that matches
    # transformers exactly.
    np.testing.assert_allclose(np_features, onnx_features, atol=2e-4, rtol=0)


def test_frame_count_and_odd_frame_padding() -> None:
    # 16_000 + HOP samples give 99 frames: an odd count is padded to 100 and
    # stacked into 50 steps, but only 49 of them are counted as valid, because
    # the reference masks a stacked step by its second half
    samples = 16_000 + HOP
    frames = 1 + (samples - WIN) // HOP
    assert frames % 2 == 1
    features, lens = SeamlessPreprocessorNumpy("seamless")(_tone(samples)[None, :], np.array([samples]))
    assert features.shape[1] == (frames + 1) // STRIDE
    assert lens[0] == frames // STRIDE
    assert np.all(features[0, -1, 80:] == 0), "the padded half of the last step is zero"


def test_batch_rows_are_normalised_on_their_own_frames() -> None:
    long, short = 24_000, 16_000
    waveforms = np.zeros((2, long), dtype=np.float32)
    waveforms[0] = _tone(long, 1)
    waveforms[1, :short] = _tone(short, 2)
    lens = np.array([long, short], dtype=np.int64)
    batch, batch_lens = SeamlessPreprocessorNumpy("seamless")(waveforms, lens)
    alone, alone_lens = SeamlessPreprocessorNumpy("seamless")(waveforms[1:, :short], lens[1:])
    assert batch_lens[1] == alone_lens[0]
    n = alone.shape[1]
    assert np.array_equal(batch[1, :n], alone[0]), "a padded row must see only its own frames"
    assert np.all(batch[1, n:] == 0)
