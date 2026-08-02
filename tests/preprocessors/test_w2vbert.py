import numpy as np
import pytest

from onnx_asr.preprocessors.numpy_preprocessor import W2vBertPreprocessorNumpy
from onnx_asr.utils import pad_list
from preprocessors import w2vbert

from .test_kaldi import pad_features, preprocessor_origin


def preprocessor_reference(waveforms: np.ndarray, lens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reference w2v-BERT features: Kaldi fbank, per mel bin CMVN, 2 frame stacking."""
    features, features_lens = preprocessor_origin(
        waveforms, lens, snip_edges=w2vbert.snip_edges, high_freq=w2vbert.high_freq, window_type="povey"
    )

    results = []
    for feature, length in zip(features, features_lens, strict=True):
        feature = feature[:length]
        feature = (feature - feature.mean(0)) / np.sqrt(feature.var(0, ddof=1) + w2vbert.norm_eps)
        if feature.shape[0] % w2vbert.stride != 0:
            feature = np.pad(feature, ((0, w2vbert.stride - feature.shape[0] % w2vbert.stride), (0, 0)))
        results.append(feature.reshape(feature.shape[0] // w2vbert.stride, -1))

    return pad_features(results)[0], features_lens // w2vbert.stride


@pytest.fixture(scope="module")
def preprocessor() -> W2vBertPreprocessorNumpy:
    return W2vBertPreprocessorNumpy("w2vbert")


def test_w2vbert_preprocessor(preprocessor: W2vBertPreprocessorNumpy, waveforms: list[np.ndarray]) -> None:
    padded, lens = pad_list(waveforms)
    expected, expected_lens = preprocessor_reference(padded, lens)
    actual, actual_lens = preprocessor(padded, lens)

    assert actual.dtype == np.float32
    assert actual.shape[-1] == w2vbert.num_mel_bins * w2vbert.stride
    np.testing.assert_equal(actual_lens, expected_lens)

    for i, length in enumerate(actual_lens):
        np.testing.assert_allclose(actual[i, :length], expected[i, :length], atol=1e-4, rtol=1e-4)


def test_w2vbert_preprocessor_pads_last_frame(preprocessor: W2vBertPreprocessorNumpy) -> None:
    rng = np.random.default_rng(0)
    # 400 + 4 * 160 samples give 5 frames, an odd number that must be padded to 3 stacked frames.
    waveform = (rng.random((1, 400 + 4 * 160), dtype=np.float32) * 2 - 1).astype(np.float32)
    features, features_lens = preprocessor(waveform, np.array([waveform.shape[-1]], dtype=np.int64))

    assert features.shape == (1, 3, 160)
    np.testing.assert_equal(features_lens, [2])
