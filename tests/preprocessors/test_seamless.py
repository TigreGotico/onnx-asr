"""The seamless preprocessor against torchaudio's Kaldi fbank.

transformers' SeamlessM4TFeatureExtractor is a Kaldi fbank (25 ms Povey
window, 10 ms hop, no edge frames, DC removal, pre-emphasis 0.97, 80 mel bins
between 20 Hz and 8 kHz) followed by per-bin normalisation with the sample
variance and pairs of frames stacked. torchaudio computes the same fbank in
float32, so it is the independent reference here, to a tolerance; the
bit-for-bit comparison with transformers is not run in CI, which has no
transformers.
"""

import numpy as np
import pytest
import torch
import torchaudio

from onnx_asr.preprocessors.numpy_preprocessor import SeamlessPreprocessorNumpy
from onnx_asr.preprocessors.preprocessor import OnnxPreprocessor
from onnx_asr.utils import pad_list
from preprocessors import seamless


def preprocessor_origin(waveforms, lens):
    features, feature_lens = [], []
    for waveform, n in zip(waveforms, lens, strict=True):
        fbank = torchaudio.compliance.kaldi.fbank(
            torch.from_numpy(waveform[:n])[None, :] * 32768.0,
            num_mel_bins=seamless.num_mel_bins,
            frame_length=25,
            frame_shift=10,
            dither=0.0,
            energy_floor=0.0,
            preemphasis_coefficient=seamless.preemphasis_coefficient,
            remove_dc_offset=True,
            window_type="povey",
            low_freq=seamless.low_freq,
            high_freq=seamless.high_freq,
            snip_edges=True,
            use_energy=False,
            sample_frequency=seamless.sample_rate,
        ).numpy()
        fbank = (fbank - fbank.mean(0)) / np.sqrt(fbank.var(0, ddof=1) + seamless.norm_eps)
        frames = fbank.shape[0]
        padded = np.pad(fbank, ((0, frames % seamless.stride), (0, 0)))
        features.append(padded.reshape(-1, seamless.num_mel_bins * seamless.stride))
        feature_lens.append(frames // seamless.stride)
    max_len = max(f.shape[0] for f in features)
    return np.stack([np.pad(f, ((0, max_len - f.shape[0]), (0, 0))) for f in features]), np.array(feature_lens)


@pytest.mark.parametrize("path", ["numpy", "onnx"])
def test_seamless_preprocessor(waveforms: list[np.ndarray], path: str):
    padded, lens = pad_list(waveforms)
    expected, expected_lens = preprocessor_origin(waveforms, lens)
    preprocessor = SeamlessPreprocessorNumpy("seamless") if path == "numpy" else OnnxPreprocessor("seamless", {})
    actual, actual_lens = preprocessor(padded, lens)

    assert np.array_equal(actual_lens, expected_lens)
    assert actual.shape[-1] == seamless.num_mel_bins * seamless.stride
    for i, n in enumerate(expected_lens):
        np.testing.assert_allclose(actual[i, :n], expected[i, :n], atol=1e-3, rtol=0)
