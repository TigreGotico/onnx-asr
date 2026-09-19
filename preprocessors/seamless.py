"""Fbank feature extractor for w2v-BERT 2.0 models (SeamlessM4T front end).

The HuggingFace ``SeamlessM4TFeatureExtractor`` computes 80 Kaldi-style log mel
bins per 25 ms frame (hop 10 ms, no edge padding, Povey window, DC removal and
pre-emphasis per frame), normalises every mel bin to zero mean and unit sample
variance over the frames of the utterance, and stacks pairs of frames into
160-dimensional vectors. A pre-built w2v-BERT ONNX graph expects exactly these
``input_features``.

Two details decide whether a model quantised to int8 decodes the same text as
the PyTorch pipeline: the spectrogram is rounded to complex64 before the power
is taken, and the per-bin variance is the sample variance (ddof=1). Both are
reproduced here.
"""

import numpy as np
import numpy.typing as npt
from onnxscript import DOUBLE, FLOAT, INT64, script
from onnxscript import opset17 as op

sample_rate = 16_000
n_fft = 512
win_length = 400
hop_length = 160
num_mel_bins = 80
stride = 2

preemphasis_coefficient = 0.97
mel_floor = 1.192092955078125e-07
norm_eps = 1e-7
low_freq = 20
high_freq = sample_rate // 2


def _hertz_to_mel(freq: float | npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return 1127.0 * np.log(1.0 + (freq / 700.0))


def seamless_mel_banks() -> npt.NDArray[np.float64]:
    """Build the mel filter bank the way ``transformers.audio_utils.mel_filter_bank`` does.

    Kaldi mel scale, filters triangularised in mel space, no normalisation.
    Kept in float64: the model card's features are float64 up to the log.
    """
    num_frequency_bins = n_fft // 2 + 1
    mel_freqs = np.linspace(_hertz_to_mel(low_freq), _hertz_to_mel(high_freq), num_mel_bins + 2)
    fft_bin_width = sample_rate / ((num_frequency_bins - 1) * 2)
    fft_freqs = _hertz_to_mel(fft_bin_width * np.arange(num_frequency_bins))
    filter_diff = np.diff(mel_freqs)
    slopes = np.expand_dims(mel_freqs, 0) - np.expand_dims(fft_freqs, 1)
    down_slopes = -slopes[:, :-2] / filter_diff[:-1]
    up_slopes = slopes[:, 2:] / filter_diff[1:]
    return np.maximum(np.zeros(1), np.minimum(down_slopes, up_slopes))


def povey_window() -> npt.NDArray[np.float64]:
    """Build the Povey window: Hann to the power 0.85, symmetric, float64."""
    return np.power(np.hanning(win_length), 0.85)


seamless_mel_banks_f64 = seamless_mel_banks()
seamless_window_f64 = povey_window()

_dft_indices = np.arange(n_fft // 2 + 1)[:, np.newaxis] * np.arange(n_fft)[np.newaxis, :]
_dft_angle = 2 * np.pi * _dft_indices / n_fft
seamless_dft_cos = np.cos(_dft_angle).T.copy()
seamless_dft_sin = (-np.sin(_dft_angle)).T.copy()
mel_floor_f64 = np.array(mel_floor, dtype=np.float64)
scale_f64 = np.array(32768.0, dtype=np.float64)
zero_f64 = np.array(0.0, dtype=np.float64)
one_f64 = np.array(1.0, dtype=np.float64)
norm_eps_f64 = np.array(norm_eps, dtype=np.float64)
preemphasis_f64 = np.array(preemphasis_coefficient, dtype=np.float64)
first_gain_f64 = np.array(1 - preemphasis_coefficient, dtype=np.float64)
zero_1d = np.array([0], dtype=np.int64)
one_1d = np.array([1], dtype=np.int64)
stacked_1d = np.array([num_mel_bins * stride], dtype=np.int64)
shift_pads = np.array([0, 0, 1, 0, 0, -1], dtype=np.int64)
fft_pads = np.array([0, 0, 0, 0, 0, n_fft - win_length], dtype=np.int64)


@script()
def frames_f64(waveforms: DOUBLE["batch_size", "N"]):
    """[batch, N] -> [batch, frames, win_length] without edge padding (snip_edges)."""
    samples = op.Squeeze(op.Shape(waveforms, start=1, end=2))
    num_frames = (samples - win_length) / hop_length + 1
    starts = op.Range(0, num_frames, 1) * hop_length
    indices = op.Unsqueeze(starts, axes=[1]) + op.Unsqueeze(op.Range(0, win_length, 1), axes=[0])
    return op.Gather(waveforms, indices, axis=1)


@script(doc_string="Fbank feature extractor for w2v-BERT 2.0 models (SeamlessM4T front end)")
def SeamlessPreprocessor(
    waveforms: FLOAT["batch_size", "N"], waveforms_lens: INT64["batch_size"]
) -> tuple[FLOAT["batch_size", "T", num_mel_bins * stride], INT64["batch_size"]]:
    scaled = op.Cast(waveforms, to=DOUBLE.dtype) * scale_f64
    frames = frames_f64(scaled)
    frames = frames - op.ReduceMean(frames, axes=[-1])
    # pre-emphasis per frame: x[0] *= 1 - c, x[n] -= c * x[n - 1]
    shifted = op.Pad(frames, shift_pads, mode="edge")
    emphasised = frames - preemphasis_f64 * shifted
    first = frames[:, :, :1] * first_gain_f64
    frames = op.Concat(first, emphasised[:, :, 1:], axis=2) * seamless_window_f64
    padded = op.Pad(frames, fft_pads, mode="constant")
    # the reference rounds the spectrum to complex64 before the power is taken
    real = op.Cast(op.Cast(op.MatMul(padded, seamless_dft_cos), to=FLOAT.dtype), to=DOUBLE.dtype)
    imag = op.Cast(op.Cast(op.MatMul(padded, seamless_dft_sin), to=FLOAT.dtype), to=DOUBLE.dtype)
    magnitude = op.Sqrt(real * real + imag * imag)
    power = magnitude * magnitude
    mel = op.Max(op.MatMul(power, seamless_mel_banks_f64), mel_floor_f64)
    log_mel = op.Cast(op.Log(mel), to=FLOAT.dtype)

    # per mel bin over the valid frames: zero mean, unit sample variance. The
    # reference does this in float32 with NumPy's pairwise summation; the sums
    # here are taken in float64 and rounded once, which is the same value to
    # within one float32 unit. The NumPy preprocessor is the bit-exact path.
    frame_lens = (waveforms_lens - win_length) / hop_length + 1
    frame_index = op.Unsqueeze(op.Range(0, op.Squeeze(op.Shape(log_mel, start=1, end=2)), 1), axes=[0, 2])
    valid = frame_index < op.Unsqueeze(frame_lens, axes=[1, 2])
    count = op.Cast(op.Unsqueeze(frame_lens, axes=[1, 2]), to=DOUBLE.dtype)
    log_mel_f64 = op.Cast(log_mel, to=DOUBLE.dtype)
    masked = op.Where(valid, log_mel_f64, zero_f64)
    mean = op.ReduceSum(masked, axes=[1]) / count
    centred = op.Where(valid, log_mel_f64 - mean, zero_f64)
    var = op.ReduceSum(centred * centred, axes=[1]) / (count - one_f64)
    normed = op.Cast(op.Where(valid, centred / op.Sqrt(var + norm_eps_f64), zero_f64), to=FLOAT.dtype)

    # pad an odd frame count to a pair, as the reference does, then stack pairs
    num_frames = op.Shape(normed, start=1, end=2)
    odd = num_frames % stride
    tail_pads = op.Concat(zero_1d, zero_1d, zero_1d, zero_1d, odd, zero_1d, axis=0)
    padded_frames = op.Pad(normed, tail_pads, mode="constant")
    shape = op.Concat(zero_1d, (num_frames + odd) / stride, stacked_1d, axis=0)
    features = op.Reshape(padded_frames, shape)
    features_lens = frame_lens / stride
    return features, features_lens
