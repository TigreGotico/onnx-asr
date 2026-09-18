"""LogMelSpectrogram feature extractor for w2v-BERT 2.0 models.

The features match `transformers.SeamlessM4TFeatureExtractor`, which is the
feature extractor of `facebook/w2v-bert-2.0`: Kaldi-style 80 bin log mel
filterbanks with a Povey window, followed by per mel bin mean-variance
normalization and 2 frame stacking.
"""

import numpy as np

from preprocessors.fbanks import melscale_fbanks

sample_rate = 16_000
n_fft = 512
win_length = 400
hop_length = 160
num_mel_bins = 80

snip_edges = True
dither = 0.0
remove_dc_offset = True
preemphasis_coefficient = 0.97

low_freq = 20
high_freq = 0

stride = 2
mel_floor = 1.192092955078125e-07
norm_eps = 1e-7

w2vbert_mel_banks = melscale_fbanks(
    n_fft // 2 + 1, low_freq, high_freq, num_mel_bins, sample_rate, mel_scale="kaldi"
).astype(np.float32)
