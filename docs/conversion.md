# Convert Model to ONNX

Save the model according to the instructions below and add config.json:

```json
{
    "model_type": "nemo-conformer-rnnt", // See "Supported model types"
    "features_size": 80, // Size of preprocessor features for Whisper or Nemo models, supported 80 and 128
    "subsampling_factor": 8, // Subsampling factor - 4 for conformer models and 8 for fastconformer and parakeet models
    "max_tokens_per_step": 10 // Max tokens per step for RNN-T decoder
}
```

Then you can upload the model into Hugging Face and use `load_model` to download it.

## Nvidia NeMo Conformer/FastConformer/Parakeet

Install **NeMo Toolkit**:

```sh
pip install nemo_toolkit['asr']
```

Download model and export to ONNX format:

```py
import nemo.collections.asr as nemo_asr
from pathlib import Path

model = nemo_asr.models.ASRModel.from_pretrained("nvidia/stt_ru_fastconformer_hybrid_large_pc")

# To export Hybrid models with CTC decoder
# model.set_export_config({"decoder_type": "ctc"})

onnx_dir = Path("nemo-onnx")
onnx_dir.mkdir(exist_ok=True)
model.export(str(Path(onnx_dir, "model.onnx")))

with Path(onnx_dir, "vocab.txt").open("wt") as f:
    for i, token in enumerate([*model.tokenizer.vocab, "<blk>"]):
        f.write(f"{token} {i}\n")
```

## GigaChat GigaAM v2/v3

Install **GigaAM**:

```sh
git clone https://github.com/salute-developers/GigaAM.git
pip install ./GigaAM --extra-index-url https://download.pytorch.org/whl/cpu
```

Download model and export to ONNX format:

```py
import gigaam
from pathlib import Path

onnx_dir = "gigaam-onnx"
model_type = "rnnt"  # or "ctc"

model = gigaam.load_model(
    model_type,
    fp16_encoder=False,  # only fp32 tensors
    use_flash=False,  # disable flash attention
)
model.to_onnx(dir_path=onnx_dir)

with Path(onnx_dir, "v2_vocab.txt").open("wt") as f:
    for i, token in enumerate(["\u2581", *(chr(ord("а") + i) for i in range(32)), "<blk>"]):
        f.write(f"{token} {i}\n")
```

## OpenAI Whisper (with `onnxruntime` export)

Read the onnxruntime [instruction](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/transformers/models/whisper/README.md) to convert Whisper to ONNX.

Download model and export with *Beam Search* and *Forced Decoder Input Ids*:

```sh
python3 -m onnxruntime.transformers.models.whisper.convert_to_onnx -m openai/whisper-base --output ./whisper-onnx --use_forced_decoder_ids --optimize_onnx --precision fp32
```

Save the tokenizer config:

```py
from transformers import WhisperTokenizer

processor = WhisperTokenizer.from_pretrained("openai/whisper-base")
processor.save_pretrained("whisper-onnx")
```

## OpenAI Whisper (with `optimum` export)

Export model to ONNX with Hugging Face `optimum-cli`:

```sh
optimum-cli export onnx --model openai/whisper-base ./whisper-onnx
```

## HuggingFace Wav2Vec2 CTC

Install **transformers** and **torch**:

```sh
pip install transformers torch
```

Export the model with feature normalization (`do_normalize`) baked into the graph, so
the model uses the plain `"identity"` preprocessor (raw waveform in, no separate
feature-extractor asset needed):

```py
import json
from pathlib import Path

import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

hf_model_id = "proxectonos/Nos_ASR-wav2vec2-xls-r-300m-gl"
onnx_dir = Path("wav2vec2-onnx")
onnx_dir.mkdir(exist_ok=True)

model = Wav2Vec2ForCTC.from_pretrained(hf_model_id).eval()
processor = Wav2Vec2Processor.from_pretrained(hf_model_id)
do_normalize = bool(processor.feature_extractor.do_normalize)


class NormalizedWav2Vec2(torch.nn.Module):
    def forward(self, input_values, input_lengths):
        if do_normalize:
            mask = torch.arange(input_values.shape[1])[None, :] < input_lengths[:, None]
            count = input_lengths.to(input_values.dtype).clamp(min=1)[:, None]
            mean = (input_values * mask).sum(dim=1, keepdim=True) / count
            var = (((input_values - mean) * mask) ** 2).sum(dim=1, keepdim=True) / count
            input_values = torch.where(mask, (input_values - mean) / torch.sqrt(var + 1e-5), input_values)
        logits = model(input_values).logits
        return torch.nn.functional.log_softmax(logits, dim=-1)


torch.onnx.export(
    NormalizedWav2Vec2(),
    (torch.randn(1, 16000), torch.tensor([16000], dtype=torch.int64)),
    str(onnx_dir / "model.onnx"),
    input_names=["input_values", "input_lengths"],
    output_names=["logprobs"],
    dynamic_axes={
        "input_values": {0: "batch", 1: "time"},
        "input_lengths": {0: "batch"},
        "logprobs": {0: "batch", 1: "frames"},
    },
    opset_version=18,
)

vocab = processor.tokenizer.get_vocab()
pad_token = processor.tokenizer.pad_token
word_delimiter = processor.tokenizer.word_delimiter_token

with (onnx_dir / "vocab.txt").open("wt") as f:
    for token, idx in sorted(vocab.items(), key=lambda kv: kv[1]):
        token = "<blk>" if token == pad_token else "▁" if token == word_delimiter else token
        f.write(f"{token} {idx}\n")

subsampling_factor = 1
for layer in model.wav2vec2.feature_extractor.conv_layers:
    stride = layer.conv.stride
    subsampling_factor *= stride[0] if isinstance(stride, tuple) else stride

with (onnx_dir / "config.json").open("wt") as f:
    json.dump({"model_type": "wav2vec2-ctc", "subsampling_factor": subsampling_factor}, f, indent=2)
```

The pad token becomes `<blk>` (CTC blank, auto-detected by the vocab loader) and the
word-delimiter token (`|`) becomes `▁`, which onnx-asr converts to a literal space
when decoding. `subsampling_factor` is the product of the feature-encoder conv strides
(320 for the standard wav2vec2/XLS-R conv stack) and is only used to scale token
timestamps.

## FunASR SenseVoice

Install **FunASR** and **torch**:

```sh
pip install funasr torch torchaudio onnxscript
```

`FunAudioLLM/SenseVoiceSmall` has a SANM encoder with a CTC head and no decoder. Its
`WavFrontend` does three things: a kaldi fbank, a low frame rate stack (`lfr_m` 7,
`lfr_n` 6) and the `am.mvn` mean-variance statistics. Only the fbank stays outside the
graph, because the onnx-asr `wespeaker` preprocessor already computes exactly that
fbank (hamming window, `snip_edges`, dither 0, preemphasis 0.97, 400/160/512, 80 mel
bins, `log(max(x, eps))`). The LFR stack, the CMVN and the four prompt frames are
folded into the graph, so the graph takes fbank features and two selector inputs:

| Input | Type | Shape |
| --- | --- | --- |
| `features` | f32 | `(batch, time, 80)` |
| `features_lens` | i64 | `(batch,)` |
| `language` | i64 | `(batch,)` |
| `textnorm` | i64 | `(batch,)` |

| Output | Type | Shape |
| --- | --- | --- |
| `logprobs` | f32 | `(batch, ceil(time / 6) + 4, 25055)` |
| `logprobs_lens` | i64 | `(batch,)` |

The FunASR export in `funasr.models.sense_voice.export_meta` is not used, because it
takes the 560 dimension LFR features and leaves the frontend in Python. The wrapper
below puts that frontend in the graph instead.

Download the checkpoint and export:

```py
import json
from pathlib import Path

import sentencepiece
import torch
from funasr import AutoModel
from funasr.frontends.wav_frontend import load_cmvn
from huggingface_hub import snapshot_download

LFR_M, LFR_N = 7, 6
LEFT = (LFR_M - 1) // 2

src_dir = Path(snapshot_download("FunAudioLLM/SenseVoiceSmall"))
onnx_dir = Path("sensevoice-onnx")
onnx_dir.mkdir(exist_ok=True)

model = AutoModel(model=str(src_dir), device="cpu").model.eval()
cmvn = load_cmvn(src_dir / "am.mvn")


class SenseVoiceExport(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = model
        self.register_buffer("cmvn_add", cmvn[0].clone())
        self.register_buffer("cmvn_mul", cmvn[1].clone())

    def lfr(self, features, features_lens):
        batch, time, dim = features.shape

        # Replace the batch padding by each item's last valid frame, which is what the
        # FunASR right padding does, so a batched run and a single clip run agree.
        index = torch.minimum(torch.arange(time), features_lens[:, None] - 1)
        features = torch.gather(features, 1, index[:, :, None].expand(batch, time, dim))

        # Round up with positive operands only. The `-(-x // n)` form rounds up in
        # Python, which floors, but ONNX integer division truncates towards zero, so
        # that form exports as a round *down* and drops the last frame whenever
        # time % 6 is not 0.
        frames = (time + LFR_N - 1) // LFR_N
        # Pad past the last window rather than up to it, so no window is ever clipped
        # by the end of the sequence. `sym_max` never changes the value, which is at
        # least LFR_M; it is what proves the width non-negative to the exporter.
        pad = torch.sym_max(LFR_N * frames + LFR_M - time, 0)
        padded = torch.cat(
            [features[:, :1].expand(batch, LEFT, dim), features, features[:, -1:].expand(batch, pad, dim)], dim=1
        )
        stacked = torch.stack([padded[:, i : i + LFR_N * frames : LFR_N] for i in range(LFR_M)], dim=2)
        return (stacked.reshape(batch, frames, LFR_M * dim) + self.cmvn_add) * self.cmvn_mul

    def forward(self, features, features_lens, language, textnorm):
        speech = self.lfr(features, features_lens)
        speech_lens = (features_lens + LFR_N - 1) // LFR_N

        # Prompt frames, in the FunASR order: language, event, emotion, text norm.
        event_emo = self.model.embed(torch.tensor([[1, 2]])).expand(speech.shape[0], 2, -1)
        speech = torch.cat(
            [self.model.embed(language)[:, None], event_emo, self.model.embed(textnorm)[:, None], speech], dim=1
        )

        encoder_out, _ = self.model.encoder(speech, speech_lens + 4)
        if isinstance(encoder_out, tuple):
            encoder_out = encoder_out[0]

        return torch.log_softmax(self.model.ctc.ctc_lo(encoder_out), dim=-1), speech_lens + 4


batch = torch.export.Dim("batch", min=1)
time = torch.export.Dim("time", min=1)

torch.onnx.export(
    SenseVoiceExport().eval(),
    (torch.randn(2, 200, 80), torch.tensor([137, 200]), torch.tensor([0, 0]), torch.tensor([15, 15])),
    str(onnx_dir / "model.onnx"),
    input_names=["features", "features_lens", "language", "textnorm"],
    output_names=["logprobs", "logprobs_lens"],
    dynamic_shapes={
        "features": {0: batch, 1: time},
        "features_lens": {0: batch},
        "language": {0: batch},
        "textnorm": {0: batch},
    },
    opset_version=18,
    external_data=True,
)

sp = sentencepiece.SentencePieceProcessor(model_file=str(src_dir / "chn_jpn_yue_eng_ko_spectok.bpe.model"))
with (onnx_dir / "vocab.txt").open("wt") as f:
    for i in range(sp.get_piece_size()):
        f.write(f"{'<blk>' if i == 0 else sp.id_to_piece(i)} {i}\n")

with (onnx_dir / "config.json").open("wt") as f:
    json.dump(
        {
            "model_type": "sensevoice",
            "preprocessor": "wespeaker",
            "subsampling_factor": LFR_N,
            "blank_token_id": 0,
            "languages": {"auto": 0, "zh": 3, "en": 4, "yue": 7, "ja": 11, "ko": 12, "nospeech": 13},
            "textnorm": {"withitn": 14, "woitn": 15},
            "default_language": "auto",
            "default_textnorm": "woitn",
            "waveform_scale": 32768,
        },
        f,
        indent=2,
    )
```

Two values in `config.json` are model behaviour and not free choices:

* `waveform_scale` is 32768. The FunASR fbank runs on a waveform in the int16 range,
  and onnx-asr hands preprocessors a waveform in `[-1, 1]`. The scale is not a constant
  offset that the CMVN absorbs, because `log(max(x, eps))` floors the low energy mel
  bins at a different point on each scale. Against
  `torchaudio.compliance.kaldi.fbank` with the FunASR settings, the int16 scale gives a
  max absolute error of 3.6e-04 and the unit scale gives 1.1e+01.
* `languages` and `textnorm` are the FunASR `embed` table indices. They select the
  prompt frames, so they must match the checkpoint and not be renumbered.

Set `dither` to 0 when you compare against native FunASR. `WavFrontend` defaults to
`dither = 1.0`, one LSB of noise per sample at the int16 scale, and native runs on the
same clip then give different transcripts. The export is deterministic.

The first four output tokens are the detected language, the emotion, the audio event
and the text normalization mode, for example `<|en|>`, `<|NEUTRAL|>`, `<|Speech|>` and
`<|woitn|>`. onnx-asr keeps them in `TimestampedResult.tokens` and out of
`TimestampedResult.text`.
