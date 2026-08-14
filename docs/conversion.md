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

## ESPnet E-Branchformer (CTC and attention decoder)

Install **ESPnet**, **transformers** and **torch**:

```sh
pip install espnet espnet_model_zoo transformers torch
```

ESPnet models with an s3prl frontend keep the upstream (here `hf_w2v2_bert2`, which wraps
`facebook/w2v-bert-2.0`) outside the ONNX graph in ESPnet, so the export rebuilds the
model without the s3prl frontend and puts the HuggingFace upstream back in front of it.
The graph input is the w2v-BERT feature (80 mel bins, 2 frame stacking, 160 values per
frame), which onnx-asr computes with the `"w2vbert"` preprocessor.

```py
import argparse
import json
from pathlib import Path

import torch
import yaml
from espnet2.tasks.asr import ASRTask
from huggingface_hub import hf_hub_download
from transformers import Wav2Vec2BertModel

repo_id = "inesc-id/EBranch-w2vBERT2-EP"
exp = "exp/asr_train_set_425_NEW_SEAMLESS"
onnx_dir = Path("espnet-onnx")
onnx_dir.mkdir(exist_ok=True)

config = hf_hub_download(repo_id, f"{exp}/config.yaml")
stats = hf_hub_download(repo_id, "exp/asr_stats_raw_pt_bpe5000_sp/train/feats_stats.npz")
checkpoint = hf_hub_download(repo_id, f"{exp}/valid.acc.ave_10best.pth")

with open(config) as f:
    cfg = yaml.safe_load(f)

token_list = list(cfg["token_list"])
cfg |= {
    "frontend": None,
    "frontend_conf": {},
    "input_size": 1024,  # output size of the s3prl frontend
    "specaug": None,
    "specaug_conf": {},
    "normalize_conf": {"stats_file": stats},
    "bpemodel": None,
}

model = ASRTask.build_model(argparse.Namespace(**cfg)).eval()
state_dict = torch.load(checkpoint, map_location="cpu", weights_only=False)
model.load_state_dict({k: v for k, v in state_dict.items() if not k.startswith("frontend.")}, strict=False)

upstream = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0").eval()
layer_weights = state_dict["frontend.featurizer.weights"]


class EspnetEncoderCtc(torch.nn.Module):
    def encode(self, features, features_lens):
        mask = torch.arange(features.shape[1])[None, :] < features_lens[:, None]
        hidden_states = upstream(
            input_features=features, attention_mask=mask.long(), output_hidden_states=True
        ).hidden_states
        weights = torch.softmax(layer_weights, dim=0)
        feats = (torch.stack(hidden_states, dim=0) * weights[:, None, None, None]).sum(dim=0)
        feats_lens = torch.minimum(torch.full_like(features_lens, feats.shape[1]), features_lens)
        feats, feats_lens = model.normalize(feats, feats_lens)
        feats, feats_lens = model.preencoder(feats, feats_lens)
        encoder_out, encoder_out_lens, _ = model.encoder(feats, feats_lens)
        return encoder_out, encoder_out_lens

    def forward(self, features, features_lens):
        encoder_out, encoder_out_lens = self.encode(features, features_lens)
        return model.ctc.log_softmax(encoder_out), encoder_out_lens


class EspnetDecoder(torch.nn.Module):
    def forward(self, tokens, encoder_out, encoder_out_lens):
        tokens_lens = torch.full((tokens.shape[0],), tokens.shape[1], dtype=torch.long)
        logits, _ = model.decoder(encoder_out, encoder_out_lens, tokens, tokens_lens)
        return torch.log_softmax(logits, dim=-1)


features = torch.randn(1, 200, 160)
features_lens = torch.tensor([200])

torch.onnx.export(
    EspnetEncoderCtc().eval(),
    (features, features_lens),
    str(onnx_dir / "model.onnx"),
    input_names=["features", "features_lens"],
    output_names=["logprobs", "logprobs_lens"],
    dynamic_axes={
        "features": {0: "batch", 1: "frames"},
        "features_lens": {0: "batch"},
        "logprobs": {0: "batch", 1: "frames_out"},
        "logprobs_lens": {0: "batch"},
    },
    opset_version=17,
)

with (onnx_dir / "vocab.txt").open("wt") as f:
    for i, token in enumerate(token_list):
        f.write(f"{'<blk>' if token == '<blank>' else token} {i}\n")

with (onnx_dir / "config.json").open("wt") as f:
    json.dump({"model_type": "espnet-ctc", "subsampling_factor": 8}, f, indent=2)
```

ESPnet puts the CTC blank at index 0 and names it `<blank>`; rename it to `<blk>` so the
vocab loader finds it. The SentencePiece `▁` prefix is kept, onnx-asr converts it to a
space when decoding. `subsampling_factor` scales token timestamps: the w2v-BERT frames
are 20 ms long and the E-Branchformer `conv2d` input layer subsamples them by 4, so one
output frame is 80 ms and the factor is 8.

For attention decoding, export `EspnetEncoderCtc.encode` as `encoder.onnx` (outputs
`encoder_out` and `encoder_out_lens`) and `EspnetDecoder` as `decoder.onnx` (inputs
`tokens`, `encoder_out` and `encoder_out_lens`, output `logprobs`), and set `model_type`
to `espnet-aed`. The decoder graph has no key-value cache, onnx-asr recomputes it over
the whole prefix at every step.

### Notes

Use the `torch.export` based exporter (`dynamo=True`). With the older TorchScript
exporter the branch-merge `torch.cat` of the E-Branchformer layer fails to convert
(`All tensors must have the same rank`), because the rank of the attention output is not
known statically after its reshape.

The frontend makes the graph larger than the 2 GB protobuf limit, so the weights go into
a sidecar file next to the graph. The exporter writes `<name>.onnx.data`. Keep the
sidecar in the same directory as the graph, and upload it with the model: the download
pattern of onnx-asr is `<name>.onnx?data`, which matches `<name>.onnx.data` and
`<name>.onnx_data`.

Trim the features to `features_lens.max()` before the encoder. ESPnet builds its masks
with length `max(ilens)`, and the 2 frame stacking of the preprocessor can leave one
extra half-padded frame, which would put the subsampled mask out of step with the
convolution output.

Some published ESPnet checkpoints are trained with recipe code that is not in any ESPnet
release. `inesc-id/EBranch-w2vBERT2-EP`, for example, sets `use_rope: true` and
`pos_enc_layer_type: ''`, which no public ESPnet accepts. Check that the checkpoint loads
with `strict=True` before you export; if keys such as `attn.use_rope.freqs` are reported
as unexpected, the model needs code that the release does not have.
