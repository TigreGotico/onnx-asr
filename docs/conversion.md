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
timestamps.
## HuggingFace Wav2Vec2 with per-language adapters

`facebook/mms-1b-all` is one 1B-parameter base with a small attention adapter and a
CTC head per language, for more than a thousand languages. A merged export per
language costs 3.6 GB every time. The `wav2vec2-adapters` model type keeps the base
shared: the adapter layers and the CTC head are **graph inputs**, not initializers,
so one session serves every language and a language costs only its pack.

### Model layout

```text
model.onnx            shared base, language tensors exposed as inputs
config.json           {"model_type": "wav2vec2-adapters", ...}
adapters/<iso>.npz    one array per language-dependent graph input, plus vocab_size
vocabs/<iso>.txt      onnx-asr vocabulary for that language
```

### Graph contract

| Input | Shape | Note |
| --- | --- | --- |
| `input_values` | `[batch, time]` | raw waveform, 16 kHz |
| `input_lengths` | `[batch]` | sample count per item |
| `wav2vec2.encoder.layers.<i>.adapter_layer.norm.weight` | `[hidden]` | one set per encoder layer |
| `wav2vec2.encoder.layers.<i>.adapter_layer.norm.bias` | `[hidden]` | |
| `wav2vec2.encoder.layers.<i>.adapter_layer.linear_1.weight` | `[adapter_dim, hidden]` | |
| `wav2vec2.encoder.layers.<i>.adapter_layer.linear_1.bias` | `[adapter_dim]` | |
| `wav2vec2.encoder.layers.<i>.adapter_layer.linear_2.weight` | `[hidden, adapter_dim]` | |
| `wav2vec2.encoder.layers.<i>.adapter_layer.linear_2.bias` | `[hidden]` | |
| `lm_head.weight` | `[vocab, hidden]` | `vocab` is a dynamic dimension |
| `lm_head.bias` | `[vocab]` | |

The only output is `logprobs` with shape `[batch, frames, vocab]`.

The `.npz` key names must match the graph input names. `vocab_size` is an extra key
that holds the true vocabulary size; onnx-asr trims the logits to it, so a base graph
with a padded head still decodes correctly.

### Export

Detach the language-dependent parameters from the modules and set them from the
forward arguments, so the tracer sees them as graph inputs. The stateless API
(`torch.func.functional_call`) does not work here, because the exporter traces the
module:

```py
import torch
from transformers import Wav2Vec2ForCTC

model = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all", target_lang="eng",
                                       ignore_mismatched_sizes=True).eval()
keys = sorted(n for n, _ in model.named_parameters() if ".adapter_layer." in n)
keys += ["lm_head.weight", "lm_head.bias"]


state = {k: v.detach().clone() for k, v in model.state_dict().items()}
model.requires_grad_(False)

slots = []
for key in keys:
    parent = model.get_submodule(key.rsplit(".", 1)[0])
    leaf = key.rsplit(".", 1)[1]
    del parent._parameters[leaf]
    setattr(parent, leaf, None)
    slots.append((parent, leaf))


class AdapterFed(torch.nn.Module):
    def forward(self, input_values, input_lengths, *tensors):
        for (parent, leaf), value in zip(slots, tensors, strict=True):
            setattr(parent, leaf, value)
        logits = model(input_values).logits
        return torch.nn.functional.log_softmax(logits, dim=-1)


tensors = tuple(state[k] for k in keys)
torch.onnx.export(
    AdapterFed(),
    (torch.randn(1, 16000), torch.tensor([16000], dtype=torch.int64), *tensors),
    "model.onnx",
    input_names=["input_values", "input_lengths", *keys],
    output_names=["logprobs"],
    dynamic_axes={
        "input_values": {0: "batch", 1: "time"},
        "input_lengths": {0: "batch"},
        "logprobs": {0: "batch", 1: "frames", 2: "vocab"},
        "lm_head.weight": {0: "vocab"},
        "lm_head.bias": {0: "vocab"},
    },
    opset_version=18,
)
```

Then, for each language, `model.load_adapter(iso)` and save the same keys to
`adapters/<iso>.npz` with the language's `vocab_size`, and write `vocabs/<iso>.txt`
in the format described in [Wav2Vec2 CTC](#huggingface-wav2vec2-ctc).

### Usage

```py
import onnx_asr

model = onnx_asr.load_model("wav2vec2-adapters", "mms-1b-all-onnx")
print(model.asr.languages)
print(model.recognize("test.wav", language="lg"))
```

`language` takes the pack name (`lug`), a BCP-47 tag whose primary subtag is a pack
name (`lg-UG`), or an alias from `language_aliases` in `config.json`. Without
`language`, the model uses `default_language`. An unknown language raises an error
that lists the languages the model has.

Feature normalization is baked into the graph, so the model uses the `identity`
preprocessor.

### Downloads: one language at a time

A repository with more than a thousand packs is about 10 GB, so `adapters/` and
`vocabs/` are **not** downloaded as a whole. From the Hub, `load_model` gets the base
graph, `config.json` and the pack of `default_language`. Every other pack arrives the
first time its language is used, and stays in the local cache.

| Config key | Effect |
| --- | --- |
| `languages` | Names every pack in the repository, so `model.asr.languages` and the language check work before anything is downloaded. |
| `default_language` | Fetched and loaded when the model is created. |
| `preload_languages` | Also fetched when the model is created. |

To pay the download cost up front for a known set of languages:

```py
model = onnx_asr.load_model("OpenVoiceOS/mms-1b-all-onnx")  # model type from config.json
model.asr.preload("swh", "yor", "pt")
```

A full local directory keeps working exactly as before: the packs on disk are used and
nothing is downloaded.

## Speech-LLM (audio encoder + projector + causal LM)

Models in this family transcribe with a causal language model that receives audio
embeddings, for example Qwen3-ASR, SLAM-ASR and Cohere Transcribe. The export has
three graphs:

| File | Inputs | Outputs |
| --- | --- | --- |
| `encoder.onnx` | `input_features` `(1, mel, frames)`, and, for windowed encoders, `valid_indices` `(L,)` and `attn_bias` `(1, 1, L, L)` | `audio_embeds` `(1, L, hidden)` |
| `embed_tokens.onnx` | `input_ids` `(1, S)` | `inputs_embeds` `(1, S, hidden)` |
| `decoder.onnx` | `inputs_embeds` `(1, S, hidden)`, `attn_bias` `(1, 1, S, P + S)`, `position_ids` `(1, S)`, `past_key_values.{i}.{key,value}` `(1, kv_heads, P, head_dim)` | `logits` `(1, S, vocab)`, `present.{i}.{key,value}` `(1, kv_heads, P + S, head_dim)` |

`encoder.onnx` must include the projector, so its output is already in the
embedding space of the language model. `attn_bias` is an additive float mask, so
the runtime controls the attention pattern and the graphs need no branches. The
audio encoder of Qwen3-ASR attends inside fixed windows, and the runtime builds
that block-diagonal mask in NumPy. An encoder that attends over the whole fixed
feature length, like a Whisper encoder, declares only `input_features`, and the
runtime then sends the features unchanged.

Add a `config.json`:

```json
{
    "model_type": "speech-llm",
    "features_size": 128,
    "preprocessor": "whisper128",
    "n_window": 50,
    "n_window_infer": 800,
    "eos_token_ids": [151643, 151645],
    "max_sequence_length": 512,
    "prompt_prefix_ids": [151644, 8948],
    "prompt_suffix_ids": [151645, 198],
    "language_prompt_ids": {"English": [151644, 8948]},
    "text_start_token_id": 151704,
    "tokenizer_type": "byte-level",
    "normalize_waveform": false
}
```

The prompt token ids are encoded at export time with the Hugging Face tokenizer,
so the package needs no tokenizer at runtime. `prompt_prefix_ids` ends with the
audio start token and `prompt_suffix_ids` starts with the audio end token; the
audio embeddings go between them. `language_prompt_ids` is optional and gives one
prefix per language for the `language` argument. `text_start_token_id` is also
optional: some models write a preamble before the transcription (Qwen3-ASR
writes the detected language), and the runtime drops everything up to and
including that marker token.

Two more optional keys cover the differences between model families:

* `tokenizer_type` is `"byte-level"` (default) or `"sentencepiece"`.
* `normalize_waveform` scales the waveform to zero mean and unit variance before
  the feature extractor. SLAM-ASR models need this.

For detokenization, save the tokenizer vocabulary as `vocab.json` (a
`{token: id}` map). Byte-level BPE tokens are decoded with the standard GPT-2
byte table, the same way as for Whisper. SentencePiece tokens are decoded by
replacing the space marker and resolving `<0xHH>` byte-fallback tokens.

For a SLAM-ASR model the audio embeddings come before the prompt, so
`prompt_prefix_ids` is empty and `prompt_suffix_ids` holds the whole prompt.

An export script for `Qwen/Qwen3-ASR-0.6B-hf` is in the
[issue #73 discussion](https://github.com/istupakov/onnx-asr/issues/73).

Limitations:

* The runtime processes one waveform at a time, so a batch is a loop.
* Greedy decoding only.
* Custom prompts are not supported; only the baked prompt ids.

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
## Granite Speech NAR (CTC encoder + bidirectional editor)

`ibm-granite/granite-speech-4.1-2b-nar` does not decode token by token. A conformer
encoder with a BPE CTC head writes a first-pass hypothesis, and a bidirectional
Granite language model rewrites that hypothesis in **one** forward pass. There is no
KV cache and no loop, so the export has two graphs:

| File | Inputs | Outputs |
| --- | --- | --- |
| `encoder.onnx` | `input_features` `(1, N)` raw 16 kHz waveform | `audio_embeds` `(1, L, hidden)`, `ctc_logits` `(1, C, vocab)`, `audio_embeds_lens` `(1,)`, `ctc_lens` `(1,)` |
| `editor.onnx` | `audio_embeds` `(1, L, hidden)`, `text_ids` `(1, T)` | `logits` `(1, T, vocab)` |

`encoder.onnx` holds the whole audio path: feature extraction, the conformer, the
CTC head and the Q-Former projector. It takes the raw waveform, so `config.json`
declares `"preprocessor": "identity"`.

`editor.onnx` holds the token embedding table, which is tied to the output head, so
its text input is token ids and not embeddings. The attention is bidirectional and
the runtime never pads, so the graph needs no attention mask.

Between the two graphs the runtime does three things in NumPy:

1. **CTC greedy collapse** of `ctc_logits`: argmax, merge repeated ids, then drop
   the blanks. Merging before dropping is what lets a doubled letter survive.
2. **Slot insertion**: put a blank before, after and between every surviving token,
   so `[a, b]` becomes `[_, a, _, b, _]`, padded up to `min_edit_sequence_length`.
   Each blank is a slot the editor may fill, and each hypothesis token is a slot the
   editor may keep or delete.
3. **A second CTC collapse** over the editor logits, then byte-level BPE decoding.

Add a `config.json`:

```json
{
    "model_type": "granite-nar",
    "preprocessor": "identity",
    "blank_token_id": 100257,
    "min_edit_sequence_length": 8
}
```

For detokenization, save the tokenizer vocabulary as `vocab.json` (a `{token: id}`
map), decoded with the standard GPT-2 byte table, the same way as for Whisper.

An export of `ibm-granite/granite-speech-4.1-2b-nar` is at
[OpenVoiceOS/granite-speech-4.1-2b-nar-onnx](https://huggingface.co/OpenVoiceOS/granite-speech-4.1-2b-nar-onnx).

Limitations:

* The runtime processes one waveform at a time, so a batch is a loop.
* Greedy decoding only, and no timestamps.
* Transcription only. The model card lists en, fr, de, es and pt.
