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
