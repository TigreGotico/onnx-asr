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

## Nvidia NeMo models without feature normalization

The NeMo preprocessor normalizes the log-mel features per utterance. Some checkpoints
are trained with `normalize: NA` instead, and their encoder expects the raw log-mel.
Nemotron ASR is one family: the Hugging Face
`NemotronAsrStreamingFeatureExtractor` of `nvidia/nemotron-3.5-asr-streaming-0.6b`
computes 128 log-mel bins with `n_fft` 512, `win_length` 400, `hop_length` 160 and
preemphasis 0.97, and applies no normalization at all. Running such a model through
the normalizing preprocessor gives a transcript that is wrong but still looks like
text, so it is worth checking the source feature extractor rather than assuming.

Add `"normalize": false` to `config.json`. onnx-asr then selects the `nemo<size>_raw`
preprocessor, which is the same log-mel front end with the normalization step removed:

```json
{
    "model_type": "nemo-conformer-rnnt",
    "features_size": 128,
    "subsampling_factor": 8,
    "max_tokens_per_step": 10,
    "normalize": false
}
```

Nemotron ASR is a cache-aware streaming FastConformer. Exported for full-utterance
offline use, the encoder runs over the whole utterance with the chunked-limited
attention mask baked in at the largest supported lookahead, and the language prompt is
a frozen one-hot, so the graph needs no streaming caches and no extra inputs:

```py
import torch
from torch import nn
from transformers import AutoProcessor, Nemotron3_5AsrForRNNT
from transformers.models.nemotron_asr_streaming import modeling_nemotron_asr_streaming as enc_mod

model_id = "nvidia/nemotron-3.5-asr-streaming-0.6b"
language = "auto"
num_lookahead_tokens = 13

processor = AutoProcessor.from_pretrained(model_id)
model = Nemotron3_5AsrForRNNT.from_pretrained(model_id, dtype=torch.float32).eval()
prompt_id = processor.prompt_dictionary.get(language, model.config.default_prompt_id)

chunk_size = num_lookahead_tokens + 1
left_chunks = (model.config.encoder_config.sliding_window - 1) // chunk_size


def create_bidirectional_mask(config=None, inputs_embeds=None, attention_mask=None, **kwargs):
    """Traceable replacement for the chunked-limited mask builder.

    `create_bidirectional_mask` builds the mask through `torch.vmap`, which the ONNX
    exporter cannot trace. The result is exactly the padding mask AND
    `0 <= q_chunk - kv_chunk <= left_context_chunks`.
    """
    batch, seq_len = inputs_embeds.shape[0], inputs_embeds.shape[1]
    chunk = torch.div(torch.arange(seq_len), chunk_size, rounding_mode="trunc")
    chunk_diff = chunk[:, None] - chunk[None, :]
    mask = ((chunk_diff >= 0) & (chunk_diff <= left_chunks))[None, None].expand(batch, 1, seq_len, seq_len)
    if attention_mask is not None:
        mask = mask & attention_mask.bool()[:, None, None, :]
    return mask


enc_mod.create_bidirectional_mask = create_bidirectional_mask


class Encoder(nn.Module):
    def forward(self, audio_signal, length):
        features = audio_signal.transpose(1, 2)
        attention_mask = (torch.arange(features.shape[1])[None, :] < length[:, None]).long()
        hidden = model.encoder(
            input_features=features,
            attention_mask=attention_mask,
            num_lookahead_tokens=num_lookahead_tokens,
            use_cache=False,
        ).last_hidden_state

        one_hot = torch.zeros(model.config.num_prompts)
        one_hot[prompt_id] = 1.0
        one_hot = one_hot[None, None, :].expand(hidden.shape[0], hidden.shape[1], -1)

        outputs = model.encoder_projector(model.prompt_projector(torch.cat([hidden, one_hot], dim=-1)))
        return outputs.transpose(1, 2), model.encoder._get_subsampling_output_length(length).to(torch.int64)


class DecoderJoint(nn.Module):
    def forward(self, encoder_outputs, targets, target_length, input_states_1, input_states_2):
        embeddings = model.decoder.embedding(targets.to(torch.long))
        lstm_out, (h, c) = model.decoder.lstm(embeddings, (input_states_1, input_states_2))
        dec = model.decoder.decoder_projector(lstm_out)
        # Keep `target_length` in the graph: onnx-asr always feeds it and onnxruntime
        # rejects an input the graph does not declare.
        dec = dec + 0.0 * target_length.to(dec.dtype).view(-1, 1, 1)
        logits = model.joint(
            encoder_hidden_states=encoder_outputs.transpose(1, 2)[:, :, None, :],
            decoder_hidden_states=dec[:, None, :, :],
        )
        return logits, h, c


torch.onnx.export(
    Encoder().eval(),
    (torch.randn(1, model.config.encoder_config.num_mel_bins, 400), torch.tensor([400], dtype=torch.int64)),
    "encoder-model.onnx",
    input_names=["audio_signal", "length"],
    output_names=["outputs", "encoded_lengths"],
    dynamic_axes={
        "audio_signal": {0: "batch", 2: "time"},
        "length": {0: "batch"},
        "outputs": {0: "batch", 2: "time_out"},
        "encoded_lengths": {0: "batch"},
    },
    opset_version=17,
    dynamo=False,
)

states = torch.zeros(model.config.num_decoder_layers, 1, model.config.decoder_hidden_size)
torch.onnx.export(
    DecoderJoint().eval(),
    (
        torch.randn(1, model.config.decoder_hidden_size, 1),
        torch.zeros(1, 1, dtype=torch.int32),
        torch.ones(1, dtype=torch.int32),
        states,
        states.clone(),
    ),
    "decoder_joint-model.onnx",
    input_names=["encoder_outputs", "targets", "target_length", "input_states_1", "input_states_2"],
    output_names=["outputs", "output_states_1", "output_states_2"],
    dynamic_axes={
        "encoder_outputs": {0: "batch", 2: "time"},
        "targets": {0: "batch", 1: "tokens"},
        "target_length": {0: "batch"},
        "input_states_1": {1: "batch"},
        "input_states_2": {1: "batch"},
        "outputs": {0: "batch", 1: "time", 2: "tokens"},
        "output_states_1": {1: "batch"},
        "output_states_2": {1: "batch"},
    },
    opset_version=17,
    dynamo=False,
)

with open("vocab.txt", "wt") as f:
    for i in range(model.config.vocab_size):
        token = processor.tokenizer.convert_ids_to_tokens(i)
        f.write(f"{'<blk>' if i == model.config.blank_token_id else token} {i}\n")
```

Compare the patched mask against the original one on a fixed input before you export.
The replacement must be exact, not close.
