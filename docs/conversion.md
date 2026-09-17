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

Install **transformers**, **torch** and **onnx**:

```sh
pip install transformers torch onnx
```

Export the model with feature normalization (`do_normalize`) baked into the graph, so
the model uses the plain `"identity"` preprocessor (raw waveform in, no separate
feature-extractor asset needed). The same script exports WavLM and SEW CTC checkpoints,
because `AutoModelForCTC` resolves the architecture from the checkpoint:

```py
import json
from pathlib import Path

import onnx
import torch
from transformers import AutoFeatureExtractor, AutoModelForCTC, Wav2Vec2CTCTokenizer

hf_model_id = "proxectonos/Nos_ASR-wav2vec2-xls-r-300m-gl"
onnx_dir = Path("wav2vec2-onnx")
onnx_dir.mkdir(exist_ok=True)

model = AutoModelForCTC.from_pretrained(hf_model_id).eval()
# Not AutoProcessor: a checkpoint that ships an n-gram language model resolves to
# Wav2Vec2ProcessorWithLM, which imports pyctcdecode. The export needs only the
# feature extractor and the CTC vocabulary.
feature_extractor = AutoFeatureExtractor.from_pretrained(hf_model_id)
tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(hf_model_id)
do_normalize = bool(feature_extractor.do_normalize)


class NormalizedWav2Vec2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # The model is a submodule. A module-level reference fails under torch 2.14
        # with "Cannot insert a Tensor that requires grad as a constant".
        self.model = model

    def forward(self, input_values, input_lengths):
        if do_normalize:
            mask = torch.arange(input_values.shape[1])[None, :] < input_lengths[:, None]
            count = input_lengths.to(input_values.dtype).clamp(min=1)[:, None]
            mean = (input_values * mask).sum(dim=1, keepdim=True) / count
            var = (((input_values - mean) * mask) ** 2).sum(dim=1, keepdim=True) / count
            input_values = torch.where(mask, (input_values - mean) / torch.sqrt(var + 1e-5), input_values)
        logits = self.model(input_values).logits
        return torch.nn.functional.log_softmax(logits, dim=-1)


torch.set_grad_enabled(False)
torch.onnx.export(
    NormalizedWav2Vec2().eval(),
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
    dynamo=False,
)

# A graph over 2 GB (the XLS-R 1B checkpoints) is written with one external file per
# tensor. Rewrite it as model.onnx plus one model.onnx.data.
tensor_files = [p for p in onnx_dir.iterdir() if p.name != "model.onnx"]
if tensor_files:
    proto = onnx.load(str(onnx_dir / "model.onnx"))
    for p in tensor_files:
        p.unlink()
    onnx.save_model(
        proto,
        str(onnx_dir / "model.onnx"),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="model.onnx.data",
        size_threshold=0,
    )

vocab = tokenizer.get_vocab()
pad_token = tokenizer.pad_token
word_delimiter = tokenizer.word_delimiter_token

with (onnx_dir / "vocab.txt").open("wt") as f:
    for token, idx in sorted(vocab.items(), key=lambda kv: kv[1]):
        token = "<blk>" if token == pad_token else "▁" if token == word_delimiter else token
        f.write(f"{token} {idx}\n")

subsampling_factor = 1
for layer in model.base_model.feature_extractor.conv_layers:
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

The exported graph decodes greedily. A language model shipped next to the checkpoint
(`language_model/`, read by `Wav2Vec2ProcessorWithLM`) is not part of the export, so a
parity check against PyTorch must use greedy decoding on both sides.