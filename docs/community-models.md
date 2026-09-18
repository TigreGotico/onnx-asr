# Community Models

The models in [supported model names](usage.md#supported-model-names) are maintained or
selected by the onnx-asr project. The wider community also publishes compatible models
on Hugging Face, including fine-tunes for additional languages and alternative
quantizations.

[Browse all models tagged `onnx-asr`](https://huggingface.co/models?other=onnx-asr&sort=trending)

Use the full Hugging Face repository ID to load a community model:

```py
import onnx_asr

model = onnx_asr.load_model("xezpeleta/parakeet-tdt-0.6b-v3-basque-onnx-asr")
print(model.recognize("test.wav"))
```

> [!IMPORTANT]
> This is a curated snapshot inspected on **July 16, 2026**, not an exhaustive registry
> or compatibility guarantee. The entries below passed metadata and file-layout
> inspection only: they were not downloaded, executed, benchmarked, or checked for
> transcription quality. Community models are maintained and supported by their
> publishers.

## Language fine-tunes

These repositories convert distinct upstream fine-tunes rather than copying an existing
onnx-asr export.

| Repository | Owner | Language | Architecture | Upstream model | Precision | License | Verification |
|---|---|---|---|---|---|---|---|
| [`xezpeleta/parakeet-tdt-0.6b-v3-basque-onnx-asr`](https://huggingface.co/xezpeleta/parakeet-tdt-0.6b-v3-basque-onnx-asr) | xezpeleta | Basque | NeMo Conformer TDT | [`itzune/parakeet-tdt-0.6b-v3-basque`](https://huggingface.co/itzune/parakeet-tdt-0.6b-v3-basque) | FP32, INT8 encoder | CC-BY-4.0 | Metadata inspected |
| [`alefiury/parakeet-tdt-0.6b-v3-ptBR-TAGARELA-onnx`](https://huggingface.co/alefiury/parakeet-tdt-0.6b-v3-ptBR-TAGARELA-onnx) | alefiury | Brazilian Portuguese | NeMo Conformer TDT | [`alexandreacff/parakeet-tdt-0.6b-v3-ptBR-plus`](https://huggingface.co/alexandreacff/parakeet-tdt-0.6b-v3-ptBR-plus) | FP32 | CC-BY-4.0 | Metadata inspected |
| [`Buttermilk03/parakeet-primeline-onnx`](https://huggingface.co/Buttermilk03/parakeet-primeline-onnx) | Buttermilk03 | German | NeMo Conformer TDT | [`primeline/parakeet-primeline`](https://huggingface.co/primeline/parakeet-primeline) | FP32, INT8 | CC-BY-4.0 | Metadata inspected |
| [`AlinClaudiu/SpeD-ParakeetRo-110M-onnx`](https://huggingface.co/AlinClaudiu/SpeD-ParakeetRo-110M-onnx) | AlinClaudiu | Romanian | NeMo Conformer CTC | [`gabrielpirlo/SpeD_ParakeetRo_110M_TDT-CTC`](https://huggingface.co/gabrielpirlo/SpeD_ParakeetRo_110M_TDT-CTC) | FP32 | Apache-2.0 | Metadata inspected |
| [`AigizK/GigaAM-Bashkir-CV25-ONNX`](https://huggingface.co/AigizK/GigaAM-Bashkir-CV25-ONNX) | AigizK | Bashkir | GigaAM Multilingual CTC | [`AigizK/GigaAM-Bashkir-CV25`](https://huggingface.co/AigizK/GigaAM-Bashkir-CV25) | FP32, INT8 | MIT | Metadata inspected |

## OpenVoiceOS model families

[OpenVoiceOS](https://huggingface.co/OpenVoiceOS) publishes a large coordinated set of
conversions. The table uses one representative repository for each family; follow its
owner link to find other languages and model sizes.

| Family and representative | Languages | Architecture | Upstream publisher | Precision in representative | License | Verification |
|---|---|---|---|---|---|---|
| [AI4Bharat IndicConformer](https://huggingface.co/OpenVoiceOS/ai4bharat-indicconformer-hi-onnx) | Indic languages | NeMo Conformer CTC | [AI4Bharat](https://huggingface.co/ai4bharat) | FP32, INT8 | MIT | Metadata inspected |
| [IISc Vaani FastConformer](https://huggingface.co/OpenVoiceOS/artpark-iisc-vaani-fastconformer-multi-onnx) | Multilingual and individual Indic languages | NeMo Conformer TDT | [ARTPARK-IISc](https://huggingface.co/ARTPARK-IISc) | FP32, INT8 | MIT | Metadata inspected |
| [NVIDIA monolingual Conformer](https://huggingface.co/OpenVoiceOS/nvidia-en-conformer-ctc-large-onnx) | Multiple languages | NeMo Conformer CTC/RNN-T | [NVIDIA](https://huggingface.co/nvidia) | FP32, INT8 | CC-BY-4.0 | Metadata inspected |
| [Localized Parakeet](https://huggingface.co/OpenVoiceOS/yuriyvnv-parakeet-tdt-0.6b-pl-onnx) | Polish, Estonian, Dutch, Slovenian, Portuguese, and others | NeMo Conformer TDT | Community fine-tunes | FP32 | CC-BY-4.0 | Metadata inspected |
| [Wav2Vec2](https://huggingface.co/OpenVoiceOS/wav2vec2-xlsr-300m-finnish-onnx) | Multiple languages | Wav2Vec2 CTC | Multiple publishers | FP32 | Apache-2.0 | Metadata inspected |

Licenses can differ between repositories in a family. Always check the selected model
card and its upstream model before redistribution or commercial use.

### One repository, one model per subfolder

[`OpenVoiceOS/onnx-asr-community-w2v-ctc`](https://huggingface.co/OpenVoiceOS/onnx-asr-community-w2v-ctc)
holds 47 Wav2Vec2-family CTC conversions as subfolders of one repository. Name the
subfolder after the repository id to load one of them:

```py
import onnx_asr

model = onnx_asr.load_model("OpenVoiceOS/onnx-asr-community-w2v-ctc/lgris__bp500-xlsr")
print(model.recognize("test.wav"))
```

Each subfolder carries `config.json`, `model.onnx` (with `model.onnx.data` for the
1B models), `vocab.txt`, a card and the licence file. The card states the checks
made at export: transcript parity against the PyTorch model on a few clips, and a
word error rate only where an evaluation set could be named. Feature normalisation
is inside the graph; input is 16 kHz mono.

| Subfolder | Language | Architecture | Upstream model | License | WER at export |
|---|---|---|---|---|---|
| `ghananlpcommunity__w2v-bert-2.0_brazilian_portugese_alpha_farmerline` | Portuguese (Brazil) | w2v-bert 2.0 CTC | [`ghananlpcommunity/w2v-bert-2.0_brazilian_portugese_alpha_farmerline`](https://huggingface.co/ghananlpcommunity/w2v-bert-2.0_brazilian_portugese_alpha_farmerline) | mit | not recorded |
| `ghananlpcommunity__w2v-bert-2.0_dyula_farmerline` | Dyula | w2v-bert 2.0 CTC | [`ghananlpcommunity/w2v-bert-2.0_dyula_farmerline`](https://huggingface.co/ghananlpcommunity/w2v-bert-2.0_dyula_farmerline) | mit | not recorded |
| `ghananlpcommunity__w2v-bert-2.0_ewe_2_farmerline` | Ewe | w2v-bert 2.0 CTC | [`ghananlpcommunity/w2v-bert-2.0_ewe_2_farmerline`](https://huggingface.co/ghananlpcommunity/w2v-bert-2.0_ewe_2_farmerline) | mit | not recorded |
| `ghananlpcommunity__w2v-bert-2.0_igbo_v1_farmerline` | Igbo | w2v-bert 2.0 CTC | [`ghananlpcommunity/w2v-bert-2.0_igbo_v1_farmerline`](https://huggingface.co/ghananlpcommunity/w2v-bert-2.0_igbo_v1_farmerline) | mit | not recorded |
| `ghananlpcommunity__w2v-bert-2.0_kamba_farmerline` | Kamba | w2v-bert 2.0 CTC | [`ghananlpcommunity/w2v-bert-2.0_kamba_farmerline`](https://huggingface.co/ghananlpcommunity/w2v-bert-2.0_kamba_farmerline) | mit | not recorded |
| `ghananlpcommunity__w2v-bert-2.0_kikuyu_farmerline` | Kikuyu | w2v-bert 2.0 CTC | [`ghananlpcommunity/w2v-bert-2.0_kikuyu_farmerline`](https://huggingface.co/ghananlpcommunity/w2v-bert-2.0_kikuyu_farmerline) | mit | not recorded |
| `ghananlpcommunity__w2v-bert-2.0_krio_v3_farmerline` | Krio | w2v-bert 2.0 CTC | [`ghananlpcommunity/w2v-bert-2.0_krio_v3_farmerline`](https://huggingface.co/ghananlpcommunity/w2v-bert-2.0_krio_v3_farmerline) | mit | not recorded |
| `ghananlpcommunity__w2v-bert-2.0_luganda_farmerline` | Luganda | w2v-bert 2.0 CTC | [`ghananlpcommunity/w2v-bert-2.0_luganda_farmerline`](https://huggingface.co/ghananlpcommunity/w2v-bert-2.0_luganda_farmerline) | mit | not recorded |
| `ghananlpcommunity__w2v-bert-2.0_somali_alpha_farmerline` | Somali | w2v-bert 2.0 CTC | [`ghananlpcommunity/w2v-bert-2.0_somali_alpha_farmerline`](https://huggingface.co/ghananlpcommunity/w2v-bert-2.0_somali_alpha_farmerline) | mit | not recorded |
| `ghananlpcommunity__w2v-bert-2.0_swahili_alpha_farmerline` | Swahili | w2v-bert 2.0 CTC | [`ghananlpcommunity/w2v-bert-2.0_swahili_alpha_farmerline`](https://huggingface.co/ghananlpcommunity/w2v-bert-2.0_swahili_alpha_farmerline) | mit | not recorded |
| `ghananlpcommunity__w2v-bert-2.0_yoruba_v1_farmerline` | Yoruba | w2v-bert 2.0 CTC | [`ghananlpcommunity/w2v-bert-2.0_yoruba_v1_farmerline`](https://huggingface.co/ghananlpcommunity/w2v-bert-2.0_yoruba_v1_farmerline) | mit | not recorded |
| `lgris__WavLM-large-CORAA-pt` | Portuguese (Brazil) | WavLM CTC | [`lgris/WavLM-large-CORAA-pt`](https://huggingface.co/lgris/WavLM-large-CORAA-pt) | apache-2.0 | not recorded |
| `lgris__base_10k_8khz_pt` | Portuguese (Brazil) | Wav2Vec2 / XLS-R CTC | [`lgris/base_10k_8khz_pt`](https://huggingface.co/lgris/base_10k_8khz_pt) | apache-2.0 | 0.983 (FLEURS pt_br test, first 50 clips) |
| `lgris__bp-cetuc100-xlsr` | Portuguese (Brazil) | Wav2Vec2 / XLS-R CTC | [`lgris/bp-cetuc100-xlsr`](https://huggingface.co/lgris/bp-cetuc100-xlsr) | apache-2.0 | 0.886 (FLEURS pt_br test, first 50 clips) |
| `lgris__bp-commonvoice10-xlsr` | Portuguese (Brazil) | Wav2Vec2 / XLS-R CTC | [`lgris/bp-commonvoice10-xlsr`](https://huggingface.co/lgris/bp-commonvoice10-xlsr) | apache-2.0 | 0.206 (FLEURS pt_br test, first 50 clips) |
| `lgris__bp-commonvoice100-xlsr` | Portuguese (Brazil) | Wav2Vec2 / XLS-R CTC | [`lgris/bp-commonvoice100-xlsr`](https://huggingface.co/lgris/bp-commonvoice100-xlsr) | apache-2.0 | 0.169 (FLEURS pt_br test, first 50 clips) |
| `lgris__bp-lapsbm1-xlsr` | Portuguese (Brazil) | Wav2Vec2 / XLS-R CTC | [`lgris/bp-lapsbm1-xlsr`](https://huggingface.co/lgris/bp-lapsbm1-xlsr) | apache-2.0 | 0.282 (FLEURS pt_br test, first 50 clips) |
| `lgris__bp-mls100-xlsr` | Portuguese (Brazil) | Wav2Vec2 / XLS-R CTC | [`lgris/bp-mls100-xlsr`](https://huggingface.co/lgris/bp-mls100-xlsr) | apache-2.0 | 0.235 (FLEURS pt_br test, first 50 clips) |
| `lgris__bp-sid10-xlsr` | Portuguese (Brazil) | Wav2Vec2 / XLS-R CTC | [`lgris/bp-sid10-xlsr`](https://huggingface.co/lgris/bp-sid10-xlsr) | apache-2.0 | 0.340 (FLEURS pt_br test, first 50 clips) |
| `lgris__bp-tedx100-xlsr` | Portuguese (Brazil) | Wav2Vec2 / XLS-R CTC | [`lgris/bp-tedx100-xlsr`](https://huggingface.co/lgris/bp-tedx100-xlsr) | apache-2.0 | 0.203 (FLEURS pt_br test, first 50 clips) |
| `lgris__bp-voxforge1-xlsr` | Portuguese (Brazil) | Wav2Vec2 / XLS-R CTC | [`lgris/bp-voxforge1-xlsr`](https://huggingface.co/lgris/bp-voxforge1-xlsr) | apache-2.0 | 0.546 (FLEURS pt_br test, first 50 clips) |
| `lgris__bp400-xlsr` | Portuguese (Brazil) | Wav2Vec2 / XLS-R CTC | [`lgris/bp400-xlsr`](https://huggingface.co/lgris/bp400-xlsr) | apache-2.0 | 0.154 (FLEURS pt_br test, first 50 clips) |
| `lgris__bp500-base100k_voxpopuli` | Portuguese (Brazil) | Wav2Vec2 / XLS-R CTC | [`lgris/bp500-base100k_voxpopuli`](https://huggingface.co/lgris/bp500-base100k_voxpopuli) | apache-2.0 | 0.214 (FLEURS pt_br test, first 50 clips) |
| `lgris__bp500-base10k_voxpopuli` | Portuguese (Brazil) | Wav2Vec2 / XLS-R CTC | [`lgris/bp500-base10k_voxpopuli`](https://huggingface.co/lgris/bp500-base10k_voxpopuli) | apache-2.0 | 0.196 (FLEURS pt_br test, first 50 clips) |
| `lgris__bp500-xlsr` | Portuguese (Brazil) | Wav2Vec2 / XLS-R CTC | [`lgris/bp500-xlsr`](https://huggingface.co/lgris/bp500-xlsr) | apache-2.0 | 0.151 (FLEURS pt_br test, first 50 clips) |
| `lgris__bp_400h_xlsr2_300M` | Portuguese (Brazil) | Wav2Vec2 / XLS-R CTC | [`lgris/bp_400h_xlsr2_300M`](https://huggingface.co/lgris/bp_400h_xlsr2_300M) | apache-2.0 | 0.123 (FLEURS pt_br test, first 50 clips) |
| `lgris__sew-tiny-portuguese-cv` | Portuguese (Brazil) | SEW CTC | [`lgris/sew-tiny-portuguese-cv`](https://huggingface.co/lgris/sew-tiny-portuguese-cv) | apache-2.0 | not recorded |
| `lgris__sew-tiny-portuguese-cv7` | Portuguese (Brazil) | SEW CTC | [`lgris/sew-tiny-portuguese-cv7`](https://huggingface.co/lgris/sew-tiny-portuguese-cv7) | apache-2.0 | 0.317 (FLEURS pt_br test, first 50 clips) |
| `lgris__sew-tiny-portuguese-cv8` | Portuguese (Brazil) | SEW CTC | [`lgris/sew-tiny-portuguese-cv8`](https://huggingface.co/lgris/sew-tiny-portuguese-cv8) | apache-2.0 | not recorded |
| `lgris__wav2vec2-large-xls-r-300m-pt-cv` | Portuguese (Brazil) | Wav2Vec2 / XLS-R CTC | [`lgris/wav2vec2-large-xls-r-300m-pt-cv`](https://huggingface.co/lgris/wav2vec2-large-xls-r-300m-pt-cv) | apache-2.0 | 0.340 (FLEURS pt_br test, first 50 clips) |
| `lgris__wav2vec2-large-xlsr-coraa-portuguese-cv7` | Portuguese (Brazil) | Wav2Vec2 / XLS-R CTC | [`lgris/wav2vec2-large-xlsr-coraa-portuguese-cv7`](https://huggingface.co/lgris/wav2vec2-large-xlsr-coraa-portuguese-cv7) | apache-2.0 | 0.197 (FLEURS pt_br test, first 50 clips) |
| `lgris__wav2vec2-large-xlsr-coraa-portuguese-cv8` | Portuguese (Brazil) | Wav2Vec2 / XLS-R CTC | [`lgris/wav2vec2-large-xlsr-coraa-portuguese-cv8`](https://huggingface.co/lgris/wav2vec2-large-xlsr-coraa-portuguese-cv8) | apache-2.0 | 0.197 (FLEURS pt_br test, first 50 clips) |
| `lgris__wav2vec2-large-xlsr-open-brazilian-portuguese` | Portuguese (Brazil) | Wav2Vec2 / XLS-R CTC | [`lgris/wav2vec2-large-xlsr-open-brazilian-portuguese`](https://huggingface.co/lgris/wav2vec2-large-xlsr-open-brazilian-portuguese) | apache-2.0 | 0.184 (FLEURS pt_br test, first 50 clips) |
| `lgris__wav2vec2-large-xlsr-open-brazilian-portuguese-v2` | Portuguese (Brazil) | Wav2Vec2 / XLS-R CTC | [`lgris/wav2vec2-large-xlsr-open-brazilian-portuguese-v2`](https://huggingface.co/lgris/wav2vec2-large-xlsr-open-brazilian-portuguese-v2) | apache-2.0 | 0.156 (FLEURS pt_br test, first 50 clips) |
| `lgris__wav2vec2-podcasts-tagarela-combined` | Portuguese (Brazil) | Wav2Vec2 / XLS-R CTC | [`lgris/wav2vec2-podcasts-tagarela-combined`](https://huggingface.co/lgris/wav2vec2-podcasts-tagarela-combined) | apache-2.0 | 0.162 (FLEURS pt_br test, first 50 clips) |
| `lgris__wav2vec2-podcasts-tagarela-v2` | Portuguese (Brazil) | Wav2Vec2 / XLS-R CTC | [`lgris/wav2vec2-podcasts-tagarela-v2`](https://huggingface.co/lgris/wav2vec2-podcasts-tagarela-v2) | apache-2.0 | 0.152 (FLEURS pt_br test, first 50 clips) |
| `lgris__wav2vec2-xls-r-1b-cv8` | Portuguese (Brazil) | Wav2Vec2 / XLS-R CTC | [`lgris/wav2vec2-xls-r-1b-cv8`](https://huggingface.co/lgris/wav2vec2-xls-r-1b-cv8) | apache-2.0 | 0.272 (FLEURS pt_br test, first 50 clips) |
| `lgris__wav2vec2-xls-r-1b-portuguese-CORAA-3` | Portuguese (Brazil) | Wav2Vec2 / XLS-R CTC | [`lgris/wav2vec2-xls-r-1b-portuguese-CORAA-3`](https://huggingface.co/lgris/wav2vec2-xls-r-1b-portuguese-CORAA-3) | apache-2.0 | 0.524 (FLEURS pt_br test, first 50 clips) |
| `lgris__wav2vec2-xls-r-300m-gn-cv8` | Guarani | Wav2Vec2 / XLS-R CTC | [`lgris/wav2vec2-xls-r-300m-gn-cv8`](https://huggingface.co/lgris/wav2vec2-xls-r-300m-gn-cv8) | apache-2.0 | not recorded |
| `lgris__wav2vec2-xls-r-300m-gn-cv8-3` | Guarani | Wav2Vec2 / XLS-R CTC | [`lgris/wav2vec2-xls-r-300m-gn-cv8-3`](https://huggingface.co/lgris/wav2vec2-xls-r-300m-gn-cv8-3) | apache-2.0 | not recorded |
| `lgris__wav2vec2-xls-r-300m-gn-cv8-4` | Guarani | Wav2Vec2 / XLS-R CTC | [`lgris/wav2vec2-xls-r-300m-gn-cv8-4`](https://huggingface.co/lgris/wav2vec2-xls-r-300m-gn-cv8-4) | apache-2.0 | not recorded |
| `lgris__wav2vec2-xls-r-300m-tagarela-combined` | Portuguese (Brazil) | Wav2Vec2 / XLS-R CTC | [`lgris/wav2vec2-xls-r-300m-tagarela-combined`](https://huggingface.co/lgris/wav2vec2-xls-r-300m-tagarela-combined) | apache-2.0 | 0.145 (FLEURS pt_br test, first 50 clips) |
| `lgris__wav2vec2-xls-r-300m-tagarela-v2` | Portuguese (Brazil) | Wav2Vec2 / XLS-R CTC | [`lgris/wav2vec2-xls-r-300m-tagarela-v2`](https://huggingface.co/lgris/wav2vec2-xls-r-300m-tagarela-v2) | apache-2.0 | 0.125 (FLEURS pt_br test, first 50 clips) |
| `lgris__wav2vec2-xls-r-gn-cv7` | Guarani | Wav2Vec2 / XLS-R CTC | [`lgris/wav2vec2-xls-r-gn-cv7`](https://huggingface.co/lgris/wav2vec2-xls-r-gn-cv7) | apache-2.0 | not recorded |
| `lgris__wav2vec2-xls-r-pt-cv7-from-bp400h` | Portuguese (Brazil) | Wav2Vec2 / XLS-R CTC | [`lgris/wav2vec2-xls-r-pt-cv7-from-bp400h`](https://huggingface.co/lgris/wav2vec2-xls-r-pt-cv7-from-bp400h) | apache-2.0 | 0.155 (FLEURS pt_br test, first 50 clips) |
| `lgris__wav2vec2_base_10k_8khz_pt_cv7_2` | Portuguese (Brazil) | Wav2Vec2 / XLS-R CTC | [`lgris/wav2vec2_base_10k_8khz_pt_cv7_2`](https://huggingface.co/lgris/wav2vec2_base_10k_8khz_pt_cv7_2) | apache-2.0 | 0.967 (FLEURS pt_br test, first 50 clips) |
| `lgris__wavlm-large-CORAA-pt-cv7` | Portuguese (Brazil) | WavLM CTC | [`lgris/wavlm-large-CORAA-pt-cv7`](https://huggingface.co/lgris/wavlm-large-CORAA-pt-cv7) | apache-2.0 | not recorded |


## Optimized variants

Some community repositories retain an existing model but provide a materially different
ONNX optimization. These are listed separately from language fine-tunes.

| Repository | Owner | Language | Architecture | Upstream model | Precision or optimization | License | Verification |
|---|---|---|---|---|---|---|---|
| [`Olicorne/parakeet-tdt-0.6b-v3-smoothquant-onnx`](https://huggingface.co/Olicorne/parakeet-tdt-0.6b-v3-smoothquant-onnx) | Olicorne | Multilingual | NeMo Conformer TDT | [`istupakov/parakeet-tdt-0.6b-v3-onnx`](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx) | FP32, FP16, SmoothQuant INT8 | CC-BY-4.0 | Metadata inspected |
| [`gvij/parakeet-tdt-0.6b-v3-onnx-static-qdq-pc`](https://huggingface.co/gvij/parakeet-tdt-0.6b-v3-onnx-static-qdq-pc) | gvij | Multilingual | NeMo Conformer TDT | [`nvidia/parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) | Static QDQ per-channel INT8 | CC-BY-4.0 | Metadata inspected |

## Curation criteria

A repository is included when it is public and ungated, declares a model type supported
by onnx-asr, contains a valid `config.json`, and provides the expected ONNX graphs and
tokenizer or vocabulary files. It must also identify its purpose, upstream lineage, and
license.

A different fine-tuned upstream model, language, architecture, or documented
quantization method is considered meaningful. Unchanged forks, mirrors, re-uploads,
incomplete repositories, and repositories that differ only by owner are omitted. Large
coordinated conversion sets are represented as families to keep this page maintainable.

Metadata inspection does not validate ONNX graph integrity or recognition quality. If a
model fails to load, report the problem to its publisher and see the
[troubleshooting guide](troubleshooting.md).
