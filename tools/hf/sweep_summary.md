# Accuracy sweep results

Total: 23 models, PASS=23, FAIL=0

| Family | Model | Verdict | Min cos_sim | Time |
|---|---|---|---|---|
| encoder | `bert-base-uncased` | PASS | 0.999998 | 14s |
| encoder | `distilbert-base-uncased` | PASS | 0.999999 | 11s |
| encoder | `roberta-base` | PASS | 0.999999 | 14s |
| encoder | `albert-base-v2` | PASS | 1.000000 | 14s |
| encoder | `google/electra-small-discriminator` | PASS | 0.999996 | 13s |
| encoder | `sentence-transformers/all-MiniLM-L6-v2` | PASS | 0.999998 | 11s |
| encoder | `google/vit-base-patch16-224` | PASS | 0.999994 | 14s |
| encoder | `microsoft/resnet-50` | PASS | 0.999887 | 13s |
| encoder | `microsoft/swin-tiny-patch4-window7-224` | PASS | 0.999997 | 16s |
| encoder | `facebook/convnext-tiny-224` | PASS | 0.999999 | 13s |
| llm | `gpt2` | PASS | 1.000000 | 19s |
| llm | `meta-llama/Llama-3.2-1B-Instruct` | PASS | 0.999998 | 31s |
| llm | `Qwen/Qwen2.5-0.5B-Instruct` | PASS | 0.999994 | 36s |
| llm | `microsoft/Phi-3-mini-4k-instruct` | PASS | 0.999998 | 55s |
| llm | `google/gemma-3-1b-it` | PASS | 0.999987 | 50s |
| llm | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | PASS | 0.999998 | 35s |
| llm | `facebook/opt-125m` | PASS | 1.000000 | 18s |
| llm | `EleutherAI/pythia-160m` | PASS | 0.999999 | 21s |
| seq2seq | `t5-small` | PASS | 1.000000 | 13s |
| audio | `openai/whisper-tiny` | PASS | 0.999997 | 14s |
| multimodal | `openai/clip-vit-base-patch32` | PASS | 0.999992 | 16s |
| multimodal | `google/siglip-base-patch16-224` | PASS | 0.999973 | 18s |
| diffusion | `OFA-Sys/small-stable-diffusion-v0` | PASS | 0.999999 | 21s |
