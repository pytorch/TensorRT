# Flux smoke test on a small GPU

Run a single dual-stream Flux transformer block with synthetic inputs:

```shell
python tools/perf/Flux/flux_perf.py --dtype fp16 --low_vram_mode --smoke_test
```

This mode constructs a reduced transformer with random weights. It keeps the
standard Flux hidden dimensions but uses 64 image tokens and 16 text tokens.
It does not load the pretrained checkpoint, text encoders, VAE, or ModelOpt, and
does not generate an image. `--model` is unused in this mode. It checks PyTorch
inference, full TensorRT compilation, and agreement between their outputs.
TensorRT workspace is limited to 1 GiB; this is not a limit on total GPU memory.

To check PyTorch first, add `--pytorch`. To test a single-stream block instead:

```shell
python tools/perf/Flux/flux_perf.py --dtype fp16 --low_vram_mode --smoke_test --num_layers 0 --num_single_layers 1
```

Use `--num_layers 1 --num_single_layers 1` to test both block types together.
These counts apply only to `--smoke_test`. The normal benchmark still loads the
full pipeline. Passing the reduced test does not establish that the full pipeline
fits in memory or can generate images on the target system.
