# Source and fixture attribution

The BEVFormer adapter is an inference adaptation, not an unchanged copy of the
upstream project. It retains checkpoint-compatible parameter names and replaces
framework wrappers with fixed-shape PyTorch tensor operations.

- [BEVFormer](https://github.com/fundamentalvision/BEVFormer), revision
  `66b65f3a1f58caf0507cb2a971b9c0e7f842376c`: Tiny architecture, temporal/spatial
  attention, geometry, encoder/decoder, and detection heads; Apache-2.0.
- [MMDetection](https://github.com/open-mmlab/mmdetection/tree/v2.14.0):
  ResNet/FPN and positional encoding; Apache-2.0.
- [MMCV](https://github.com/open-mmlab/mmcv/tree/v1.4.0): deformable sampling
  and transformer components; Apache-2.0.
- [torchvision](https://github.com/pytorch/vision/tree/v0.10.1): rotation/grid
  conventions; BSD-3-Clause.

The applicable license texts and original reference-file hashes are under
`models/bevformer/`. The CUDA geometry plugin is a handwritten implementation
of the adapter's projection and motion math for the fixed Tiny input shapes.

SegFormer uses the existing Hugging Face Transformers implementation, with a
small wrapper returning raw logits. Checkpoint:
`nvidia/segformer-b0-finetuned-cityscapes-1024-1024`, revision
`21b3847fae21ddee674abd31129307b6a1235bd9`. The checkpoint and upstream software
retain their respective licenses. Checkpoint hashes are verified at load time.

BEV benchmark tensors are synthetic reset/history snapshots. SegFormer tensors
are derived from two Wikimedia Commons images by Mjrmtg, resized to 1024×1024,
converted to RGB tensors, and normalized:

- [North Valdosta Rd US41 NB Foxborough Ave intersection, dashcam](https://commons.wikimedia.org/wiki/File:North_Valdosta_Rd_US41_NB_Foxborough_Ave_intersection,_dashcam.jpg), CC0-1.0.
- [EB GA37 Side road Bryan Road, Colquitt County dashcam 2](https://commons.wikimedia.org/wiki/File:EB_GA37_Side_road_Bryan_Road,_Colquitt_County_dashcam_2.jpg), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Their detailed provenance and preprocessing records accompany the fixtures in
the bundle. They are not labelled Cityscapes evaluation inputs.

The binary bundle contains NVIDIA CUDA components, PyTorch, ExecuTorch, model
weights, and other third-party dependencies. It is a benchmark attachment for
recipients entitled to use those components; the source repository license does
not relicense them. Preserve the bundled license notices when sharing it.
