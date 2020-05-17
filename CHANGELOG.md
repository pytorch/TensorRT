# Changelog

## 0.0.2  (2020-05-17)


### Bug Fixes

* **//core/conversion:** Check for calibrator before setting int8 mode ([3afd209](https://github.com/NVIDIA/TRTorch/commit/3afd209))
* **//core/conversion/conversionctx:** Check both tensor and eval maps ([2d65ece](https://github.com/NVIDIA/TRTorch/commit/2d65ece))
* **//core/conversion/converters/impl/element_wise:** Fix broadcast ([a9f33e4](https://github.com/NVIDIA/TRTorch/commit/a9f33e4))
* **//cpp:** Remove deprecated script namespace ([d70760f](https://github.com/NVIDIA/TRTorch/commit/d70760f))
* **//cpp/api:** Better inital condition for the dataloader iterator to ([8d22bdd](https://github.com/NVIDIA/TRTorch/commit/8d22bdd))
* **//cpp/api:** Remove unecessary destructor in ptq class ([fc70267](https://github.com/NVIDIA/TRTorch/commit/fc70267))
* **//cpp/api:** set a default for calibrator ([825be69](https://github.com/NVIDIA/TRTorch/commit/825be69))
* **//cpp/ptq:** remove some logging from ptq app ([b989c7f](https://github.com/NVIDIA/TRTorch/commit/b989c7f))
* Address issues in PR ([cd24f26](https://github.com/NVIDIA/TRTorch/commit/cd24f26))
* **//cpp/ptq:** Tracing model in eval mode wrecks accuracy in Libtorch ([54a24b3](https://github.com/NVIDIA/TRTorch/commit/54a24b3))
* **//docs:** add nojekyll file ([2a02cd5](https://github.com/NVIDIA/TRTorch/commit/2a02cd5))
* **//docs:** fix version links ([11555f7](https://github.com/NVIDIA/TRTorch/commit/11555f7))
* **//py:** Build system issues ([c1de126](https://github.com/NVIDIA/TRTorch/commit/c1de126))
* **//py:** Ignore generated version file ([9e37dc1](https://github.com/NVIDIA/TRTorch/commit/9e37dc1))
* bypass jeykll, also add PR template ([a41c400](https://github.com/NVIDIA/TRTorch/commit/a41c400))


### Features

* **//core/conversion/conversionctx:** Make op precision available at ([78a1c61](https://github.com/NVIDIA/TRTorch/commit/78a1c61))
* **//core/conversion/converters/impl/shuffle:** Implement aten::resize ([353f2d2](https://github.com/NVIDIA/TRTorch/commit/353f2d2))
* **//core/execution:** Type checking for the executor, now is the ([2dd1ba3](https://github.com/NVIDIA/TRTorch/commit/2dd1ba3))
* **//core/lowering:** New freeze model pass and new exception ([4acc3fd](https://github.com/NVIDIA/TRTorch/commit/4acc3fd))
* **//core/quantization:** skeleton of INT8 PTQ calibrator ([dd443a6](https://github.com/NVIDIA/TRTorch/commit/dd443a6))
* **//core/util:** New logging level for Graph Dumping ([90c44b9](https://github.com/NVIDIA/TRTorch/commit/90c44b9))
* **//cpp/api:** Adding max batch size setting ([1b25542](https://github.com/NVIDIA/TRTorch/commit/1b25542))
* **//cpp/api:** Functional Dataloader based PTQ ([f022dfe](https://github.com/NVIDIA/TRTorch/commit/f022dfe))
* **//cpp/api:** Remove the extra includes in the API header ([2f86f84](https://github.com/NVIDIA/TRTorch/commit/2f86f84))
* **//cpp/ptq:** Add a feature to the dataset to use less than the full ([5f36f47](https://github.com/NVIDIA/TRTorch/commit/5f36f47))
* **//cpp/ptq/training:** Training recipe for VGG16 Classifier on ([676bf56](https://github.com/NVIDIA/TRTorch/commit/676bf56))
* **//lowering:** centralize lowering and try to use PyTorch Conv2DBN folding ([fad4a10](https://github.com/NVIDIA/TRTorch/commit/fad4a10))
* **//py:** API now produces valid engines that are consumable by ([72bc1f7](https://github.com/NVIDIA/TRTorch/commit/72bc1f7))
* **//py:** Inital introduction of the Python API ([7088245](https://github.com/NVIDIA/TRTorch/commit/7088245))
* **//py:** Manylinux container and build system for multiple python ([639c2a3](https://github.com/NVIDIA/TRTorch/commit/639c2a3))
* **//py:** Working portable package ([482ef2c](https://github.com/NVIDIA/TRTorch/commit/482ef2c))
* **//tests:** New optional accuracy tests to check INT8 and FP16 ([df74136](https://github.com/NVIDIA/TRTorch/commit/df74136))
* **//cpp/api:** Working INT8 Calibrator, also resolves [#41](https://github.com/NVIDIA/TRTorch/issues/41) ([5c0d737](https://github.com/NVIDIA/TRTorch/commit/5c0d737))
* **aten::flatten:** Adds a converter for aten flatten since MM is the ([d945eb9](https://github.com/NVIDIA/TRTorch/commit/d945eb9))
* **aten::matmul|aten::addmm:** Adds support for aten::matmul and ([c5b6202](https://github.com/NVIDIA/TRTorch/commit/c5b6202))
* Support non cxx11-abi builds for use in python api ([83e0ed6](https://github.com/NVIDIA/TRTorch/commit/83e0ed6))
* **aten::size [static]:** Implement a aten::size converter for static input size ([0548540](https://github.com/NVIDIA/TRTorch/commit/0548540))
* **conv2d_to_convolution:** A pass to map aten::conv2d to _convolution ([2c5c0d5](https://github.com/NVIDIA/TRTorch/commit/2c5c0d5))



## 0.0.1 (2020-03-31)


### Bug Fixes

* **//core/conversion/converters/impl/linear:** In inserting flatten for ([377ad67](https://github.com/NVIDIA/TRTorch/commit/377ad67))
* **//core/conversion/converters/impl/reduce:** Adds support for multiple ([7622a97](https://github.com/NVIDIA/TRTorch/commit/7622a97))
* **//cpp/api:** The actual api was getting stripped, using alwayslink to ([cf4a8aa](https://github.com/NVIDIA/TRTorch/commit/cf4a8aa))
* **//tests:** Forgot to change this path to modules ([89bff0f](https://github.com/NVIDIA/TRTorch/commit/89bff0f))
* **//tests/modules:** Remove an old script ([8be79e1](https://github.com/NVIDIA/TRTorch/commit/8be79e1))
* **//tests/modules:** Remove lenet test and rename generated ([4b58d3b](https://github.com/NVIDIA/TRTorch/commit/4b58d3b))


### Features

* **//core/conversion/conversionctx:** Move inline function to associate ([6ab9814](https://github.com/NVIDIA/TRTorch/commit/6ab9814))
* **//core/conversion/converter/Arg:** Add typechecking to the unwrap ([73bfd4c](https://github.com/NVIDIA/TRTorch/commit/73bfd4c))
* **//core/conversion/converters/impl:** Non dimensional reduce ([ccab7b9](https://github.com/NVIDIA/TRTorch/commit/ccab7b9))
* **//core/conversion/converters/impl/reduce:** adds the rest of TRT's ([956b0c5](https://github.com/NVIDIA/TRTorch/commit/956b0c5))
* **//core/conversion/converters/impl/reduce:** Mean reduce converter ([259aa4c](https://github.com/NVIDIA/TRTorch/commit/259aa4c))
* **CheckMethodOperatorSupport:** A new API which will check the graph ([28ee445](https://github.com/NVIDIA/TRTorch/commit/28ee445)), closes [#26](https://github.com/NVIDIA/TRTorch/issues/26)
* **hardtanh:** Adds support for the the hard tanh operator ([391af52](https://github.com/NVIDIA/TRTorch/commit/391af52))


