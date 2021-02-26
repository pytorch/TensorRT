# Changelog


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


## 0.0.3 (2020-07-18)


* feat!: Lock bazel version ([25f4371](https://github.com/NVIDIA/TRTorch/commit/25f4371))
* refactor(//cpp/api)!: Refactoring ptq to use includes but seperate from ([d2f8a59](https://github.com/NVIDIA/TRTorch/commit/d2f8a59))


### Bug Fixes

* **//core:** Do not compile hidden methods ([6bd1a3f](https://github.com/NVIDIA/TRTorch/commit/6bd1a3f))
* **//core/conversion:** Check for calibrator before setting int8 mode ([3afd209](https://github.com/NVIDIA/TRTorch/commit/3afd209))
* **//core/conversion:** Supress unnecessary debug messages ([2b23874](https://github.com/NVIDIA/TRTorch/commit/2b23874))
* **//core/conversion/conversionctx:** Check both tensor and eval maps ([2d65ece](https://github.com/NVIDIA/TRTorch/commit/2d65ece))
* **//core/conversion/conversionctx:** In the case of strict types and ([3611778](https://github.com/NVIDIA/TRTorch/commit/3611778))
* **//core/conversion/converters:** Fix plugin implementation for TRT 7 ([94d6a0f](https://github.com/NVIDIA/TRTorch/commit/94d6a0f))
* **//core/conversion/converters/impl:** 1d case not working ([f42562b](https://github.com/NVIDIA/TRTorch/commit/f42562b))
* **//core/conversion/converters/impl:** code works for interpolate2d/3d, doesn't work for 1d yet ([e4cb117](https://github.com/NVIDIA/TRTorch/commit/e4cb117))
* **//core/conversion/converters/impl:** Fix interpolate.cpp ([b6942a2](https://github.com/NVIDIA/TRTorch/commit/b6942a2))
* **//core/conversion/converters/impl/element_wise:** Fix broadcast ([a9f33e4](https://github.com/NVIDIA/TRTorch/commit/a9f33e4))
* **//core/conversion/evaluators:** A couple fixes for evaluators ([07ba980](https://github.com/NVIDIA/TRTorch/commit/07ba980))
* **//core/lowering:** Conv2D -> _convolution pass was triggering conv ([ca2b5f9](https://github.com/NVIDIA/TRTorch/commit/ca2b5f9))
* **//cpp:** Remove deprecated script namespace ([d70760f](https://github.com/NVIDIA/TRTorch/commit/d70760f))
* **//cpp/api:** Better inital condition for the dataloader iterator to ([8d22bdd](https://github.com/NVIDIA/TRTorch/commit/8d22bdd))
* **//cpp/api:** Remove unecessary destructor in ptq class ([fc70267](https://github.com/NVIDIA/TRTorch/commit/fc70267))
* **//cpp/api:** set a default for calibrator ([825be69](https://github.com/NVIDIA/TRTorch/commit/825be69))
* **//cpp/benchmark:** reorder benchmark so FP16 bn issue in JIT doesnt ([98527d2](https://github.com/NVIDIA/TRTorch/commit/98527d2))
* **//cpp/ptq:** Default version of the app should not resize images ([de3cbc4](https://github.com/NVIDIA/TRTorch/commit/de3cbc4))
* **//cpp/ptq:** Enable FP16 kernels for INT8 applications ([26709cc](https://github.com/NVIDIA/TRTorch/commit/26709cc))
* **//cpp/ptq:** Enable FP16 kernels for INT8 applications ([e1c5416](https://github.com/NVIDIA/TRTorch/commit/e1c5416))
* **//cpp/ptq:** remove some logging from ptq app ([b989c7f](https://github.com/NVIDIA/TRTorch/commit/b989c7f))
* **//cpp/ptq:** Tracing model in eval mode wrecks accuracy in Libtorch ([54a24b3](https://github.com/NVIDIA/TRTorch/commit/54a24b3))
* **//cpp/trtorchc:** Refactor trtorchc to use new C++ API ([789e1be](https://github.com/NVIDIA/TRTorch/commit/789e1be)), closes [#132](https://github.com/NVIDIA/TRTorch/issues/132)
* **//cpp/trtorchc:** Support building trtorchc with the pre_cxx11_abi ([172d4d5](https://github.com/NVIDIA/TRTorch/commit/172d4d5))
* **//docs:** add nojekyll file ([2a02cd5](https://github.com/NVIDIA/TRTorch/commit/2a02cd5))
* **//docs:** fix version links ([11555f7](https://github.com/NVIDIA/TRTorch/commit/11555f7))
* **//notebooks:** Fix WORKSPACE template file to reflect new build system layout ([c8ea9b7](https://github.com/NVIDIA/TRTorch/commit/c8ea9b7))
* **//py:** Build system issues ([c1de126](https://github.com/NVIDIA/TRTorch/commit/c1de126))
* **//py:** Ignore generated version file ([9e37dc1](https://github.com/NVIDIA/TRTorch/commit/9e37dc1))
* **//py:** Lib path incorrect ([ff2b13c](https://github.com/NVIDIA/TRTorch/commit/ff2b13c))
* **//tests:** Duplicated tensorrt dep ([5cd697e](https://github.com/NVIDIA/TRTorch/commit/5cd697e))
* **//third_party/tensorrt:** Fix include dir for library headers  ([22ed5cf](https://github.com/NVIDIA/TRTorch/commit/22ed5cf))
* **//third_party/tensorrt:** Fix TensorRT paths for local x86 builds ([73d804b](https://github.com/NVIDIA/TRTorch/commit/73d804b))
* **aarch64:** fixes and issues for aarch64 toolchain ([9a6cccd](https://github.com/NVIDIA/TRTorch/commit/9a6cccd))
* **aten::_convolution:** out channels was passed in incorrectly for ([ee727f8](https://github.com/NVIDIA/TRTorch/commit/ee727f8))
* **aten::_convolution:** Pass dummy bias when there is no bias ([b20671c](https://github.com/NVIDIA/TRTorch/commit/b20671c))
* **aten::batch_norm:** A new batch norm implementation that hopefully ([6461872](https://github.com/NVIDIA/TRTorch/commit/6461872))
* **aten::batchnorm|aten::view:** Fix converter implementation for ([bf651dd](https://github.com/NVIDIA/TRTorch/commit/bf651dd))
* **aten::contiguous:** Blacklist aten::contiguous from conversion ([b718121](https://github.com/NVIDIA/TRTorch/commit/b718121))
* **aten::flatten:** Fixes dynamic shape for flatten ([4eb20bb](https://github.com/NVIDIA/TRTorch/commit/4eb20bb))
* fixed FP16 bug, fixed README, addressed some other PR comments ([d9c0e84](https://github.com/NVIDIA/TRTorch/commit/d9c0e84))
* **aten::neg:** Fix a index bug in neg ([1b2cde4](https://github.com/NVIDIA/TRTorch/commit/1b2cde4))
* **aten::size, other aten evaluators:** Removes aten::size converter in ([c83447e](https://github.com/NVIDIA/TRTorch/commit/c83447e))
* **BUILD:** modified BUILD ([a0d8586](https://github.com/NVIDIA/TRTorch/commit/a0d8586))
* trying to resolve interpolate plugin problems ([f0fefaa](https://github.com/NVIDIA/TRTorch/commit/f0fefaa))
* **core/conversion/converters/impl:** fix error message in interpolate ([5ddab8b](https://github.com/NVIDIA/TRTorch/commit/5ddab8b))
* Address issues in PR ([cd24f26](https://github.com/NVIDIA/TRTorch/commit/cd24f26))
* bypass jeykll, also add PR template ([a41c400](https://github.com/NVIDIA/TRTorch/commit/a41c400))
* first commit ([4f1a9df](https://github.com/NVIDIA/TRTorch/commit/4f1a9df))
* Fix pre CXX11 ABI python builds and regen docs ([42013ab](https://github.com/NVIDIA/TRTorch/commit/42013ab))
* fixed interpolate_plugin to handle dynamically sized inputs for adaptive_pool2d ([7794c78](https://github.com/NVIDIA/TRTorch/commit/7794c78))
* need to fix gather converter ([024a6b2](https://github.com/NVIDIA/TRTorch/commit/024a6b2))
* **plugin:** trying to fix bug in plugin ([cafcced](https://github.com/NVIDIA/TRTorch/commit/cafcced))
* **pooling:** fix the tests and the 1D pooling cases ([a90e6db](https://github.com/NVIDIA/TRTorch/commit/a90e6db))
* RunGraphEngineDynamic fixed to work with dynamically sized input tensors ([6308190](https://github.com/NVIDIA/TRTorch/commit/6308190))

### Features

* **//:libtrtorch:** Ship trtorchc with the tarball ([d647447](https://github.com/NVIDIA/TRTorch/commit/d647447))
* **//core/compiler:** Multiple outputs supported now via tuple ([f9af574](https://github.com/NVIDIA/TRTorch/commit/f9af574))
* **//core/conversion:** Adds the ability to evaluate loops ([dcb1474](https://github.com/NVIDIA/TRTorch/commit/dcb1474))
* **//core/conversion:** Compiler can now create graphs ([9d1946e](https://github.com/NVIDIA/TRTorch/commit/9d1946e))
* **//core/conversion:** Evaluation of static conditionals works now ([6421f3d](https://github.com/NVIDIA/TRTorch/commit/6421f3d))
* **//core/conversion/conversionctx:** Make op precision available at ([78a1c61](https://github.com/NVIDIA/TRTorch/commit/78a1c61))
* **//core/conversion/converters:** Throw a warning if a converter is ([6cce381](https://github.com/NVIDIA/TRTorch/commit/6cce381))
* **//core/conversion/converters/impl:** added support for aten::stack ([415378e](https://github.com/NVIDIA/TRTorch/commit/415378e))
* **//core/conversion/converters/impl:** added support for linear1d and bilinear2d ops ([4416d1f](https://github.com/NVIDIA/TRTorch/commit/4416d1f))
* **//core/conversion/converters/impl:** added support for trilinear3d op ([bb46e70](https://github.com/NVIDIA/TRTorch/commit/bb46e70))
* **//core/conversion/converters/impl:** all function schemas for upsample_nearest ([1b50484](https://github.com/NVIDIA/TRTorch/commit/1b50484))
* **//core/conversion/converters/impl:** logic implemented ([7f12160](https://github.com/NVIDIA/TRTorch/commit/7f12160))
* **//core/conversion/converters/impl:** Round out pooling ([7dc4af4](https://github.com/NVIDIA/TRTorch/commit/7dc4af4))
* **//core/conversion/converters/impl:** select converter, which adds support for aten::select.int ([5151c34](https://github.com/NVIDIA/TRTorch/commit/5151c34))
* **//core/conversion/converters/impl/plugins:** Created interpolate plugin, works for mode='linear' ([205ab99](https://github.com/NVIDIA/TRTorch/commit/205ab99))
* **//core/conversion/converters/impl/plugins:** interpolate plugin compiles now. time to test it. ([58dbaef](https://github.com/NVIDIA/TRTorch/commit/58dbaef))
* **//core/conversion/converters/impl/plugins:** template for interpolate plugin ([7c91dec](https://github.com/NVIDIA/TRTorch/commit/7c91dec))
* **//core/conversion/converters/impl/shuffle:** Implement aten::resize ([353f2d2](https://github.com/NVIDIA/TRTorch/commit/353f2d2))
* **//core/conversion/evaluators:** A whole bunch of new evaluators ([7466b8a](https://github.com/NVIDIA/TRTorch/commit/7466b8a))
* **//core/conversion/evaluators:** adding support for common evaluation ([d351717](https://github.com/NVIDIA/TRTorch/commit/d351717))
* **//core/conversion/evaluators:** Adds new applicability filters for ([2cc3226](https://github.com/NVIDIA/TRTorch/commit/2cc3226))
* **//core/conversion/evaluators:** Allow ITensors to be wrapped in ([619e345](https://github.com/NVIDIA/TRTorch/commit/619e345))
* **//core/execution:** Type checking for the executor, now is the ([2dd1ba3](https://github.com/NVIDIA/TRTorch/commit/2dd1ba3))
* **//core/lowering:** Add tuple lowering pass to remove tuples if ([ce6cf75](https://github.com/NVIDIA/TRTorch/commit/ce6cf75))
* **//core/lowering:** Adds peephole optimization pass ([0014b84](https://github.com/NVIDIA/TRTorch/commit/0014b84))
* **//core/lowering:** Fuse aten::addmm branches into a single ([68f0317](https://github.com/NVIDIA/TRTorch/commit/68f0317))
* **//core/lowering:** New freeze model pass and new exception ([4acc3fd](https://github.com/NVIDIA/TRTorch/commit/4acc3fd))
* **//core/lowering:** Remove aten::contiguous ([630b615](https://github.com/NVIDIA/TRTorch/commit/630b615))
* **//core/quantization:** skeleton of INT8 PTQ calibrator ([dd443a6](https://github.com/NVIDIA/TRTorch/commit/dd443a6))
* **//core/util:** New logging level for Graph Dumping ([90c44b9](https://github.com/NVIDIA/TRTorch/commit/90c44b9))
* **//cpp/api:** Adding max batch size setting ([1b25542](https://github.com/NVIDIA/TRTorch/commit/1b25542))
* **//cpp/api:** Functional Dataloader based PTQ ([f022dfe](https://github.com/NVIDIA/TRTorch/commit/f022dfe))
* **//cpp/api:** Remove the extra includes in the API header ([2f86f84](https://github.com/NVIDIA/TRTorch/commit/2f86f84))
* **//cpp/benchmark:** Increased workspace size for benchmark, may help ([8171f79](https://github.com/NVIDIA/TRTorch/commit/8171f79))
* **//cpp/ptq:** Add a feature to the dataset to use less than the full ([5f36f47](https://github.com/NVIDIA/TRTorch/commit/5f36f47))
* **//cpp/ptq:** do real benchmarking in the PTQ app instead of rough ([65e71c7](https://github.com/NVIDIA/TRTorch/commit/65e71c7))
* **//cpp/ptq/training:** Training recipe for VGG16 Classifier on ([676bf56](https://github.com/NVIDIA/TRTorch/commit/676bf56))
* **//cpp/trtorchc:** Adding a new CLI application for TRTorch which ([4f349a1](https://github.com/NVIDIA/TRTorch/commit/4f349a1))
* **//cpp/trtorchexec:** TRTorch exec now supports checking correctness ([80808b7](https://github.com/NVIDIA/TRTorch/commit/80808b7))
* **//lowering:** centralize lowering and try to use PyTorch Conv2DBN folding ([fad4a10](https://github.com/NVIDIA/TRTorch/commit/fad4a10))
* **//py:** add the option to build python package with CXX11 abi ([fdbd7d2](https://github.com/NVIDIA/TRTorch/commit/fdbd7d2))
* **//py:** API now produces valid engines that are consumable by ([72bc1f7](https://github.com/NVIDIA/TRTorch/commit/72bc1f7))
* **//py:** Inital introduction of the Python API ([7088245](https://github.com/NVIDIA/TRTorch/commit/7088245))
* **//py:** Manylinux container and build system for multiple python ([639c2a3](https://github.com/NVIDIA/TRTorch/commit/639c2a3))
* **//py:** register trtorch with torch op library to support ([736e914](https://github.com/NVIDIA/TRTorch/commit/736e914))
* **//py:** setup.py now searches for bazel executable ([737fe5c](https://github.com/NVIDIA/TRTorch/commit/737fe5c))
* **//py:** Working portable package ([482ef2c](https://github.com/NVIDIA/TRTorch/commit/482ef2c))
* added adaptive_avg_pool2d plugin, and added test for it ([fa227b0](https://github.com/NVIDIA/TRTorch/commit/fa227b0))
* **//tests:** New optional accuracy tests to check INT8 and FP16 ([df74136](https://github.com/NVIDIA/TRTorch/commit/df74136))
* **//toolchains:** Adding platform targets for supported platforms ([7889ebd](https://github.com/NVIDIA/TRTorch/commit/7889ebd))
* **/cpp/api:** Working INT8 Calibrator, also resolves [#41](https://github.com/NVIDIA/TRTorch/issues/41) ([5c0d737](https://github.com/NVIDIA/TRTorch/commit/5c0d737))
* **aten::add_t:** aten::add_.t evaluator that adds lists together ([c4c3ce1](https://github.com/NVIDIA/TRTorch/commit/c4c3ce1))
* **aten::avg_pool2d:** Implement Average Pooling 2D ([0c39519](https://github.com/NVIDIA/TRTorch/commit/0c39519))
* **aten::cat:** Implements aten::cat and completes support for SSD ([c2d3a6e](https://github.com/NVIDIA/TRTorch/commit/c2d3a6e))
* **aten::conv_transpose:** Add support for dilated and group ([48b950a](https://github.com/NVIDIA/TRTorch/commit/48b950a))
* **aten::dropout_:** Remove inplace dropout ([7aa57c3](https://github.com/NVIDIA/TRTorch/commit/7aa57c3))
* **aten::flatten:** Adds a converter for aten flatten since MM is the ([d945eb9](https://github.com/NVIDIA/TRTorch/commit/d945eb9))
* addressed some PR comments, refactored code ([141763f](https://github.com/NVIDIA/TRTorch/commit/141763f))
* **aten::matmul|aten::addmm:** Adds support for aten::matmul and ([c5b6202](https://github.com/NVIDIA/TRTorch/commit/c5b6202))
* **aten::permute:** Implement permute support ([c7d6b49](https://github.com/NVIDIA/TRTorch/commit/c7d6b49))
* **aten::size [static]:** Implement a aten::size converter for static input size ([0548540](https://github.com/NVIDIA/TRTorch/commit/0548540))
* started to work on add_.t evaluator, doesn't work yet ([f216d3f](https://github.com/NVIDIA/TRTorch/commit/f216d3f))
* **aten::to:** Remove remaining typecast operators (should be a very ([0f63ffa](https://github.com/NVIDIA/TRTorch/commit/0f63ffa))
* **aten::view:** Adds support for ATen view also fixes some tests ([24b422e](https://github.com/NVIDIA/TRTorch/commit/24b422e))
* **aten::zeros:** Implement aten::zeros evaluator ([670817c](https://github.com/NVIDIA/TRTorch/commit/670817c))
* **conv2d_to_convolution:** A pass to map aten::conv2d to _convolution ([2c5c0d5](https://github.com/NVIDIA/TRTorch/commit/2c5c0d5))
* **prim::NumToTensor:** Implement evaluator for NumToTensor ([60df888](https://github.com/NVIDIA/TRTorch/commit/60df888))
* **tests/util:** added RunGraphEngineDynamic to handle dynamic input sized tensors ([9458f21](https://github.com/NVIDIA/TRTorch/commit/9458f21))
* **trt_util:** from Naren, added unpadDims tool ([164a1a6](https://github.com/NVIDIA/TRTorch/commit/164a1a6))
* support for adaptive_avg_pool2d plugin ([52be580](https://github.com/NVIDIA/TRTorch/commit/52be580))
* Support non cxx11-abi builds for use in python api ([83e0ed6](https://github.com/NVIDIA/TRTorch/commit/83e0ed6))


### BREAKING CHANGES

* Bazel version is now locked to Bazel 3.3.1 and will be
bumped manually from now on. Builds will fail on all other versions
since now bazel will check the version before it compiles.

Documentation on how to install bazel is added as well to support
aarch64 until bazel releases binaries for the platform (which is soon)

Signed-off-by: Naren Dasan <naren@narendasan.com>
Signed-off-by: Naren Dasan <narens@nvidia.com>
* To use ptq you now need to include trtorch/ptq.h in
addition to trtorch/trtorch.h, similarly for logging commands you need
to include trtorch/logging.h

Signed-off-by: Naren Dasan <naren@narendasan.com>
Signed-off-by: Naren Dasan <narens@nvidia.com>


# v0.1.0 (2020-10-23)


### Bug Fixes

* added some fixes, trt/jit output still mismatches ([723ac1d](https://github.com/NVIDIA/TRTorch/commit/723ac1d))
* added test cases to explicitly check hidden/cell state outputs ([d7c3164](https://github.com/NVIDIA/TRTorch/commit/d7c3164))
* cleaned up logic, added case where bias doesn't exist for LSTM cell converter ([a3e1093](https://github.com/NVIDIA/TRTorch/commit/a3e1093))
* **//core/conversion/evaluator:** Custom to IValue that handles int[] ([68c934a](https://github.com/NVIDIA/TRTorch/commit/68c934a))
* **//docker:** Workaround only shared libraries being available in ([50c7eda](https://github.com/NVIDIA/TRTorch/commit/50c7eda))
* **//py:** Fix long description section of setup.py ([efd2099](https://github.com/NVIDIA/TRTorch/commit/efd2099))
* **//tests:** Add stride to complete tensors ([af5d28e](https://github.com/NVIDIA/TRTorch/commit/af5d28e))
* **//tests/accuracy:** Fix int8 accuracy test for new PTQ api ([a53bea7](https://github.com/NVIDIA/TRTorch/commit/a53bea7))
* **//tests/core/converters/activations:** Complete tensors in prelu test ([0e90f78](https://github.com/NVIDIA/TRTorch/commit/0e90f78))
* **docsrc:** Update docsrc container for bazel 3.4.1 ([4eb53b5](https://github.com/NVIDIA/TRTorch/commit/4eb53b5))


* fix(Windows)!: Fix dependency resolution for local builds ([858d8c3](https://github.com/NVIDIA/TRTorch/commit/858d8c3))
* chore!: Update dependencies to PyTorch 1.6.0 ([8eda27d](https://github.com/NVIDIA/TRTorch/commit/8eda27d))
* chore!: Bumping version numbers to 0.1.0 ([b84c90b](https://github.com/NVIDIA/TRTorch/commit/b84c90b))
* refactor(//core)!: Introducing a binding convention that will address ([5a105c6](https://github.com/NVIDIA/TRTorch/commit/5a105c6))
* refactor!: Renaming extra info to compile spec to be more consistent ([b8fa228](https://github.com/NVIDIA/TRTorch/commit/b8fa228))


### Features

* **//core/conversion/converters:** LSTMCell converter ([8c61248](https://github.com/NVIDIA/TRTorch/commit/8c61248))
* **//core/conversion/var:** created ITensorOrFreeze() method, to replace functionality of Var::ITensor() ([2ccf8d0](https://github.com/NVIDIA/TRTorch/commit/2ccf8d0))
* **//core/converters:** Add power layer conversion support and minor README edits ([a801506](https://github.com/NVIDIA/TRTorch/commit/a801506))
* **//core/lowering:** Add functionalization pass to replace implace ([90a9ed6](https://github.com/NVIDIA/TRTorch/commit/90a9ed6)), closes [#30](https://github.com/NVIDIA/TRTorch/issues/30)
* **//docker:** Adding CUDA11 based container for Ampere support ([970d775](https://github.com/NVIDIA/TRTorch/commit/970d775))
* started working on lstm_cell converter ([546d790](https://github.com/NVIDIA/TRTorch/commit/546d790))
* **//py:** Initial compiliant implementation of the to_backend api for ([59113cf](https://github.com/NVIDIA/TRTorch/commit/59113cf))
* **//third_party/tensorrt:** Add back TensorRT static lib in a cross ([d3c2e7e](https://github.com/NVIDIA/TRTorch/commit/d3c2e7e))
* **aten::prelu:** Basic prelu support ([8bc4369](https://github.com/NVIDIA/TRTorch/commit/8bc4369))
* **aten::prelu:** Implement the multi-channel version of prelu and ([c066581](https://github.com/NVIDIA/TRTorch/commit/c066581))
* finished logic for LSTM cell, now to test ([a88cfaf](https://github.com/NVIDIA/TRTorch/commit/a88cfaf))


### BREAKING CHANGES

* Users on Windows trying to use cuDNN 8 must manually
configure third_party/cudnn/local/BUILD to use cuDNN 8.

Signed-off-by: Naren Dasan <naren@narendasan.com>
Signed-off-by: Naren Dasan <narens@nvidia.com>
* Support for Python 3.5 is being dropped with this
update

Signed-off-by: Naren Dasan <naren@narendasan.com>
Signed-off-by: Naren Dasan <narens@nvidia.com>
* Version is being bumped to version 0.1.0a0 to target
PyTorch 1.6.0

Signed-off-by: Naren Dasan <naren@narendasan.com>
Signed-off-by: Naren Dasan <narens@nvidia.com>
* This changes the "ABI" of compiled TRTorch programs and
the runtime and breaks backwards compatability between the runtime in
0.1.0+ and programs compiled pre-0.1.0

Signed-off-by: Naren Dasan <naren@narendasan.com>
Signed-off-by: Naren Dasan <narens@nvidia.com>
* This changes the top level api for setting the
specification for compilation, a simple find and replace should allow
users to port forward

Signed-off-by: Naren Dasan <naren@narendasan.com>
Signed-off-by: Naren Dasan <narens@nvidia.com>



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



## 0.0.3 (2020-07-18)


* feat!: Lock bazel version ([25f4371](https://github.com/NVIDIA/TRTorch/commit/25f4371))
* refactor(//cpp/api)!: Refactoring ptq to use includes but seperate from ([d2f8a59](https://github.com/NVIDIA/TRTorch/commit/d2f8a59))


### Bug Fixes

* Address issues in PR ([cd24f26](https://github.com/NVIDIA/TRTorch/commit/cd24f26))
* bypass jeykll, also add PR template ([a41c400](https://github.com/NVIDIA/TRTorch/commit/a41c400))
* first commit ([4f1a9df](https://github.com/NVIDIA/TRTorch/commit/4f1a9df))
* Fix pre CXX11 ABI python builds and regen docs ([42013ab](https://github.com/NVIDIA/TRTorch/commit/42013ab))
* fixed FP16 bug, fixed README, addressed some other PR comments ([d9c0e84](https://github.com/NVIDIA/TRTorch/commit/d9c0e84))
* fixed interpolate_plugin to handle dynamically sized inputs for adaptive_pool2d ([7794c78](https://github.com/NVIDIA/TRTorch/commit/7794c78))
* need to fix gather converter ([024a6b2](https://github.com/NVIDIA/TRTorch/commit/024a6b2))
* Remove line no longer necessary in wheel builder dockerfile ([fe06d09](https://github.com/NVIDIA/TRTorch/commit/fe06d09))
* **//core:** Do not compile hidden methods ([6bd1a3f](https://github.com/NVIDIA/TRTorch/commit/6bd1a3f))
* **//core/conversion:** Check for calibrator before setting int8 mode ([3afd209](https://github.com/NVIDIA/TRTorch/commit/3afd209))
* **//core/conversion:** Supress unnecessary debug messages ([2b23874](https://github.com/NVIDIA/TRTorch/commit/2b23874))
* **//core/conversion/conversionctx:** Check both tensor and eval maps ([2d65ece](https://github.com/NVIDIA/TRTorch/commit/2d65ece))
* **//core/conversion/conversionctx:** In the case of strict types and ([3611778](https://github.com/NVIDIA/TRTorch/commit/3611778))
* **//core/conversion/converters:** Fix plugin implementation for TRT 7 ([94d6a0f](https://github.com/NVIDIA/TRTorch/commit/94d6a0f))
* **//core/conversion/converters/impl:** 1d case not working ([f42562b](https://github.com/NVIDIA/TRTorch/commit/f42562b))
* **//core/conversion/converters/impl:** code works for interpolate2d/3d, doesn't work for 1d yet ([e4cb117](https://github.com/NVIDIA/TRTorch/commit/e4cb117))
* **//core/conversion/converters/impl:** Fix interpolate.cpp ([b6942a2](https://github.com/NVIDIA/TRTorch/commit/b6942a2))
* **//core/conversion/converters/impl/element_wise:** Fix broadcast ([a9f33e4](https://github.com/NVIDIA/TRTorch/commit/a9f33e4))
* **//core/conversion/evaluators:** A couple fixes for evaluators ([07ba980](https://github.com/NVIDIA/TRTorch/commit/07ba980))
* **//core/lowering:** Conv2D -> _convolution pass was triggering conv ([ca2b5f9](https://github.com/NVIDIA/TRTorch/commit/ca2b5f9))
* **//cpp:** Remove deprecated script namespace ([d70760f](https://github.com/NVIDIA/TRTorch/commit/d70760f))
* **//cpp/api:** Better inital condition for the dataloader iterator to ([8d22bdd](https://github.com/NVIDIA/TRTorch/commit/8d22bdd))
* **//cpp/api:** Remove unecessary destructor in ptq class ([fc70267](https://github.com/NVIDIA/TRTorch/commit/fc70267))
* **//cpp/api:** set a default for calibrator ([825be69](https://github.com/NVIDIA/TRTorch/commit/825be69))
* **//cpp/benchmark:** reorder benchmark so FP16 bn issue in JIT doesnt ([98527d2](https://github.com/NVIDIA/TRTorch/commit/98527d2))
* **//cpp/ptq:** Default version of the app should not resize images ([de3cbc4](https://github.com/NVIDIA/TRTorch/commit/de3cbc4))
* **//cpp/ptq:** Enable FP16 kernels for INT8 applications ([26709cc](https://github.com/NVIDIA/TRTorch/commit/26709cc))
* **//cpp/ptq:** Enable FP16 kernels for INT8 applications ([e1c5416](https://github.com/NVIDIA/TRTorch/commit/e1c5416))
* **//cpp/ptq:** remove some logging from ptq app ([b989c7f](https://github.com/NVIDIA/TRTorch/commit/b989c7f))
* **//cpp/ptq:** Tracing model in eval mode wrecks accuracy in Libtorch ([54a24b3](https://github.com/NVIDIA/TRTorch/commit/54a24b3))
* **//cpp/trtorchc:** Refactor trtorchc to use new C++ API ([789e1be](https://github.com/NVIDIA/TRTorch/commit/789e1be)), closes [#132](https://github.com/NVIDIA/TRTorch/issues/132)
* **//cpp/trtorchc:** Support building trtorchc with the pre_cxx11_abi ([172d4d5](https://github.com/NVIDIA/TRTorch/commit/172d4d5))
* **//docs:** add nojekyll file ([2a02cd5](https://github.com/NVIDIA/TRTorch/commit/2a02cd5))
* **//docs:** fix version links ([11555f7](https://github.com/NVIDIA/TRTorch/commit/11555f7))
* **//notebooks:** Fix WORKSPACE template file to reflect new build system layout ([c8ea9b7](https://github.com/NVIDIA/TRTorch/commit/c8ea9b7))
* **//py:** Build system issues ([c1de126](https://github.com/NVIDIA/TRTorch/commit/c1de126))
* **//py:** Ignore generated version file ([9e37dc1](https://github.com/NVIDIA/TRTorch/commit/9e37dc1))
* **//py:** Lib path incorrect ([ff2b13c](https://github.com/NVIDIA/TRTorch/commit/ff2b13c))
* **//tests:** Duplicated tensorrt dep ([5cd697e](https://github.com/NVIDIA/TRTorch/commit/5cd697e))
* **//third_party/tensorrt:** Fix include dir for library headers  ([22ed5cf](https://github.com/NVIDIA/TRTorch/commit/22ed5cf))
* **//third_party/tensorrt:** Fix TensorRT paths for local x86 builds ([73d804b](https://github.com/NVIDIA/TRTorch/commit/73d804b))
* **aarch64:** fixes and issues for aarch64 toolchain ([9a6cccd](https://github.com/NVIDIA/TRTorch/commit/9a6cccd))
* **aten::_convolution:** out channels was passed in incorrectly for ([ee727f8](https://github.com/NVIDIA/TRTorch/commit/ee727f8))
* **aten::_convolution:** Pass dummy bias when there is no bias ([b20671c](https://github.com/NVIDIA/TRTorch/commit/b20671c))
* **aten::batch_norm:** A new batch norm implementation that hopefully ([6461872](https://github.com/NVIDIA/TRTorch/commit/6461872))
* **aten::batchnorm|aten::view:** Fix converter implementation for ([bf651dd](https://github.com/NVIDIA/TRTorch/commit/bf651dd))
* **aten::contiguous:** Blacklist aten::contiguous from conversion ([b718121](https://github.com/NVIDIA/TRTorch/commit/b718121))
* **aten::flatten:** Fixes dynamic shape for flatten ([4eb20bb](https://github.com/NVIDIA/TRTorch/commit/4eb20bb))
* **aten::neg:** Fix a index bug in neg ([1b2cde4](https://github.com/NVIDIA/TRTorch/commit/1b2cde4))
* **aten::size, other aten evaluators:** Removes aten::size converter in ([c83447e](https://github.com/NVIDIA/TRTorch/commit/c83447e))
* **BUILD:** modified BUILD ([a0d8586](https://github.com/NVIDIA/TRTorch/commit/a0d8586))
* **core/conversion/converters/impl:** fix error message in interpolate ([5ddab8b](https://github.com/NVIDIA/TRTorch/commit/5ddab8b))
* **plugin:** trying to fix bug in plugin ([cafcced](https://github.com/NVIDIA/TRTorch/commit/cafcced))
* **pooling:** fix the tests and the 1D pooling cases ([a90e6db](https://github.com/NVIDIA/TRTorch/commit/a90e6db))
* RunGraphEngineDynamic fixed to work with dynamically sized input tensors ([6308190](https://github.com/NVIDIA/TRTorch/commit/6308190))
* trying to resolve interpolate plugin problems ([f0fefaa](https://github.com/NVIDIA/TRTorch/commit/f0fefaa))


### Features

* **//:libtrtorch:** Ship trtorchc with the tarball ([d647447](https://github.com/NVIDIA/TRTorch/commit/d647447))
* **//core/compiler:** Multiple outputs supported now via tuple ([f9af574](https://github.com/NVIDIA/TRTorch/commit/f9af574))
* **//core/conversion:** Adds the ability to evaluate loops ([dcb1474](https://github.com/NVIDIA/TRTorch/commit/dcb1474))
* **//core/conversion:** Compiler can now create graphs ([9d1946e](https://github.com/NVIDIA/TRTorch/commit/9d1946e))
* **//core/conversion:** Evaluation of static conditionals works now ([6421f3d](https://github.com/NVIDIA/TRTorch/commit/6421f3d))
* **//core/conversion/conversionctx:** Make op precision available at ([78a1c61](https://github.com/NVIDIA/TRTorch/commit/78a1c61))
* **//core/conversion/converters:** Throw a warning if a converter is ([6cce381](https://github.com/NVIDIA/TRTorch/commit/6cce381))
* **//core/conversion/converters/impl:** added support for aten::stack ([415378e](https://github.com/NVIDIA/TRTorch/commit/415378e))
* **//core/conversion/converters/impl:** added support for linear1d and bilinear2d ops ([4416d1f](https://github.com/NVIDIA/TRTorch/commit/4416d1f))
* **//core/conversion/converters/impl:** added support for trilinear3d op ([bb46e70](https://github.com/NVIDIA/TRTorch/commit/bb46e70))
* **//core/conversion/converters/impl:** all function schemas for upsample_nearest ([1b50484](https://github.com/NVIDIA/TRTorch/commit/1b50484))
* **//core/conversion/converters/impl:** logic implemented ([7f12160](https://github.com/NVIDIA/TRTorch/commit/7f12160))
* **//core/conversion/converters/impl:** Round out pooling ([7dc4af4](https://github.com/NVIDIA/TRTorch/commit/7dc4af4))
* **//core/conversion/converters/impl:** select converter, which adds support for aten::select.int ([5151c34](https://github.com/NVIDIA/TRTorch/commit/5151c34))
* **//core/conversion/converters/impl/plugins:** Created interpolate plugin, works for mode='linear' ([205ab99](https://github.com/NVIDIA/TRTorch/commit/205ab99))
* **//core/conversion/converters/impl/plugins:** interpolate plugin compiles now. time to test it. ([58dbaef](https://github.com/NVIDIA/TRTorch/commit/58dbaef))
* **//core/conversion/converters/impl/plugins:** template for interpolate plugin ([7c91dec](https://github.com/NVIDIA/TRTorch/commit/7c91dec))
* **//core/conversion/converters/impl/shuffle:** Implement aten::resize ([353f2d2](https://github.com/NVIDIA/TRTorch/commit/353f2d2))
* **//core/conversion/evaluators:** A whole bunch of new evaluators ([7466b8a](https://github.com/NVIDIA/TRTorch/commit/7466b8a))
* **//core/conversion/evaluators:** adding support for common evaluation ([d351717](https://github.com/NVIDIA/TRTorch/commit/d351717))
* **//core/conversion/evaluators:** Adds new applicability filters for ([2cc3226](https://github.com/NVIDIA/TRTorch/commit/2cc3226))
* **//core/conversion/evaluators:** Allow ITensors to be wrapped in ([619e345](https://github.com/NVIDIA/TRTorch/commit/619e345))
* **//core/execution:** Type checking for the executor, now is the ([2dd1ba3](https://github.com/NVIDIA/TRTorch/commit/2dd1ba3))
* **//core/lowering:** Add tuple lowering pass to remove tuples if ([ce6cf75](https://github.com/NVIDIA/TRTorch/commit/ce6cf75))
* **//core/lowering:** Adds peephole optimization pass ([0014b84](https://github.com/NVIDIA/TRTorch/commit/0014b84))
* **//core/lowering:** Fuse aten::addmm branches into a single ([68f0317](https://github.com/NVIDIA/TRTorch/commit/68f0317))
* **//core/lowering:** New freeze model pass and new exception ([4acc3fd](https://github.com/NVIDIA/TRTorch/commit/4acc3fd))
* **//core/lowering:** Remove aten::contiguous ([630b615](https://github.com/NVIDIA/TRTorch/commit/630b615))
* **//core/quantization:** skeleton of INT8 PTQ calibrator ([dd443a6](https://github.com/NVIDIA/TRTorch/commit/dd443a6))
* **//core/util:** New logging level for Graph Dumping ([90c44b9](https://github.com/NVIDIA/TRTorch/commit/90c44b9))
* **//cpp/api:** Adding max batch size setting ([1b25542](https://github.com/NVIDIA/TRTorch/commit/1b25542))
* **//cpp/api:** Functional Dataloader based PTQ ([f022dfe](https://github.com/NVIDIA/TRTorch/commit/f022dfe))
* **//cpp/api:** Remove the extra includes in the API header ([2f86f84](https://github.com/NVIDIA/TRTorch/commit/2f86f84))
* **//cpp/benchmark:** Increased workspace size for benchmark, may help ([8171f79](https://github.com/NVIDIA/TRTorch/commit/8171f79))
* **//cpp/ptq:** Add a feature to the dataset to use less than the full ([5f36f47](https://github.com/NVIDIA/TRTorch/commit/5f36f47))
* **//cpp/ptq:** do real benchmarking in the PTQ app instead of rough ([65e71c7](https://github.com/NVIDIA/TRTorch/commit/65e71c7))
* **//cpp/ptq/training:** Training recipe for VGG16 Classifier on ([676bf56](https://github.com/NVIDIA/TRTorch/commit/676bf56))
* **//cpp/trtorchc:** Adding a new CLI application for TRTorch which ([4f349a1](https://github.com/NVIDIA/TRTorch/commit/4f349a1))
* **//cpp/trtorchexec:** TRTorch exec now supports checking correctness ([80808b7](https://github.com/NVIDIA/TRTorch/commit/80808b7))
* **//lowering:** centralize lowering and try to use PyTorch Conv2DBN folding ([fad4a10](https://github.com/NVIDIA/TRTorch/commit/fad4a10))
* **//py:** add the option to build python package with CXX11 abi ([fdbd7d2](https://github.com/NVIDIA/TRTorch/commit/fdbd7d2))
* **//py:** API now produces valid engines that are consumable by ([72bc1f7](https://github.com/NVIDIA/TRTorch/commit/72bc1f7))
* **//py:** Inital introduction of the Python API ([7088245](https://github.com/NVIDIA/TRTorch/commit/7088245))
* **//py:** Manylinux container and build system for multiple python ([639c2a3](https://github.com/NVIDIA/TRTorch/commit/639c2a3))
* **//py:** register trtorch with torch op library to support ([736e914](https://github.com/NVIDIA/TRTorch/commit/736e914))
* **//py:** setup.py now searches for bazel executable ([737fe5c](https://github.com/NVIDIA/TRTorch/commit/737fe5c))
* **//py:** Working portable package ([482ef2c](https://github.com/NVIDIA/TRTorch/commit/482ef2c))
* added adaptive_avg_pool2d plugin, and added test for it ([fa227b0](https://github.com/NVIDIA/TRTorch/commit/fa227b0))
* **//tests:** New optional accuracy tests to check INT8 and FP16 ([df74136](https://github.com/NVIDIA/TRTorch/commit/df74136))
* **//toolchains:** Adding platform targets for supported platforms ([7889ebd](https://github.com/NVIDIA/TRTorch/commit/7889ebd))
* **/cpp/api:** Working INT8 Calibrator, also resolves [#41](https://github.com/NVIDIA/TRTorch/issues/41) ([5c0d737](https://github.com/NVIDIA/TRTorch/commit/5c0d737))
* **aten::add_t:** aten::add_.t evaluator that adds lists together ([c4c3ce1](https://github.com/NVIDIA/TRTorch/commit/c4c3ce1))
* **aten::avg_pool2d:** Implement Average Pooling 2D ([0c39519](https://github.com/NVIDIA/TRTorch/commit/0c39519))
* **aten::cat:** Implements aten::cat and completes support for SSD ([c2d3a6e](https://github.com/NVIDIA/TRTorch/commit/c2d3a6e))
* **aten::conv_transpose:** Add support for dilated and group ([48b950a](https://github.com/NVIDIA/TRTorch/commit/48b950a))
* **aten::dropout_:** Remove inplace dropout ([7aa57c3](https://github.com/NVIDIA/TRTorch/commit/7aa57c3))
* **aten::flatten:** Adds a converter for aten flatten since MM is the ([d945eb9](https://github.com/NVIDIA/TRTorch/commit/d945eb9))
* addressed some PR comments, refactored code ([141763f](https://github.com/NVIDIA/TRTorch/commit/141763f))
* **aten::matmul|aten::addmm:** Adds support for aten::matmul and ([c5b6202](https://github.com/NVIDIA/TRTorch/commit/c5b6202))
* **aten::permute:** Implement permute support ([c7d6b49](https://github.com/NVIDIA/TRTorch/commit/c7d6b49))
* **aten::size [static]:** Implement a aten::size converter for static input size ([0548540](https://github.com/NVIDIA/TRTorch/commit/0548540))
* started to work on add_.t evaluator, doesn't work yet ([f216d3f](https://github.com/NVIDIA/TRTorch/commit/f216d3f))
* **aten::to:** Remove remaining typecast operators (should be a very ([0f63ffa](https://github.com/NVIDIA/TRTorch/commit/0f63ffa))
* **aten::view:** Adds support for ATen view also fixes some tests ([24b422e](https://github.com/NVIDIA/TRTorch/commit/24b422e))
* **aten::zeros:** Implement aten::zeros evaluator ([670817c](https://github.com/NVIDIA/TRTorch/commit/670817c))
* **conv2d_to_convolution:** A pass to map aten::conv2d to _convolution ([2c5c0d5](https://github.com/NVIDIA/TRTorch/commit/2c5c0d5))
* **prim::NumToTensor:** Implement evaluator for NumToTensor ([60df888](https://github.com/NVIDIA/TRTorch/commit/60df888))
* **tests/util:** added RunGraphEngineDynamic to handle dynamic input sized tensors ([9458f21](https://github.com/NVIDIA/TRTorch/commit/9458f21))
* **trt_util:** from Naren, added unpadDims tool ([164a1a6](https://github.com/NVIDIA/TRTorch/commit/164a1a6))
* support for adaptive_avg_pool2d plugin ([52be580](https://github.com/NVIDIA/TRTorch/commit/52be580))
* Support non cxx11-abi builds for use in python api ([83e0ed6](https://github.com/NVIDIA/TRTorch/commit/83e0ed6))


### BREAKING CHANGES

* Bazel version is now locked to Bazel 3.2.0 and will be
bumped manually from now on. Builds will fail on all other versions
since now bazel will check the version before it compiles.

Documentation on how to install bazel is added as well to support
aarch64 until bazel releases binaries for the platform (which is soon)

Signed-off-by: Naren Dasan <naren@narendasan.com>
Signed-off-by: Naren Dasan <narens@nvidia.com>
* To use ptq you now need to include trtorch/ptq.h in
addition to trtorch/trtorch.h, similarly for logging commands you need
to include trtorch/logging.h

Signed-off-by: Naren Dasan <naren@narendasan.com>
Signed-off-by: Naren Dasan <narens@nvidia.com>



# 0.1.0 (2020-10-23)


### Bug Fixes

* cleaned up logic, added case where bias doesn't exist for LSTM cell converter ([a3e1093](https://github.com/NVIDIA/TRTorch/commit/a3e1093))
* **//core:** Remove old backend.h from core includes ([85a4847](https://github.com/NVIDIA/TRTorch/commit/85a4847))
* added some fixes, trt/jit output still mismatches ([723ac1d](https://github.com/NVIDIA/TRTorch/commit/723ac1d))
* added test cases to explicitly check hidden/cell state outputs ([d7c3164](https://github.com/NVIDIA/TRTorch/commit/d7c3164))
* **//core/conversion/evaluator:** Custom to IValue that handles int[] ([68c934a](https://github.com/NVIDIA/TRTorch/commit/68c934a))
* **//docker:** Workaround only shared libraries being available in ([50c7eda](https://github.com/NVIDIA/TRTorch/commit/50c7eda))
* **//py:** Fix long description section of setup.py ([efd2099](https://github.com/NVIDIA/TRTorch/commit/efd2099))
* **//tests:** Add stride to complete tensors ([af5d28e](https://github.com/NVIDIA/TRTorch/commit/af5d28e))
* **//tests/accuracy:** Fix int8 accuracy test for new PTQ api ([a53bea7](https://github.com/NVIDIA/TRTorch/commit/a53bea7))
* **//tests/core/converters/activations:** Complete tensors in prelu test ([0e90f78](https://github.com/NVIDIA/TRTorch/commit/0e90f78))
* **docsrc:** Update docsrc container for bazel 3.4.1 ([4eb53b5](https://github.com/NVIDIA/TRTorch/commit/4eb53b5))


* fix(Windows)!: Fix dependency resolution for local builds ([858d8c3](https://github.com/NVIDIA/TRTorch/commit/858d8c3))
* chore!: Update dependencies to PyTorch 1.6.0 ([8eda27d](https://github.com/NVIDIA/TRTorch/commit/8eda27d))
* chore!: Bumping version numbers to 0.1.0 ([b84c90b](https://github.com/NVIDIA/TRTorch/commit/b84c90b))
* refactor(//core)!: Introducing a binding convention that will address ([5a105c6](https://github.com/NVIDIA/TRTorch/commit/5a105c6))
* refactor!: Renaming extra info to compile spec to be more consistent ([b8fa228](https://github.com/NVIDIA/TRTorch/commit/b8fa228))


### Features

* **//core/conversion/converters:** LSTMCell converter ([8c61248](https://github.com/NVIDIA/TRTorch/commit/8c61248))
* **//core/conversion/var:** created ITensorOrFreeze() method, to replace functionality of Var::ITensor() ([2ccf8d0](https://github.com/NVIDIA/TRTorch/commit/2ccf8d0))
* **//core/converters:** Add power layer conversion support and minor README edits ([a801506](https://github.com/NVIDIA/TRTorch/commit/a801506))
* **//core/lowering:** Add functionalization pass to replace implace ([90a9ed6](https://github.com/NVIDIA/TRTorch/commit/90a9ed6)), closes [#30](https://github.com/NVIDIA/TRTorch/issues/30)
* **//docker:** Adding CUDA11 based container for Ampere support ([970d775](https://github.com/NVIDIA/TRTorch/commit/970d775))
* started working on lstm_cell converter ([546d790](https://github.com/NVIDIA/TRTorch/commit/546d790))
* **//py:** Initial compiliant implementation of the to_backend api for ([59113cf](https://github.com/NVIDIA/TRTorch/commit/59113cf))
* **//third_party/tensorrt:** Add back TensorRT static lib in a cross ([d3c2e7e](https://github.com/NVIDIA/TRTorch/commit/d3c2e7e))
* **aten::prelu:** Basic prelu support ([8bc4369](https://github.com/NVIDIA/TRTorch/commit/8bc4369))
* **aten::prelu:** Implement the multi-channel version of prelu and ([c066581](https://github.com/NVIDIA/TRTorch/commit/c066581))
* finished logic for LSTM cell, now to test ([a88cfaf](https://github.com/NVIDIA/TRTorch/commit/a88cfaf))


### BREAKING CHANGES

* Users on Windows trying to use cuDNN 8 must manually
configure third_party/cudnn/local/BUILD to use cuDNN 8.

Signed-off-by: Naren Dasan <naren@narendasan.com>
Signed-off-by: Naren Dasan <narens@nvidia.com>
* Support for Python 3.5 is being dropped with this
update

Signed-off-by: Naren Dasan <naren@narendasan.com>
Signed-off-by: Naren Dasan <narens@nvidia.com>
* Version is being bumped to version 0.1.0a0 to target
PyTorch 1.6.0

Signed-off-by: Naren Dasan <naren@narendasan.com>
Signed-off-by: Naren Dasan <narens@nvidia.com>
* This changes the "ABI" of compiled TRTorch programs and
the runtime and breaks backwards compatability between the runtime in
0.1.0+ and programs compiled pre-0.1.0

Signed-off-by: Naren Dasan <naren@narendasan.com>
Signed-off-by: Naren Dasan <narens@nvidia.com>
* This changes the top level api for setting the
specification for compilation, a simple find and replace should allow
users to port forward

Signed-off-by: Naren Dasan <naren@narendasan.com>
Signed-off-by: Naren Dasan <narens@nvidia.com>



#  (2021-02-25)


* refactor!: Update bazel and trt versions ([0618b6b](https://github.com/NVIDIA/TRTorch/commit/0618b6b))


### Bug Fixes

* **//core/conversion/conversionctx:** Fix memory leak in conversion ([6f83b41](https://github.com/NVIDIA/TRTorch/commit/6f83b41))
* **//core/lowering:** fix debug message for bn dim check removal pass ([86bb5b7](https://github.com/NVIDIA/TRTorch/commit/86bb5b7))
* **//py:** Fix bounds for enum macros ([6b942e5](https://github.com/NVIDIA/TRTorch/commit/6b942e5))
* **aten::expand:** Fix compiler warning for unused out ITensor ([5b0f584](https://github.com/NVIDIA/TRTorch/commit/5b0f584))
* **aten::expand:** Fix compiler warnings in the expand converter ([51b09d4](https://github.com/NVIDIA/TRTorch/commit/51b09d4))
* **aten::flatten:** Fixing flatten converter to handle dynamic batch ([00f2d78](https://github.com/NVIDIA/TRTorch/commit/00f2d78))
* **aten::max_pool2d:** Supressing error due to not filling in stride in ([ed3c185](https://github.com/NVIDIA/TRTorch/commit/ed3c185))
* **aten::zeros:** verify zeros produces a tensor correctly ([00d2d0c](https://github.com/NVIDIA/TRTorch/commit/00d2d0c))
* **remove_to:** bug in remove_to.cpp, replace outputs()[0] with inputs()[0] ([6c5118a](https://github.com/NVIDIA/TRTorch/commit/6c5118a))
* **setup.py:** Broaden the supported pytorch versions to handle jetson ([e94a040](https://github.com/NVIDIA/TRTorch/commit/e94a040))
* **test_op_aliasing:** Fix the renamed op ([91c3c80](https://github.com/NVIDIA/TRTorch/commit/91c3c80))
* **tests:** Fix broken elementwise tests ([22ed944](https://github.com/NVIDIA/TRTorch/commit/22ed944))


### Features

* support true_divide, floor_divide, max, min, rsub ([a35fbf1](https://github.com/NVIDIA/TRTorch/commit/a35fbf1))
* **//.github:** Moving to python directly ([ece114c](https://github.com/NVIDIA/TRTorch/commit/ece114c))
* **//core/conversion:** Adding a check to detect programs that will ([a3d4144](https://github.com/NVIDIA/TRTorch/commit/a3d4144))
* **//core/lowering:** Adding a new pass to handle new dim checks for ([3d14cda](https://github.com/NVIDIA/TRTorch/commit/3d14cda))
* **//cpp/api/lib:** New runtime only library ([6644a9e](https://github.com/NVIDIA/TRTorch/commit/6644a9e))
* **//notebooks:** Update notebooks container for 0.1.0 ([a5851ff](https://github.com/NVIDIA/TRTorch/commit/a5851ff))
* **//py:** [to_backend] adding device specification support for ([6eeba1c](https://github.com/NVIDIA/TRTorch/commit/6eeba1c)), closes [#286](https://github.com/NVIDIA/TRTorch/issues/286)
* **aten::leaky_relu_:** Adding alias for inplace leaky relu ([bc53411](https://github.com/NVIDIA/TRTorch/commit/bc53411))
* **aten::softmax:** Adding support for any neg index ([abc29a2](https://github.com/NVIDIA/TRTorch/commit/abc29a2))
* **aten::squeeze|aten::unsqueeze:** adding BUILD files for new squeeze ([9e0a1d7](https://github.com/NVIDIA/TRTorch/commit/9e0a1d7))
* **aten::sum:** Allow for negative indices less than -1 ([769bbc9](https://github.com/NVIDIA/TRTorch/commit/769bbc9))
* **aten::topk:** Add a debug message noting that sorted is always true ([81f1e9d](https://github.com/NVIDIA/TRTorch/commit/81f1e9d))
* **aten::topk:** Adding BUILD files for topk op ([22e6a6b](https://github.com/NVIDIA/TRTorch/commit/22e6a6b))
* **disable_tf32:** Add a new API to disable TF32 ([536983b](https://github.com/NVIDIA/TRTorch/commit/536983b))
* **interpolate:** Adding support for .vec variants and overhauling test ([0cda1cc](https://github.com/NVIDIA/TRTorch/commit/0cda1cc))
* **interpolate:** Addressing the linear, scale factor, align corners edge case ([92e3818](https://github.com/NVIDIA/TRTorch/commit/92e3818))
* **supportedops:** Application to dump a list of supported operators ([872d9a3](https://github.com/NVIDIA/TRTorch/commit/872d9a3))


### BREAKING CHANGES

* Version of bazel has been bumped to 4.0.0
Version of TensorRT has been bumped to 7.2.2.3

Signed-off-by: Naren Dasan <naren@narendasan.com>
Signed-off-by: Naren Dasan <narens@nvidia.com>



