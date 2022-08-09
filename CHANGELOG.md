# Changelog


# 0.0.1 (2020-03-31)


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

# 0.0.2  (2020-05-17)


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


# 0.0.3 (2020-07-18)


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


# 0.1.0 (2020-10-23)


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



#  0.2.0 (2021-02-25)


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


# 0.3.0 (2021-05-13)


### Bug Fixes

* **//plugins:** Readding cuBLAS BUILD to allow linking of libnvinfer_plugin on Jetson ([a8008f4](https://github.com/NVIDIA/TRTorch/commit/a8008f4))
* **//tests/../concat:** Concat test fix ([2432fb8](https://github.com/NVIDIA/TRTorch/commit/2432fb8))
* **//tests/core/partitioning:** Fixing some issues with the partition ([ff89059](https://github.com/NVIDIA/TRTorch/commit/ff89059))
* erase the repetitive nodes in dependency analysis ([80b1038](https://github.com/NVIDIA/TRTorch/commit/80b1038))
* fix a typo for debug ([c823ebd](https://github.com/NVIDIA/TRTorch/commit/c823ebd))
* fix typo bug ([e491bb5](https://github.com/NVIDIA/TRTorch/commit/e491bb5))
* **aten::linear:** Fixes new issues in 1.8 that cause script based ([c5057f8](https://github.com/NVIDIA/TRTorch/commit/c5057f8))
* register the torch_fallback attribute in Python API ([8b7919f](https://github.com/NVIDIA/TRTorch/commit/8b7919f))
* support expand/repeat with IValue type input ([a4882c6](https://github.com/NVIDIA/TRTorch/commit/a4882c6))
* support shape inference for add_, support non-tensor arguments for segmented graphs ([46950bb](https://github.com/NVIDIA/TRTorch/commit/46950bb))


* feat!: Updating versions of CUDA, cuDNN, TensorRT and PyTorch ([71c4dcb](https://github.com/NVIDIA/TRTorch/commit/71c4dcb))
* feat(WORKSPACE)!: Updating PyTorch version to 1.8.1 ([c9aa99a](https://github.com/NVIDIA/TRTorch/commit/c9aa99a))


### Features

* **//.github:** Linter throws 1 when there needs to be style changes to ([a39dea7](https://github.com/NVIDIA/TRTorch/commit/a39dea7))
* **//core:** New API to register arbitrary TRT engines in TorchScript ([3ec836e](https://github.com/NVIDIA/TRTorch/commit/3ec836e))
* **//core/conversion/conversionctx:** Adding logging for truncated ([96245ee](https://github.com/NVIDIA/TRTorch/commit/96245ee))
* **//core/partitioing:** Adding ostream for Partition Info ([b3589c5](https://github.com/NVIDIA/TRTorch/commit/b3589c5))
* **//core/partitioning:** Add an ostream implementation for ([ee536b6](https://github.com/NVIDIA/TRTorch/commit/ee536b6))
* **//core/partitioning:** Refactor top level partitioning API, fix a bug with ([abc63f6](https://github.com/NVIDIA/TRTorch/commit/abc63f6))
* **//core/plugins:** Gating plugin logging based on global config ([1d5a088](https://github.com/NVIDIA/TRTorch/commit/1d5a088))
* added user level API for fallback ([f4c29b4](https://github.com/NVIDIA/TRTorch/commit/f4c29b4))
* allow users to set fallback block size and ops ([6d3064a](https://github.com/NVIDIA/TRTorch/commit/6d3064a))
* insert nodes by dependencies for nonTensor inputs/outputs ([4e32eff](https://github.com/NVIDIA/TRTorch/commit/4e32eff))
* support aten::arange converter ([014e381](https://github.com/NVIDIA/TRTorch/commit/014e381))
* support aten::transpose with negative dim ([4a1d2f3](https://github.com/NVIDIA/TRTorch/commit/4a1d2f3))
* support Int/Bool and other constants' inputs/outputs for TensorRT segments ([54e407e](https://github.com/NVIDIA/TRTorch/commit/54e407e))
* support prim::Param for fallback inputs ([ec2bbf2](https://github.com/NVIDIA/TRTorch/commit/ec2bbf2))
* support prim::Param for input type after refactor ([3cebe97](https://github.com/NVIDIA/TRTorch/commit/3cebe97))
* support Python APIs for Automatic Fallback ([100b090](https://github.com/NVIDIA/TRTorch/commit/100b090))
* support the case when the injected node is not supported in dependency analysis ([c67d8f6](https://github.com/NVIDIA/TRTorch/commit/c67d8f6))
* support truncate long/double to int/float with option ([740eb54](https://github.com/NVIDIA/TRTorch/commit/740eb54))
* Try to submit review before exit ([9a9d7f0](https://github.com/NVIDIA/TRTorch/commit/9a9d7f0))
* update truncate long/double python api ([69e49e8](https://github.com/NVIDIA/TRTorch/commit/69e49e8))
* **//docker:** Adding Docker 21.03 ([9b326e8](https://github.com/NVIDIA/TRTorch/commit/9b326e8))
* update truncate long/double warning message ([60dba12](https://github.com/NVIDIA/TRTorch/commit/60dba12))
* **//docker:** Update CI container ([df63467](https://github.com/NVIDIA/TRTorch/commit/df63467))
* **//py:** Allowing people using the PyTorch backend to use TRTorch/TRT ([6c3e0ad](https://github.com/NVIDIA/TRTorch/commit/6c3e0ad))
* **//py:** Catch when bazel is not in path and error out when running ([1da999d](https://github.com/NVIDIA/TRTorch/commit/1da999d))
* **//py:** Gate partial compilation from to_backend API ([bf1b2d8](https://github.com/NVIDIA/TRTorch/commit/bf1b2d8))
* **//py:** New API to embed engine in new module ([88d07a9](https://github.com/NVIDIA/TRTorch/commit/88d07a9))
* **aten::floor:** Adds floor.int evaluator ([a6a46e5](https://github.com/NVIDIA/TRTorch/commit/a6a46e5))


### BREAKING CHANGES

* PyTorch version has been bumped to 1.8.0
Default CUDA version is CUDA 11.1
TensorRT version is TensorRT 7.2.3.4
cuDNN version is now cuDNN 8.1

Signed-off-by: Naren Dasan <naren@narendasan.com>
Signed-off-by: Naren Dasan <narens@nvidia.com>
* Due to issues with compatability between PyTorch 1.8.0
and 1.8.1 in the Torch Python API, TRTorch 0.3.0 compiled for 1.8.0 does not
work with PyTorch 1.8.1 and will show an error about use_input_stats.
If you see this error make sure the version of libtorch you are
compiling with is PyTorch 1.8.1

TRTorch 0.3.0 will target PyTorch 1.8.1. There is no backwards
compatability with 1.8.0. If you need this specific version compile from
source with the dependencies in WORKSPACE changed

Signed-off-by: Naren Dasan <naren@narendasan.com>
Signed-off-by: Naren Dasan <narens@nvidia.com>

#  0.4.0 (2021-08-24)


* feat(serde)!: Refactor CudaDevice struct, implement ABI versioning, ([9327cce](https://github.com/NVIDIA/TRTorch/commit/9327cce))
* feat(//py)!: Implementing top level python api changes to reflect new ([482265f](https://github.com/NVIDIA/TRTorch/commit/482265f))
* feat(//cpp)!: Changes to TRTorch C++ api reflecting Input and ([08b4942](https://github.com/NVIDIA/TRTorch/commit/08b4942))
* feat!: Pytorch 1.9 version bump ([a12d249](https://github.com/NVIDIA/TRTorch/commit/a12d249))
* feat(//core/runtime)!: Better and more portable names for engines ([6eb3bb2](https://github.com/NVIDIA/TRTorch/commit/6eb3bb2))


### Bug Fixes

* **//core/conversion/conversionctx:** Guard final engine building ([dfa9ae8](https://github.com/NVIDIA/TRTorch/commit/dfa9ae8))
* **//core/lowering:** use lower_info as parameter ([370aeb9](https://github.com/NVIDIA/TRTorch/commit/370aeb9))
* **//cpp/ptq:** fixing bad accuracy in just the example code ([7efa11d](https://github.com/NVIDIA/TRTorch/commit/7efa11d))
* **//py:** Fix python setup.py with new libtrtorch.so location ([68ba63c](https://github.com/NVIDIA/TRTorch/commit/68ba63c))
* **//tests:** fix optional jetson tests ([4c32a83](https://github.com/NVIDIA/TRTorch/commit/4c32a83))
* **//tests:** use right type for masked_fill test ([4a5c28f](https://github.com/NVIDIA/TRTorch/commit/4a5c28f))
* **aten::cat:** support neg dim for cat ([d8ca182](https://github.com/NVIDIA/TRTorch/commit/d8ca182))
* **aten::select and aten::var:** Fix converters to handle negative axes ([3a734a2](https://github.com/NVIDIA/TRTorch/commit/3a734a2))
* **aten::slice:** Allow slicing of pytorch tensors ([50f012e](https://github.com/NVIDIA/TRTorch/commit/50f012e))
* **aten::tensor:** Last dim doesnt always get written right ([b68d4aa](https://github.com/NVIDIA/TRTorch/commit/b68d4aa))
* **aten::tensor:** Last dim doesnt always get written right ([38744bc](https://github.com/NVIDIA/TRTorch/commit/38744bc))
* Address review comments, fix failing tests due to bool mishandling ([13eef91](https://github.com/NVIDIA/TRTorch/commit/13eef91))
* Final working version of QAT in TRTorch ([521a0cb](https://github.com/NVIDIA/TRTorch/commit/521a0cb))
* fix aten::sub.scalar operator ([9a09514](https://github.com/NVIDIA/TRTorch/commit/9a09514))
* Fix linear lowering pass, lift layer_norm scale layer restriction and matmul layer nbdims restriction ([930d582](https://github.com/NVIDIA/TRTorch/commit/930d582))
* Fix testcases using old InputRange API ([ff87956](https://github.com/NVIDIA/TRTorch/commit/ff87956))
* Fix TRT8 engine capability flags ([2b69742](https://github.com/NVIDIA/TRTorch/commit/2b69742))
* Fix warnings thrown by noexcept functions ([c5f7eea](https://github.com/NVIDIA/TRTorch/commit/c5f7eea))
* Fix warnings thrown by noexcept functions ([ddc8950](https://github.com/NVIDIA/TRTorch/commit/ddc8950))
* Minor fixes to qat scripts ([b244423](https://github.com/NVIDIA/TRTorch/commit/b244423))
* Restrict TRTorch to compile only forward methods ([9f006d5](https://github.com/NVIDIA/TRTorch/commit/9f006d5))
* Transfer calibration data to gpu when it is not a batch ([23739cb](https://github.com/NVIDIA/TRTorch/commit/23739cb))
* typo in aten::batch_norm ([d47f48f](https://github.com/NVIDIA/TRTorch/commit/d47f48f))
* **qat:** Rescale input data for C++ application ([9dc6061](https://github.com/NVIDIA/TRTorch/commit/9dc6061))
* Use len() to get size of dataset ([ccc60d5](https://github.com/NVIDIA/TRTorch/commit/ccc60d5))
* **device_conf:** Devices never actually got swithed in multi device ([f1d0a43](https://github.com/NVIDIA/TRTorch/commit/f1d0a43))
* **exception_elimination:** Exception branches are no longer consistent ([d61b667](https://github.com/NVIDIA/TRTorch/commit/d61b667))
* **to_backend:** Clean up to_backend implementation ([4e15605](https://github.com/NVIDIA/TRTorch/commit/4e15605))
* **trtorchc:** Allow for workspaces larger than 2G and better debugging ([e1e7812](https://github.com/NVIDIA/TRTorch/commit/e1e7812))
* Using TensorRT 8 new API calls ([14691e7](https://github.com/NVIDIA/TRTorch/commit/14691e7))
* Using TensorRT 8 new API calls ([fa969a5](https://github.com/NVIDIA/TRTorch/commit/fa969a5))


### Features

* **//core/conversion:** Adding error prefix to python source traceback ([4bf2a41](https://github.com/NVIDIA/TRTorch/commit/4bf2a41))
* **//core/conversion:** Handle adding and wrapping ITensors as ([a22e99b](https://github.com/NVIDIA/TRTorch/commit/a22e99b))
* **//core/ir:** Implementing new internal input spec type ([316df28](https://github.com/NVIDIA/TRTorch/commit/316df28))
* **//core/lowering:** Adding two passes, one to delimit and one to mark ([2e04ce5](https://github.com/NVIDIA/TRTorch/commit/2e04ce5))
* **//core/lowering:** additional logging in module fallback ([ad07645](https://github.com/NVIDIA/TRTorch/commit/ad07645))
* **//core/plugins:** Add adaptive_max_pool2d plugin, enable the plugins to run on GPU ([6f4aa40](https://github.com/NVIDIA/TRTorch/commit/6f4aa40))
* **//cpp/int8/qat:** QAT application release ([d8f5d29](https://github.com/NVIDIA/TRTorch/commit/d8f5d29))
* **//examples/int8:** Implement Makefile based execution for ptq and qat ([b7f6d8a](https://github.com/NVIDIA/TRTorch/commit/b7f6d8a))
* **//examples/int8/qat:** Install pytorch-quantization with ([1ca1484](https://github.com/NVIDIA/TRTorch/commit/1ca1484))
* **//py:** add user level device class in py for embed engine ([d99169f](https://github.com/NVIDIA/TRTorch/commit/d99169f))
* **aten::masked_fill:** In progress implementation of masked_fill ([fa7d6d9](https://github.com/NVIDIA/TRTorch/commit/fa7d6d9))
* **aten::ones:** Adding support for aten::ones ([2b45a3d](https://github.com/NVIDIA/TRTorch/commit/2b45a3d))
* **aten::slice:** Patching slice for new optional params ([a11287f](https://github.com/NVIDIA/TRTorch/commit/a11287f))
* **aten::sqrt:** Adding support for sqrt evaluators ([6aaba3b](https://github.com/NVIDIA/TRTorch/commit/6aaba3b))
* **aten::std|aten::masked_fill:** Implement masked_fill, aten::std ([a086a5b](https://github.com/NVIDIA/TRTorch/commit/a086a5b))
* **aten::std|aten::masked_fill:** Implement masked_fill, aten::std ([2866627](https://github.com/NVIDIA/TRTorch/commit/2866627))
* **jetson:** Support for Jetpack 4.6 ([9760fe3](https://github.com/NVIDIA/TRTorch/commit/9760fe3))
* **to_backend:** Updating backend integration preproc function ([080b594](https://github.com/NVIDIA/TRTorch/commit/080b594))
* Enable sparsity support in TRTorch ([f9e1f2b](https://github.com/NVIDIA/TRTorch/commit/f9e1f2b))
* **trtorchc:** Adding flag for sparse weights ([bfdc6f5](https://github.com/NVIDIA/TRTorch/commit/bfdc6f5))
* Add aten::full converter, quantization ops testcases ([9f2ffd0](https://github.com/NVIDIA/TRTorch/commit/9f2ffd0))
* Add aten::type_as lowering pass ([b57a6dd](https://github.com/NVIDIA/TRTorch/commit/b57a6dd))
* Add functionality for QAT workflow ([fc8eafb](https://github.com/NVIDIA/TRTorch/commit/fc8eafb))
* Add functionality for QAT workflow ([f776e76](https://github.com/NVIDIA/TRTorch/commit/f776e76))
* Add support for providing input datatypes in TRTorch ([a3f4a3c](https://github.com/NVIDIA/TRTorch/commit/a3f4a3c))
* Adding automatic casting to compare layers ([90af26e](https://github.com/NVIDIA/TRTorch/commit/90af26e))
* Enable sparsity support in TRTorch ([decd0ed](https://github.com/NVIDIA/TRTorch/commit/decd0ed))
* Enable TRT 8.0 QAT functionality in TRTorch ([c76a28a](https://github.com/NVIDIA/TRTorch/commit/c76a28a))
* Makefile for trtorchrt.so example ([c60c521](https://github.com/NVIDIA/TRTorch/commit/c60c521))
* show pytorch code of unsupported operators ([2ee2a84](https://github.com/NVIDIA/TRTorch/commit/2ee2a84))
* support aten::Int ([5bc977d](https://github.com/NVIDIA/TRTorch/commit/5bc977d))
* **trtorchc:** Adding more dtype aliases ([652fb13](https://github.com/NVIDIA/TRTorch/commit/652fb13))
* **trtorchc:** Adding new support for dtypes and formats in ([c39bf81](https://github.com/NVIDIA/TRTorch/commit/c39bf81))
* Support fallback options in trtorchc ([ad966b7](https://github.com/NVIDIA/TRTorch/commit/ad966b7))
* Using shared_ptrs to manage TRT resources in runtime ([e336630](https://github.com/NVIDIA/TRTorch/commit/e336630))
* **trtorchc:** Embedding engines in modules from the CLI ([2b4b9e3](https://github.com/NVIDIA/TRTorch/commit/2b4b9e3))


### BREAKING CHANGES

* This commit cleans up the WIP CudaDevice class,
simplifying implementation and formalizing the seralized format for CUDA
devices.

It also implements ABI Versioning. The first entry in the serialized
format of a TRTEngine now records the ABI that the engine was compiled
with, defining expected compatibility with the TRTorch runtime. If the
ABI version does not match, the runtime will error out asking to
recompile the program.

ABI version is a monotonically increasing integer and should be
incremented everytime the serialization format changes in some way.

This commit cleans up the CudaDevice class, implementing a number of
constructors to replace the various utility functions that populate the
struct. Descriptive utility functions remain but solely call the
relevant constructor.

Signed-off-by: Naren Dasan <naren@narendasan.com>
Signed-off-by: Naren Dasan <narens@nvidia.com>
* This commit introduces the next iteration of the Python
TRTorch API. Starting in TRTorch v0.5.0 support for the "input_shapes"
and "op_precision" compile spec keys will be removed. Users should port
forward to using the "inputs" key which expects a list of trtorch.Input
objects and the "enabled_precisions" key which expects a set of data
type specifying enums.

Signed-off-by: Naren Dasan <naren@narendasan.com>
Signed-off-by: Naren Dasan <narens@nvidia.com>
* This change deprecates InputRange, and the CompileSpec
fields "input_shapes", "op_precision" and associated contructors and
functions. These are replaced wtih Input, "inputs" and
"enabled_precisions" respectively. Deprecated components will be removed
in TRTorch v0.5.0

Signed-off-by: Naren Dasan <naren@narendasan.com>
Signed-off-by: Naren Dasan <narens@nvidia.com>
* Updating PyTorch version to 1.9.0 which includes
breaking changes to the to_backend api

Signed-off-by: Naren Dasan <naren@narendasan.com>
Signed-off-by: Naren Dasan <narens@nvidia.com>
* This bumps the TRTorch ABI version to 3 due to
a new field for engine name included in the serialized form of
TRTEngine. This lets deserialized engines have the same name they
serialized with

Signed-off-by: Naren Dasan <naren@narendasan.com>
Signed-off-by: Naren Dasan <narens@nvidia.com>


# 0.4.1 (2021-10-06)

### Bug Fixes

* **//core/lowering:** Fixes module level fallback recursion ([2fc612d](https://github.com/NVIDIA/TRTorch/commit/2fc612d))
* Move some lowering passes to graph level logging ([0266f41](https://github.com/NVIDIA/TRTorch/commit/0266f41))
* **//py:** Fix trtorch.Device alternate contructor options ([ac26841](https://github.com/NVIDIA/TRTorch/commit/ac26841))

# 1.0.0 (2021-11-09)


### Bug Fixes

* aten::gelu call was wrong in test ([40bc4e3](https://github.com/NVIDIA/TRTorch/commit/40bc4e3))
* Fix a core partitioning algo bug where non-tensor input segments are not updated correctly ([cc10876](https://github.com/NVIDIA/TRTorch/commit/cc10876))
* Fix modules_as_engines test case to use trt_mod instead of pyt_mod ([282e98a](https://github.com/NVIDIA/TRTorch/commit/282e98a))
* Fix plugin registration macro ([8afab22](https://github.com/NVIDIA/TRTorch/commit/8afab22))
* Fix python API tests for mobilenet v2 ([e5a38ff](https://github.com/NVIDIA/TRTorch/commit/e5a38ff))
* Partial compilation translation to internal settings was incorrect ([648bad3](https://github.com/NVIDIA/TRTorch/commit/648bad3))
* **//py:** Don't crash harshly on import when CUDA is not available ([07e16fd](https://github.com/NVIDIA/TRTorch/commit/07e16fd))
* Renable backtrace and make it less repetitive ([1435845](https://github.com/NVIDIA/TRTorch/commit/1435845))
* **//core/lowering:** Fixes module level fallback recursion ([f94ae8f](https://github.com/NVIDIA/TRTorch/commit/f94ae8f))
* **//core/partitioing:** Fixing support for paritally compiling ([748ecf3](https://github.com/NVIDIA/TRTorch/commit/748ecf3))
* **//docker:** Update docker container build script to use release path ([9982855](https://github.com/NVIDIA/TRTorch/commit/9982855))
* **//py:** Add new dirs to remove during clean ([d2cc1e9](https://github.com/NVIDIA/TRTorch/commit/d2cc1e9))
* **//py:** Fix some api import issues ([840ca89](https://github.com/NVIDIA/TRTorch/commit/840ca89))
* **//py:** Fix trtorch.Device alternate contructor options ([fa08311](https://github.com/NVIDIA/TRTorch/commit/fa08311))
* **//py:** Fix trtorch.Device alternate contructor options ([ac26841](https://github.com/NVIDIA/TRTorch/commit/ac26841))
* Update notebooks with new library name Torch-TensorRT ([8274fd9](https://github.com/NVIDIA/TRTorch/commit/8274fd9))
* **aten::conv1d:** Update namespace, fix typo in dest IR for conv1d ([d53f136](https://github.com/NVIDIA/TRTorch/commit/d53f136))
* **eval:** Rollback 1.11a0 change + namespace issues ([ba743f5](https://github.com/NVIDIA/TRTorch/commit/ba743f5))
* Use scripting instead of tracing for module fallback tests ([32e8b53](https://github.com/NVIDIA/TRTorch/commit/32e8b53))
* Workspace defaults for other apis and centralize cuda api use ([930321e](https://github.com/NVIDIA/TRTorch/commit/930321e))


### Features

* Add functionality for tests to use precompiled libraries ([b5c324a](https://github.com/NVIDIA/TRTorch/commit/b5c324a))
* Add QAT patch which modifies scale factor dtype to INT32 ([4a10673](https://github.com/NVIDIA/TRTorch/commit/4a10673))
* Add TF32 override flag in bazelrc for CI-Testing ([7a0c9a5](https://github.com/NVIDIA/TRTorch/commit/7a0c9a5))
* Add VGG QAT sample notebook which demonstrates end-end workflow for QAT models ([8bf6dd6](https://github.com/NVIDIA/TRTorch/commit/8bf6dd6))
* Augment python package to include bin, lib, include directories ([ddc0685](https://github.com/NVIDIA/TRTorch/commit/ddc0685))
* handle scalar type of size [] in shape_analysis ([fca53ce](https://github.com/NVIDIA/TRTorch/commit/fca53ce))
* support aten::__and__.bool evaluator ([6d73e43](https://github.com/NVIDIA/TRTorch/commit/6d73e43))
* support aten::conv1d and aten::conv_transpose1d ([c8dc6e9](https://github.com/NVIDIA/TRTorch/commit/c8dc6e9))
* support aten::eq.str evaluator ([5643972](https://github.com/NVIDIA/TRTorch/commit/5643972))
* support setting input types of subgraph in fallback, handle Tensor type in evaluated_value_map branch in MarkOutputs ([4778b2b](https://github.com/NVIDIA/TRTorch/commit/4778b2b))
* support truncate_long_and_double in fallback subgraph input type ([0bc3c05](https://github.com/NVIDIA/TRTorch/commit/0bc3c05))
* Update documentation with new library name Torch-TensorRT ([e5f96d9](https://github.com/NVIDIA/TRTorch/commit/e5f96d9))
* Updating the pre_built to prebuilt ([51412c7](https://github.com/NVIDIA/TRTorch/commit/51412c7))
* **//:libtrtorch:** Ship a WORKSPACE file and BUILD file with the ([7ac6f1c](https://github.com/NVIDIA/TRTorch/commit/7ac6f1c))
* **//core/partitioning:** Improved logging and code org for the ([8927e77](https://github.com/NVIDIA/TRTorch/commit/8927e77))
* **//cpp:** Adding example tensors as a way to set input spec ([70a7bb3](https://github.com/NVIDIA/TRTorch/commit/70a7bb3))
* **//py:** Add the git revision to non release builds ([4a0a918](https://github.com/NVIDIA/TRTorch/commit/4a0a918))
* **//py:** Allow example tensors from torch to set shape ([01d525d](https://github.com/NVIDIA/TRTorch/commit/01d525d))


* feat!: Changing the default behavior for selecting the input type ([a234335](https://github.com/NVIDIA/TRTorch/commit/a234335))
* refactor!: Removing deprecated InputRange, op_precision and input_shapes ([621bc67](https://github.com/NVIDIA/TRTorch/commit/621bc67))
* feat(//py)!: Porting forward the API to use kwargs ([17e0e8a](https://github.com/NVIDIA/TRTorch/commit/17e0e8a))
* refactor(//py)!: Kwargs updates and support for shifting internal apis ([2a0d1c8](https://github.com/NVIDIA/TRTorch/commit/2a0d1c8))
* refactor!(//cpp): Inlining partial compilation settings since the ([19ecc64](https://github.com/NVIDIA/TRTorch/commit/19ecc64))
* refactor! : Update default workspace size based on platforms. ([391a4c0](https://github.com/NVIDIA/TRTorch/commit/391a4c0))
* feat!: Turning on partial compilation by default ([52e2f05](https://github.com/NVIDIA/TRTorch/commit/52e2f05))
* refactor!: API level rename ([483ef59](https://github.com/NVIDIA/TRTorch/commit/483ef59))
* refactor!: Changing the C++ api to be snake case ([f34e230](https://github.com/NVIDIA/TRTorch/commit/f34e230))
* refactor! : Update Pytorch version to 1.10 ([cc7d0b7](https://github.com/NVIDIA/TRTorch/commit/cc7d0b7))
* refactor!: Updating bazel version for py build container ([06533fe](https://github.com/NVIDIA/TRTorch/commit/06533fe))


### BREAKING CHANGES

* This removes the InputRange Class and op_precision and
input shape fields which were deprecated in TRTorch v0.4.0

Signed-off-by: Naren Dasan <naren@narendasan.com>
Signed-off-by: Naren Dasan <narens@nvidia.com>
* This change updates the bazel version
to build Torch-TensorRT to 4.2.1.

This was done since the only version of bazel available
in our build container for python apis is 4.2.1

Signed-off-by: Naren Dasan <naren@narendasan.com>
Signed-off-by: Naren Dasan <narens@nvidia.com>
* This changes the API for compile settings
from a dictionary of settings to a set of kwargs for the various
compilation functions. This will break existing code. However
there is simple guidance to port forward your code:

Given a dict of valid TRTorch CompileSpec settings

```py
spec = {
	"inputs": ...
	...
}
```

You can use this same dict with the new APIs by changing your code from:

```py
trtorch.compile(mod, spec)
```

to:

```py
trtorch.compile(mod, **spec)
```
which will unpack the dictionary as arguments to the function

Signed-off-by: Naren Dasan <naren@narendasan.com>
Signed-off-by: Naren Dasan <narens@nvidia.com>
* This commit changes the APIs from a dictionary of
arguements to a set of kwargs. You can port forward using

```py
trtorch.compile(mod, **spec)
```

Also in preparation for partial compilation to be enabled by default
settings related to torch fallback have been moved to the top level

instead of

```py
"torch_fallback": {
  "enabled": True,
  "min_block_size" " 3,
  "forced_fallback_ops" : ["aten::add"],
  "forced_fallback_mods" : ["MySubModule"]
}
```

now there are new settings

```py
require_full_compilation=False,
min_block_size=3,
torch_executed_ops=["aten::add"],
torch_executed_modules=["MySubModule"]
```

Signed-off-by: Naren Dasan <naren@narendasan.com>
Signed-off-by: Naren Dasan <narens@nvidia.com>
* This commit changes the API for automatic fallback
to inline settings regarding partial compilation in preparation
for it to be turned on by default

Now in the compile spec instead of a `torch_fallback` field with its
associated struct, there are four new fields in the compile spec

```c++
bool require_full_compilation = true;
uint64_t min_block_size = 3;
std::vector<std::string> torch_executed_ops = {};
std::vector<std::string> torch_executed_modules = {};
```

Signed-off-by: Naren Dasan <naren@narendasan.com>
Signed-off-by: Naren Dasan <narens@nvidia.com>
* This commit sets the default workspace size to 1GB for GPU platforms and 256MB for Jetson Nano/TX1 platforms whose compute capability is < 6.

Signed-off-by: Dheeraj Peri <peri.dheeraj@gmail.com>

Signed-off-by: Dheeraj Peri <peri.dheeraj@gmail.com>

Signed-off-by: Dheeraj Peri <peri.dheeraj@gmail.com>

Signed-off-by: Dheeraj Peri <peri.dheeraj@gmail.com>

Signed-off-by: Dheeraj Peri <peri.dheeraj@gmail.com>
* This commit turns on partial compilation
by default. Unsupported modules will attempt to be
run partially in PyTorch and partially in TensorRT

Signed-off-by: Naren Dasan <naren@narendasan.com>
Signed-off-by: Naren Dasan <narens@nvidia.com>
* This commit renames the namespaces of all
TRTorch/Torch-TensorRT APIs. Now torchscript specific functions
are segregated into their own torch_tensorrt::torchscript /
torch_tensorrt.ts namespaces. Generic utils will remain in the
torch_tensorrt namespace. Guidance on how to port forward will follow in
the next commits
* This changes the C++ API ::ts
APIs to be snake case and for CompileModules to
become just compile

Signed-off-by: Naren Dasan <narens@nvidia.com>
Signed-off-by: Naren Dasan <naren@narendasan.com>
* This commit updates the pytorch version to 1.10. To use python API of torch_tensorrt, please upgrade your local pytorch to 1.10 to avoid ABI incompatibility errors. WORKSPACE and requirements files are updated accordingly

Signed-off-by: Dheeraj Peri <peri.dheeraj@gmail.com>

Signed-off-by: Dheeraj Peri <peri.dheeraj@gmail.com>
* This commit changes the default behavior of
the compiler where if the user does not specify an input data
type explicity instead of using the enabled precision, now
the compiler will inspect the model provided to infer the
data type for the input that will not cause an error if
the model was run in torch. In practice this means

- If the weights are in FP32 for the first tensor calculation
  then default input type is FP32
- If the weights are in FP16 for the first tensor calculation
  then default input type is FP16
- etc.

If the data type cannot be determined the compiler will
default to FP32.

This calculation is done per input tensor so if one input
is inferred to use FP32 and another INT32 then the expected
types will be the same (FP32, INT32)

As was the same before if the user defines the data type
explicitly or provides an example tensor the data type
specified there will be respected

Signed-off-by: Naren Dasan <naren@narendasan.com>
Signed-off-by: Naren Dasan <narens@nvidia.com>



# 1.1.0 (2022-05-10)

### Bug Fixes

* add at::adaptive_avg_pool1d in interpolate plugin and fix [#791](https://github.com/NVIDIA/TRTorch/issues/791) ([deb9f74](https://github.com/NVIDIA/TRTorch/commit/deb9f74))
* Added ipywidget dependency to notebook ([0b2040a](https://github.com/NVIDIA/TRTorch/commit/0b2040a))
* Added test case names ([296e98a](https://github.com/NVIDIA/TRTorch/commit/296e98a))
* Added truncate_long_and_double ([417c096](https://github.com/NVIDIA/TRTorch/commit/417c096))
* Adding truncate_long_and_double to ptq tests ([3a0640a](https://github.com/NVIDIA/TRTorch/commit/3a0640a))
* Avoid resolving non-tensor inputs to torch segment_blocks unneccessarily ([3e090ee](https://github.com/NVIDIA/TRTorch/commit/3e090ee))
* Considering rtol and atol in threshold comparison for floating point numbers ([0b0ba8d](https://github.com/NVIDIA/TRTorch/commit/0b0ba8d))
* Disabled mobilenet_v2 test for DLFW CI ([40c611f](https://github.com/NVIDIA/TRTorch/commit/40c611f))
* fix bug that python api doesn't pass truncate_long_and_double value to internal.partition_info ([828336d](https://github.com/NVIDIA/TRTorch/commit/828336d))
* fix bugs in aten::to ([2ecd187](https://github.com/NVIDIA/TRTorch/commit/2ecd187))
* Fix BUILD file for tests/accuracy ([8b0170e](https://github.com/NVIDIA/TRTorch/commit/8b0170e))
* Fix existing uninstallation of Torch-TRT ([9ddd7a8](https://github.com/NVIDIA/TRTorch/commit/9ddd7a8))
* Fix for torch scripted module faiure with DLFW ([88c02d9](https://github.com/NVIDIA/TRTorch/commit/88c02d9))
* Fix fuse addmm pass ([58e9ea0](https://github.com/NVIDIA/TRTorch/commit/58e9ea0))
* Fix pre_built name change in bazelrc ([3ecee21](https://github.com/NVIDIA/TRTorch/commit/3ecee21))
* fix the bug that introduces kLong Tensor in prim::NumToTensor ([2c3e1d9](https://github.com/NVIDIA/TRTorch/commit/2c3e1d9))
* Fix when TRT prunes away an output ([9465e1d](https://github.com/NVIDIA/TRTorch/commit/9465e1d))
* Fixed bugs and addressed review comments ([588e1d1](https://github.com/NVIDIA/TRTorch/commit/588e1d1))
* Fixed failures for host deps sessions ([ec2232f](https://github.com/NVIDIA/TRTorch/commit/ec2232f))
* Fixed typo in the path ([43fab56](https://github.com/NVIDIA/TRTorch/commit/43fab56))
* Getting unsupported ops will now bypass non-schema ops avoiding redundant failures ([d7d1511](https://github.com/NVIDIA/TRTorch/commit/d7d1511))
* Guard test activation for CI testing ([6d1a1fd](https://github.com/NVIDIA/TRTorch/commit/6d1a1fd))
* Implement a patch for gelu schema change in older NGC containers ([9ee3a04](https://github.com/NVIDIA/TRTorch/commit/9ee3a04))
* Missing log severity ([6a4daef](https://github.com/NVIDIA/TRTorch/commit/6a4daef))
* Preempt torch package override via timm in nox session ([8964d1b](https://github.com/NVIDIA/TRTorch/commit/8964d1b))
* refactor the resegmentation for TensorRT segments in ResolveNonTensorInput ([3cc2dfb](https://github.com/NVIDIA/TRTorch/commit/3cc2dfb))
* remove outdated member variables ([0268da2](https://github.com/NVIDIA/TRTorch/commit/0268da2))
* Removed models directory dependencies ([c4413e1](https://github.com/NVIDIA/TRTorch/commit/c4413e1))
* Resolve issues in exception elmination pass ([99cea1b](https://github.com/NVIDIA/TRTorch/commit/99cea1b))
* Review comments incorporated ([962660d](https://github.com/NVIDIA/TRTorch/commit/962660d))
* Review comments incorporated ([e9865c2](https://github.com/NVIDIA/TRTorch/commit/e9865c2))
* support dict type for input in shape analysis ([630f9c4](https://github.com/NVIDIA/TRTorch/commit/630f9c4))
* truncate_long_and_double incur torchscript inference issues ([c83aa15](https://github.com/NVIDIA/TRTorch/commit/c83aa15))
* Typo fix for test case name ([2a516b2](https://github.com/NVIDIA/TRTorch/commit/2a516b2))
* Update "reduceAxes" variable in GlobalPoolingConverter function and add corresponding uTests ([f6f5e3e](https://github.com/NVIDIA/TRTorch/commit/f6f5e3e))
* **//core/conversion/evaluators:** Change how schemas are handled ([20e5d41](https://github.com/NVIDIA/TRTorch/commit/20e5d41))
* Update base container for dockerfile ([1b3245a](https://github.com/NVIDIA/TRTorch/commit/1b3245a))
* **//core:** Take user setting in the case we can't determine the ([01c89d1](https://github.com/NVIDIA/TRTorch/commit/01c89d1)), closes [#814](https://github.com/NVIDIA/TRTorch/issues/814)
* Update test for new Exception syntax ([2357099](https://github.com/NVIDIA/TRTorch/commit/2357099))
* **//core/conversion:** Add special case for If and Loop ([eacde8d](https://github.com/NVIDIA/TRTorch/commit/eacde8d))
* **//core/runtime:** Support more delimiter variants ([819c911](https://github.com/NVIDIA/TRTorch/commit/819c911))
* **//cpp/bin/torchtrtc:** Fix mbs ([aca175f](https://github.com/NVIDIA/TRTorch/commit/aca175f))
* **//docsrc:** Fix dependencies for docgen ([806e663](https://github.com/NVIDIA/TRTorch/commit/806e663))
* **//notebooks:** Render citrinet ([12dbda1](https://github.com/NVIDIA/TRTorch/commit/12dbda1))
* **//py:** Constrain the CUDA version in container builds ([a21a045](https://github.com/NVIDIA/TRTorch/commit/a21a045))
* Use user provided dtype when we can't infer it from the graph ([14650d1](https://github.com/NVIDIA/TRTorch/commit/14650d1))


### Code Refactoring

* removing the strict_types and max_batch_size apis ([b30cbd9](https://github.com/NVIDIA/TRTorch/commit/b30cbd9))
* Rename enabled precisions arugment to ([10957eb](https://github.com/NVIDIA/TRTorch/commit/10957eb))
* Removing the max-batch-size argument ([03bafc5](https://github.com/NVIDIA/TRTorch/commit/03bafc5))

### Features

* **//core/conversion:** Better tooling for debugging ([c5c5c47](https://github.com/NVIDIA/TRTorch/commit/c5c5c47))
* **//core/conversion/evaluators:** aten::pow support ([c4fdfcb](https://github.com/NVIDIA/TRTorch/commit/c4fdfcb))
* **//docker:** New base container to let master build in container ([446bf18](https://github.com/NVIDIA/TRTorch/commit/446bf18))
* **//py:** Context managers to quickly switch logging level ([12e470f](https://github.com/NVIDIA/TRTorch/commit/12e470f))
* Add converter files for reflection pad 1d and 2d ([406d860](https://github.com/NVIDIA/TRTorch/commit/406d860))
* Add converter files for torch::max ([f628aca](https://github.com/NVIDIA/TRTorch/commit/f628aca))
* Add converter files for torch::max ([569bcde](https://github.com/NVIDIA/TRTorch/commit/569bcde))
* Add converter files for torch::max ([dd7a44e](https://github.com/NVIDIA/TRTorch/commit/dd7a44e))
* Add converter for reflection pad 1d and 2d operation ([2484a43](https://github.com/NVIDIA/TRTorch/commit/2484a43))
* Added comprehensive perf benchmark script ([a8016ff](https://github.com/NVIDIA/TRTorch/commit/a8016ff))
* Added compute capability for Orin ([af3d0ff](https://github.com/NVIDIA/TRTorch/commit/af3d0ff))
* Added env var for TOP_DIR ([c26180e](https://github.com/NVIDIA/TRTorch/commit/c26180e))
* Added Python accuracy tests using Nox ([6ae8652](https://github.com/NVIDIA/TRTorch/commit/6ae8652))
* Enable prim::DictConstruct to fallback without conversion check error ([01d98c7](https://github.com/NVIDIA/TRTorch/commit/01d98c7))
* Handle empty schemas for unsupported ops ([bf6c929](https://github.com/NVIDIA/TRTorch/commit/bf6c929))
* Implement fast approximation of Gelu as lowering pass to improve performance ([8024ea2](https://github.com/NVIDIA/TRTorch/commit/8024ea2))
* Implement lowering for  aten::to.dtype schema ([4b3ae3a](https://github.com/NVIDIA/TRTorch/commit/4b3ae3a))
* Implement test case for aten::to.dtype lowering ([bde8ee0](https://github.com/NVIDIA/TRTorch/commit/bde8ee0))
* Perf benchmark initial draft ([f2d1655](https://github.com/NVIDIA/TRTorch/commit/f2d1655))
* replace view with reshape during lowering ([d39b918](https://github.com/NVIDIA/TRTorch/commit/d39b918))
* Review comment incorporated ([161ef3d](https://github.com/NVIDIA/TRTorch/commit/161ef3d))
* support aten::adaptive_max_pool1d, aten::adaptive_avg_pool3d and aten::adaptive_max_pool3d operators ([e554dbd](https://github.com/NVIDIA/TRTorch/commit/e554dbd))
* support aten::div.Tensor_mode ([bb3046a](https://github.com/NVIDIA/TRTorch/commit/bb3046a))
* support aten::extend evaluator ([33c523d](https://github.com/NVIDIA/TRTorch/commit/33c523d))
* support aten::format evaluator ([3a33d33](https://github.com/NVIDIA/TRTorch/commit/3a33d33))
* Update Pytorch version to 1.11 ([c009a1f](https://github.com/NVIDIA/TRTorch/commit/c009a1f))
* Upgrade TensorRT to 8.2.4.2 ([f1f151b](https://github.com/NVIDIA/TRTorch/commit/f1f151b))
* **//tests:** Adding BERT to the test suite ([7996a10](https://github.com/NVIDIA/TRTorch/commit/7996a10))
* **aten::__range_length:** Adding range length evaluator ([11c4608](https://github.com/NVIDIA/TRTorch/commit/11c4608))
* **aten::add:** adding string concat evaluator ([65dbf90](https://github.com/NVIDIA/TRTorch/commit/65dbf90))
* **aten::Int:** Adding a new pass to remove single use ([46ac757](https://github.com/NVIDIA/TRTorch/commit/46ac757))
* **aten::Int:** Lowers out aten::Int ([908340f](https://github.com/NVIDIA/TRTorch/commit/908340f))
* **core//conversion:** Implement converter for torch unbind ([268a49b](https://github.com/NVIDIA/TRTorch/commit/268a49b))


### BREAKING CHANGES

* This commit removes the strict types and max_batch_size apis. We are doing this because the functionality of these APIs in TRT is convoluted and likely to be ignored during building. A replacement for strict types with actual guarantees will be added at a later date.

Signed-off-by: Dheeraj Peri <peri.dheeraj@gmail.com>
* This is a minor change but may cause scripts
using torchtrtc to fail. We are renaming enabled-precisions to
enable-precision since it makes more sense as the argument can
be repeated

Signed-off-by: Naren Dasan <naren@narendasan.com>
Signed-off-by: Naren Dasan <narens@nvidia.com>
* This PR removes `--max-batch-size` from the CLI
as it has no real functional effect

Signed-off-by: Naren Dasan <naren@narendasan.com>
Signed-off-by: Naren Dasan <narens@nvidia.com>
