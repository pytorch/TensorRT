#include "checkshape_plugin.h"
#include "core/util/macros.h"

#define CUDA_MEM_ALIGN 256
#define NDEBUG
using namespace nvinfer1;

namespace trtorch {
namespace core {
namespace conversion {
namespace converters {
namespace impl {
namespace plugins {

namespace
{
    const char* CHECKSHAPE_PLUGIN_VERSION{"0"};
    const char* CHECKSHAPE_PLUGIN_NAME{"CheckShapePlugin"};
    const int CHECKSHAPE_PLUGIN_NUM_INPUT = 2;
    const int CHECKSHAPE_PLUGIN_NUM_OUTPUT = 1;
} // namespace

// Write values into buffer
template <typename T>
void writeToBuffer(char* &buffer, const T& val) {
    *reinterpret_cast<T*>(buffer) = val;
    buffer += sizeof(T);
}

// Read values from buffer
template <typename T>
T readFromBuffer(const char* &buffer) {
    T val = *reinterpret_cast<const T*>(buffer);
    buffer += sizeof(T);
    return val;
}

// Calc aligned workspace size
inline size_t calcAlignedWsS(size_t wss) {
    size_t res = wss;
    if(wss % CUDA_MEM_ALIGN) {
        res += CUDA_MEM_ALIGN - (wss % CUDA_MEM_ALIGN);
    }
    return res;
}

// CALCULATE TOTAL WORKSPACE SIZE
size_t calcWorkspaceSize(size_t *workspaces, int count) {
    size_t total = 0;
    for (int i = 0; i < count; ++i) {
        total += calcAlignedWsS(workspaces[i]);
    }
    return total;
}

// ALIGNPTR 
int8_t* alignPtr(int8_t* ptr, uintptr_t to) {
    uintptr_t addr = (uintptr_t) ptr;
    if (addr % to) {
        addr += to - addr % to;
    }
    return (int8_t*) addr;
}

// NEXTWORKSPACEPTR
int8_t* nextWorkspacePtr(int8_t* ptr, uintptr_t previousWorkspaceSize) {
    uintptr_t addr = (uintptr_t)ptr;
    addr += previousWorkspaceSize;
    return alignPtr((int8_t*)addr, CUDA_MEM_ALIGN);
}


// CheckShapePlugin
CheckShapePlugin::CheckShapePlugin(
    const std::string name, int in_rank, int expand_rank
    )
    : inputRank(in_rank),
    expandRank(expand_rank),
    mLayerName(name)
{
#ifndef NDEBUG
    std::cout << "CheckShapePlugin::CheckShapePlugin(): this=" << this << std::endl;
#endif
}


// used for deserialize from CheckShapePluginCreator::deserializePlugin
CheckShapePlugin::CheckShapePlugin(
    const std::string name,
    const void * data,
    size_t length
    )
    : mLayerName(name) {
#ifndef NDEBUG
    std::cout << "CheckShapePlugin::CheckShapePlugin() - for deserialize: this=" << this << std::endl;
#endif
    const char * d = static_cast<const char *>(data);
    const char * a = d;

    inputRank = readFromBuffer<int>(d);
    expandRank = readFromBuffer<int>(d);

    assert(d == (a + length));
}

CheckShapePlugin::CheckShapePlugin(const CheckShapePlugin &obj) {
#ifndef NDEBUG
    std::cout << "CheckShapePlugin::CheckShapePlugin(CheckShapePlugin &obj): this=" << this << std::endl;
#endif
    inputRank = obj.inputRank;
    expandRank = obj.expandRank;
    mLayerName = obj.mLayerName;
    mPluginNamespace = obj.mPluginNamespace;
}

CheckShapePlugin::~CheckShapePlugin() {
#ifndef NDEBUG
    std::cout << "CheckShapePlugin::~CheckShapePlugin(): this=" << this << std::endl;
#endif
}

// inherited from IPluginV2
const char * CheckShapePlugin::getPluginType() const {
#ifndef NDEBUG
    std::cout << "CheckShapePlugin::getPluginType(): this=" << this << std::endl;
#endif
    return CHECKSHAPE_PLUGIN_NAME;
}

const char * CheckShapePlugin::getPluginVersion() const {
#ifndef NDEBUG
    std::cout << "CheckShapePlugin::getPluginVersion(): this=" << this << std::endl;
#endif
    return CHECKSHAPE_PLUGIN_VERSION;
}

inline int CheckShapePlugin::getNbOutputs() const {
#ifndef NDEBUG
    std::cout << "CheckShapePlugin::getNbOutputs(): this=" << this << std::endl;
#endif
    return CHECKSHAPE_PLUGIN_NUM_OUTPUT;
}

inline int CheckShapePlugin::initialize() {
#ifndef NDEBUG
    std::cout << "CheckShapePlugin::initialize(): this=" << this << std::endl;
#endif 
    return 0;
}

inline void CheckShapePlugin::terminate() {
#ifndef NDEBUG
    std::cout << "CheckShapePlugin::terminate(): this=" << this << std::endl;
#endif
}

inline size_t CheckShapePlugin::getSerializationSize() const
{
#ifndef NDEBUG
    std::cout << "CheckShapePlugin::getSerializationSize(): this=" << this << std::endl;
#endif
    size_t total_size = 0;
    total_size += sizeof(int); // inputRank
    total_size += sizeof(int); // expandRank

#ifndef NDEBUG
    std::cout << "  total size = " << total_size << " Byte" << std::endl;
#endif 

    return total_size;
}

inline void CheckShapePlugin::serialize(void * buffer) const {
#ifndef NDEBUG
    std::cout << "CheckShapePlugin::serialize(): this=" << this << std::endl;
#endif
    char * d = static_cast<char *>(buffer);
    const char * a = d;
    writeToBuffer<int>(d, inputRank);
    writeToBuffer<int>(d, expandRank);

    assert(d == (a + getSerializationSize()));
}

inline void CheckShapePlugin::destroy() {
#ifndef NDEBUG
    std::cout << "CheckShapePlugin::destroy(): this=" << this << std::endl;
#endif
    delete this;
}

inline void CheckShapePlugin::setPluginNamespace(const char * pluginNamespace) {
#ifndef NDEBUG
    std::cout << "CheckShapePlugin::setPluginNamespace(): this=" << this << std::endl;
#endif
    mPluginNamespace = pluginNamespace;
}

inline const char * CheckShapePlugin::getPluginNamespace() const {
#ifndef NDEBUG
    std::cout << "CheckShapePlugin::getPluginNamespace(): this=" << this << std::endl;
#endif
    return mPluginNamespace.c_str();
}

// inherited from IPluginV2Ext
inline DataType CheckShapePlugin::getOutputDataType(int index, const DataType *inputTypes, int nbInputs) const {
#ifndef NDEBUG
    std::cout << "CheckShapePlugin::getOutputDataType(): this=" << this << std::endl;
    std::cout << "  index=" << index << ", inputTypes[0]: " << (int)inputTypes[0] << std::endl;
#endif
    assert(nbInputs == CHECKSHAPE_PLUGIN_NUM_INPUT);
    assert(index < CHECKSHAPE_PLUGIN_NUM_OUTPUT);
    assert(inputTypes[0] == DataType::kINT32);
    assert(inputTypes[1] == DataType::kINT32);

    return inputTypes[0];
}

inline void CheckShapePlugin::attachToContext(cudnnContext  *, cublasContext *, IGpuAllocator *) {
#ifndef NDEBUG
    std::cout << "CheckShapePlugin::attachToContext(): this=" << this << std::endl;
#endif
}

inline void CheckShapePlugin::detachFromContext() {
#ifndef NDEBUG
    std::cout << "CheckShapePlugin::detachFromContext(): this=" << this << std::endl;
#endif
}

// others
inline IPluginV2DynamicExt* CheckShapePlugin::clone() const {
#ifndef NDEBUG
    std::cout << "CheckShapePlugin::clone(): this=" << this << std::endl;
#endif
    return new CheckShapePlugin(*this);
}

inline DimsExprs CheckShapePlugin::getOutputDimensions(int32_t outputIndex, const DimsExprs *inputs, 
    int32_t nbInputs, IExprBuilder &exprBuilder) {
#ifndef NDEBUG
    std::cout << "CheckShapePlugin::getOutputDimensions(): this=" << this << std::endl;
#endif

    assert(outputIndex < CHECKSHAPE_PLUGIN_NUM_OUTPUT);
    assert(nbInputs == CHECKSHAPE_PLUGIN_NUM_INPUT); 
    assert(inputs[0].nbDims == 1);

    DimsExprs outputDims;
    outputDims.nbDims = 1;
    outputDims.d[0] = exprBuilder.constant(1);// shape of shape

#ifndef NDEBUG
    std::cout << "  outputIndex is " << outputIndex << ", nbDims=" << outputDims.nbDims << std::endl;
#endif

    return outputDims;
}

inline bool CheckShapePlugin::supportsFormatCombination(int32_t pos, const PluginTensorDesc *inOut, 
    int32_t nbInputs, int32_t nbOutputs) {
#ifndef NDEBUG
    std::cout << "CheckShapePlugin::supportsFormatCombination(): this=" << this << std::endl;
#endif

    assert(pos < (nbInputs + nbOutputs) && nbInputs == CHECKSHAPE_PLUGIN_NUM_INPUT && nbOutputs == CHECKSHAPE_PLUGIN_NUM_OUTPUT);

    switch (pos) {
    case 0: // input0
        return inOut[0].type==DataType::kINT32 && inOut[0].format==PluginFormat::kLINEAR;
    case 1: // input1
        return inOut[1].type==DataType::kINT32 && inOut[1].format==PluginFormat::kLINEAR;
    case 2: // output0
        return (inOut[2].type==inOut[0].type) && inOut[2].format==PluginFormat::kLINEAR;
    default:
        return false;
    }
}

inline void CheckShapePlugin::configurePlugin(const DynamicPluginTensorDesc *in, int32_t nbInputs, 
    const DynamicPluginTensorDesc *out, int32_t nbOutputs) {
#ifndef NDEBUG
    std::cout << "CheckShapePlugin::configurePlugin(): this=" << this << std::endl;
#endif
    assert(nbInputs == CHECKSHAPE_PLUGIN_NUM_INPUT);
    assert(nbOutputs == CHECKSHAPE_PLUGIN_NUM_OUTPUT);
    assert(in[0].desc.type == DataType::kINT32);
    assert(in[0].desc.format == PluginFormat::kLINEAR);
    assert(in[0].desc.dims.nbDims == 1); 
    assert(in[1].desc.type == DataType::kINT32);
    assert(in[1].desc.format == PluginFormat::kLINEAR);
    assert(in[1].desc.dims.nbDims == 1); // [batchsize, channel]
    assert(inputRank == in[0].desc.dims.d[0]);
}

inline size_t CheckShapePlugin::getWorkspaceSize(const PluginTensorDesc *inputs, 
    int32_t nbInputs, const PluginTensorDesc *outputs, int32_t nbOutputs) const {
#ifndef NDEBUG
    std::cout << "CheckShapePlugin::getWorkspaceSize(): this=" << this << std::endl;
#endif
    return 0;
}

inline int32_t CheckShapePlugin::enqueue(const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc, 
    const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) {
#ifndef NDEBUG
    std::cout << "CheckShapePlugin::enqueue(): this=" << this << std::endl;
#endif
    int status = 0;

    int32_t * inShape = (int32_t *)inputs[0];
    int32_t * expandShape = (int32_t *)inputs[1];
    int32_t * output = (int32_t *)outputs[0];

    int *h_inShape = (int*)malloc(sizeof(int)*inputRank);
    int *h_expandShape = (int*)malloc(sizeof(int)*expandRank);
    cudaMemcpy(h_inShape, inShape, sizeof(int)*inputRank, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_expandShape, expandShape, sizeof(int)*expandRank, cudaMemcpyDeviceToHost);

    int *h_output = (int*)malloc(sizeof(int)*1);
    *h_output = 0;
    cudaMemcpy(output, h_output, sizeof(int)*expandRank, cudaMemcpyHostToDevice);

    for(int i=expandRank-1; i>=0; i--){
        int index_in = i - (expandRank - inputRank);
        if(index_in >= 0){
            if(h_expandShape[i] != -1){
                if(h_inShape[i] != 1){
                    TRTORCH_CHECK(h_expandShape[i] == h_inShape[index_in], "The expanded size of the tensor (" << \
                    h_expandShape[i] << ") must match the existing size (" << h_inShape[index_in] << \
                    ") at non-singleton dimension " << i << ". Target sizes: [" << h_expandShape << \
                    "].  Tensor sizes: [" << h_inShape << "]");
                }
                
            }
        }else{
            TRTORCH_CHECK(h_expandShape[i] >=0, "The expanded size of the tensor (" << \
            h_expandShape[i] << ") isn't allowed in a leading, non-existing dimension " << \
            i);
        }
    }
    return status;
}

// CheckShapePluginCreator
CheckShapePluginCreator::CheckShapePluginCreator() {
#ifndef NDEBUG
    std::cout << "CheckShapePluginCreator::CheckShapePluginCreator(): this=" << this << std::endl;
#endif
}

inline const char* CheckShapePluginCreator::getPluginName() const {
#ifndef NDEBUG
    std::cout << "CheckShapePluginCreator::getPluginName(): this=" << this << std::endl;
#endif
    return CHECKSHAPE_PLUGIN_NAME; 
}

inline const char* CheckShapePluginCreator::getPluginVersion() const {
#ifndef NDEBUG
    std::cout << "CheckShapePluginCreator::getPluginVersion(): this=" << this << std::endl;
#endif
    return CHECKSHAPE_PLUGIN_VERSION; 
}

inline const PluginFieldCollection* CheckShapePluginCreator::getFieldNames() {
#ifndef NDEBUG
    std::cout << "CheckShapePluginCreator::getFieldNames(): this=" << this << std::endl;
#endif
    std::cout << __FUNCTION__ << std::endl;
    return nullptr;
}

IPluginV2* CheckShapePluginCreator::createPlugin(const char *name, const PluginFieldCollection *fc) {
#ifndef NDEBUG
    std::cout << "CheckShapePluginCreator::createPlugin(): this=" << this << std::endl;
#endif
    const PluginField *fields = fc->fields;
    // parse fields from PluginFieldCollection
    assert(fc->nbFields == 2);
    assert(fields[0].type == PluginFieldType::kINT32);
    int inputRank = *(static_cast<const int *>(fields[0].data));
    assert(fields[1].type == PluginFieldType::kINT32);
    int expandRank = *(static_cast<const int *>(fields[1].data));
    return new CheckShapePlugin(name, inputRank, expandRank);
}

IPluginV2* CheckShapePluginCreator::deserializePlugin(const char *name, const void *serialData, size_t serialLength) {
#ifndef NDEBUG
    std::cout << "CheckShapePluginCreator::deserializePlugin(): this=" << this << std::endl;
#endif
    return new CheckShapePlugin(name, serialData, serialLength);
}

inline void CheckShapePluginCreator::setPluginNamespace(const char *pluginNamespace) {
#ifndef NDEBUG
    std::cout << "CheckShapePluginCreator::setPluginNamespace(): this=" << this << std::endl;
#endif
    mPluginNamespace = pluginNamespace;
}

inline const char* CheckShapePluginCreator::getPluginNamespace() const {
#ifndef NDEBUG
    std::cout << "CheckShapePluginCreator::getPluginNamespace(): this=" << this << std::endl;
#endif
    return mPluginNamespace.c_str();
}

// register plugin
REGISTER_TENSORRT_PLUGIN(CheckShapePluginCreator);

} // namespace plugins
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace trtorch