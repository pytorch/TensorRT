#include "core/util/prelude.h"
#include "core/plugins/plugins.h"
// #include "core/plugins/impl/interpolate_plugin.h"
// #include "core/plugins/impl/normalize_plugin.h"
#include "NvInferPlugin.h"
#include "NvInferPluginUtils.h"

namespace trtorch{
namespace core{
namespace plugins{

class TRTorchPluginRegistry {
 public:
  TRTorchPluginRegistry(){
    // initialize initLibNvInferPlugins
    auto trtorch_logger = util::logging::TRTorchLogger("[TRTorch Plugins Context] - ",
        util::logging::get_logger().get_reportable_severity(),
        util::logging::get_logger().get_is_colored_output_on());
    initLibNvInferPlugins(&trtorch_logger, "trtorch");
  }
};

namespace {
static TRTorchPluginRegistry plugin_registry;
}

}
}
}
