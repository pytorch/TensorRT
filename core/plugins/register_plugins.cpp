#include "NvInferPlugin.h"
#include "NvInferPluginUtils.h"
#include "core/plugins/impl/interpolate_plugin.h"
#include "core/plugins/impl/normalize_plugin.h"
#include "core/plugins/plugins.h"
#include "core/util/prelude.h"

namespace trtorch {
namespace core {
namespace plugins {
namespace impl {

// Main registry for all flavours of plugins (eg: TRT plugins, TRTorch opensourced plugins)
class TRTorchPluginRegistry {
 public:
  TRTorchPluginRegistry() {
    auto trtorch_logger = util::logging::TRTorchLogger(
        "[TRTorch Plugins Context] - ",
        util::logging::get_logger().get_reportable_severity(),
        util::logging::get_logger().get_is_colored_output_on());
    // register libNvInferPlugins and TRTorch plugins
    initLibNvInferPlugins(&trtorch_logger, "");
    REGISTER_TRTORCH_PLUGIN(InterpolatePluginCreator);
    REGISTER_TRTORCH_PLUGIN(NormalizePluginCreator);
  }
};

namespace {
static TRTorchPluginRegistry plugin_registry;
}

} // namespace impl
} // namespace plugins
} // namespace core
} // namespace trtorch
