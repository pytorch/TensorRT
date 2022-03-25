#pragma once
#include <stdlib.h>
#include <iostream>
#include <sstream>

#ifdef __linux__
#include <linux/limits.h>
#else
#define PATH_MAX 260
#endif

#if defined(_WIN32)
#include <direct.h>
#define getcwd _getcwd
#define realpath(N, R) _fullpath((R), (N), PATH_MAX)
#else
#include <unistd.h>
#endif

#include "NvInfer.h"
#include "third_party/args/args.hpp"
#include "torch/script.h"
#include "torch/torch.h"

#include "torch_tensorrt/logging.h"
#include "torch_tensorrt/ptq.h"
#include "torch_tensorrt/torch_tensorrt.h"

namespace torchtrtc {
namespace fileio {

std::string read_buf(std::string const& path);
std::string get_cwd();
std::string real_path(std::string path);
std::string resolve_path(std::string path);

} // namespace fileio
} // namespace torchtrtc