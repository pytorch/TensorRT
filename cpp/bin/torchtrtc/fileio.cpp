#include "fileio.h"

namespace torchtrtc {
namespace fileio {

std::string read_buf(std::string const& path) {
  std::string buf;
  std::ifstream stream(path.c_str(), std::ios::binary);

  if (stream) {
    stream >> std::noskipws;
    std::copy(std::istream_iterator<char>(stream), std::istream_iterator<char>(), std::back_inserter(buf));
  }

  return buf;
}

std::string get_cwd() {
  char buff[FILENAME_MAX]; // create string buffer to hold path
  if (getcwd(buff, FILENAME_MAX)) {
    std::string current_working_dir(buff);
    return current_working_dir;
  } else {
    torchtrt::logging::log(torchtrt::logging::Level::kERROR, "Unable to get current directory");
    exit(1);
  }
}

std::string real_path(std::string path) {
  auto abs_path = path;
  char real_path_c[PATH_MAX];
  char* res = realpath(abs_path.c_str(), real_path_c);
  if (res) {
    return std::string(real_path_c);
  } else {
    torchtrt::logging::log(torchtrt::logging::Level::kERROR, std::string("Unable to find file ") + abs_path);
    exit(1);
  }
}

std::string resolve_path(std::string path) {
  auto rpath = path;
  if (!(rpath.rfind("/", 0) == 0)) {
    rpath = get_cwd() + '/' + rpath;
  }
  return rpath;
}

} // namespace fileio
} // namespace torchtrtc