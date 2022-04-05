#include "parser_util.h"

namespace torchtrtc {
namespace parserutil {

torchtrt::TensorFormat parse_tensor_format(std::string str) {
  std::transform(str.begin(), str.end(), str.begin(), [](unsigned char c) { return std::tolower(c); });

  if (str == "linear" || str == "nchw" || str == "chw" || str == "contiguous") {
    return torchtrt::TensorFormat::kContiguous;
  } else if (str == "nhwc" || str == "hwc" || str == "channels_last") {
    return torchtrt::TensorFormat::kChannelsLast;
  } else {
    torchtrt::logging::log(
        torchtrt::logging::Level::kERROR,
        "Invalid tensor format, options are [ linear | nchw | chw | contiguous | nhwc | hwc | channels_last ], found: " +
            str);
    return torchtrt::TensorFormat::kUnknown;
  }
}

torchtrt::DataType parse_dtype(std::string dtype_str) {
  std::transform(
      dtype_str.begin(), dtype_str.end(), dtype_str.begin(), [](unsigned char c) { return std::tolower(c); });
  if (dtype_str == "float" || dtype_str == "float32" || dtype_str == "f32" || dtype_str == "fp32") {
    return torchtrt::DataType::kFloat;
  } else if (dtype_str == "half" || dtype_str == "float16" || dtype_str == "f16" || dtype_str == "fp16") {
    return torchtrt::DataType::kHalf;
  } else if (dtype_str == "char" || dtype_str == "int8" || dtype_str == "i8") {
    return torchtrt::DataType::kChar;
  } else if (dtype_str == "int" || dtype_str == "int32" || dtype_str == "i32") {
    return torchtrt::DataType::kInt;
  } else if (dtype_str == "bool" || dtype_str == "b") {
    return torchtrt::DataType::kBool;
  } else {
    torchtrt::logging::log(
        torchtrt::logging::Level::kERROR,
        "Invalid precision, options are [ float | float32 | fp32 | f32 | half | float16 | fp16 | f16 | char | int8 | i8 | int | int32 | i32 | bool | b], found: " +
            dtype_str);
    return torchtrt::DataType::kUnknown;
  }
}

std::vector<int64_t> parse_single_dim(std::string shape_str) {
  std::vector<int64_t> shape;
  std::stringstream ss;
  for (auto c : shape_str) {
    if (c == '(' || c == ' ') {
      continue;
    } else if (c == ',') {
      int64_t dim;
      ss >> dim;
      shape.push_back(dim);
      ss.clear();
    } else if (c == ')') {
      int64_t dim;
      ss >> dim;
      shape.push_back(dim);
      ss.clear();
      return shape;
    } else {
      ss << c;
    }
  }

  torchtrt::logging::log(
      torchtrt::logging::Level::kERROR,
      "Shapes need dimensions delimited by comma in parentheses, \"(N,..,C,H,W)\"\n e.g \"(3,3,200,200)\"");
  exit(1);
  return {};
}

std::vector<std::vector<int64_t>> parse_dynamic_dim(std::string shape_str) {
  shape_str = shape_str.substr(1, shape_str.size() - 2);
  std::vector<std::vector<int64_t>> shape;
  std::stringstream ss;

  std::string delimiter = ";";

  size_t pos = 0;
  while ((pos = shape_str.find(delimiter)) != std::string::npos) {
    auto token = shape_str.substr(0, pos);
    auto range = parse_single_dim(token);
    shape_str.erase(0, pos + delimiter.length());
    shape.push_back(range);
  }

  auto range = parse_single_dim(shape_str);
  shape.push_back(range);

  if (shape.size() != 3) {
    torchtrt::logging::log(
        torchtrt::logging::Level::kERROR,
        "Dynamic shapes need three sets of dimensions delimited by semi-colons, \"[(MIN_N,..,MIN_C,MIN_H,MIN_W);(OPT_N,..,OPT_C,OPT_H,OPT_W);(MAX_N,..,MAX_C,MAX_H,MAX_W)]\"\n e.g \"[(3,3,100,100);(3,3,200,200);(3,3,300,300)]\"");
    exit(1);
  }

  return shape;
}

torchtrt::Input parse_input(std::string spec) {
  const std::string spec_err_str =
      "Dimensions should be specified in one of these types \"(N,..,C,H,W)\" \"[(MIN_N,..,MIN_C,MIN_H,MIN_W);(OPT_N,..,OPT_C,OPT_H,OPT_W);(MAX_N,..,MAX_C,MAX_H,MAX_W)]\"\n e.g \"(3,3,300,300)\" \"[(3,3,100,100);(3,3,200,200);(3,3,300,300)]\"\nTo specify input type append an @ followed by the precision\n e.g. \"(3,3,300,300)@f32\"\nTo specify input format append an \% followed by the format [contiguous | channel_last]\n e.g. \"(3,3,300,300)@f32\%channel_last\"";
  std::string shapes;
  std::string dtype;
  std::string format;
  // THERE IS A SPEC FOR DTYPE
  if (spec.find('@') != std::string::npos) {
    // THERE IS ALSO A SPEC FOR FORMAT
    if (spec.find('%') != std::string::npos) {
      auto dtype_delim = spec.find('@');
      auto format_delim = spec.find('%');
      std::string shapes = spec.substr(0, dtype_delim);
      std::string dtype = spec.substr(dtype_delim + 1, format_delim - (dtype_delim + 1));
      std::string format = spec.substr(format_delim + 1, spec.size());

      auto parsed_dtype = parse_dtype(dtype);
      if (parsed_dtype == torchtrt::DataType::kUnknown) {
        torchtrt::logging::log(torchtrt::logging::Level::kERROR, "Invalid datatype for input specification " + spec);
        exit(1);
      }
      auto parsed_format = parse_tensor_format(format);
      if (parsed_format == torchtrt::TensorFormat::kUnknown) {
        torchtrt::logging::log(torchtrt::logging::Level::kERROR, "Invalid format for input specification " + spec);
        exit(1);
      }
      if (shapes.rfind("(", 0) == 0) {
        return torchtrt::Input(parse_single_dim(shapes), parsed_dtype, parsed_format);
      } else if (shapes.rfind("[", 0) == 0) {
        auto dyn_shapes = parse_dynamic_dim(shapes);
        return torchtrt::Input(dyn_shapes[0], dyn_shapes[1], dyn_shapes[2], parsed_dtype, parsed_format);
      } else {
        torchtrt::logging::log(torchtrt::logging::Level::kERROR, spec_err_str);
        exit(1);
      }
      // THERE IS NO SPEC FOR FORMAT
    } else {
      std::string shapes = spec.substr(0, spec.find('@'));
      std::string dtype = spec.substr(spec.find('@') + 1, spec.size());

      auto parsed_dtype = parse_dtype(dtype);
      if (parsed_dtype == torchtrt::DataType::kUnknown) {
        torchtrt::logging::log(torchtrt::logging::Level::kERROR, "Invalid datatype for input specification " + spec);
        exit(1);
      }
      if (shapes.rfind("(", 0) == 0) {
        return torchtrt::Input(parse_single_dim(shapes), parsed_dtype);
      } else if (shapes.rfind("[", 0) == 0) {
        auto dyn_shapes = parse_dynamic_dim(shapes);
        return torchtrt::Input(dyn_shapes[0], dyn_shapes[1], dyn_shapes[2], parsed_dtype);
      } else {
        torchtrt::logging::log(torchtrt::logging::Level::kERROR, spec_err_str);
        exit(1);
      }
    }
    // THERE IS A SPEC FOR FORMAT BUT NOT DTYPE
  } else if (spec.find('%') != std::string::npos) {
    std::string shapes = spec.substr(0, spec.find('%'));
    std::string format = spec.substr(spec.find('%') + 1, spec.size());

    auto parsed_format = parse_tensor_format(format);
    if (parsed_format == torchtrt::TensorFormat::kUnknown) {
      torchtrt::logging::log(torchtrt::logging::Level::kERROR, "Invalid format for input specification " + spec);
      exit(1);
    }
    if (shapes.rfind("(", 0) == 0) {
      return torchtrt::Input(parse_single_dim(shapes), parsed_format);
    } else if (shapes.rfind("[", 0) == 0) {
      auto dyn_shapes = parse_dynamic_dim(shapes);
      return torchtrt::Input(dyn_shapes[0], dyn_shapes[1], dyn_shapes[2], parsed_format);
    } else {
      torchtrt::logging::log(torchtrt::logging::Level::kERROR, spec_err_str);
      exit(1);
    }
    // JUST SHAPE USE DEFAULT DTYPE
  } else {
    if (spec.rfind("(", 0) == 0) {
      return torchtrt::Input(parse_single_dim(spec));
    } else if (spec.rfind("[", 0) == 0) {
      auto dyn_shapes = parse_dynamic_dim(spec);
      return torchtrt::Input(dyn_shapes[0], dyn_shapes[1], dyn_shapes[2]);
    } else {
      torchtrt::logging::log(torchtrt::logging::Level::kERROR, spec_err_str);
      exit(1);
    }
  }
}

} // namespace parserutil
} // namespace torchtrtc