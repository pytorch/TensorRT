#include "core/conversion/converters/converters.h"
#include "core/conversion/evaluators/evaluators.h"

#include <string>
#include <sstream>
#include <vector>
#include <iostream>

int main(int argc, const char* argv[]) {
  std::vector<std::string> converters = trtorch::core::conversion::converters::get_converter_list();
  std::vector<std::string> evaluators = trtorch::core::conversion::evaluators::getEvaluatorList();

  std::stringstream ss;

  ss << R"TITLE(
=================================
Operators Supported
=================================

)TITLE";

  ss << R"SEC(
Operators Currently Supported Through Converters
-------------------------------------------------

)SEC";

  for (auto c : converters) {
    ss << "- " << c << std::endl;
  }

  ss << R"SEC(
Operators Currently Supported Through Evaluators
-------------------------------------------------

)SEC";

  for (auto e : evaluators) {
    ss << "- " << e << std::endl;
  }

  std::ofstream ofs;
  ofs.open(argv[1]);

  ofs << ss.rdbuf();

  return 0;
}