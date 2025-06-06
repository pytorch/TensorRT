import argparse
import os
import re
import shutil
import subprocess
import warnings
from typing import Tuple

import networkx as nx
import trex
import trex.engine_plan
import trex.graphing


def draw_engine(dir_path: str):
    try:
        import trex
    except ImportError:
        print("trex is required but it is not installed.\n")
        print("Check README.md for installation instructions.")
        exit()

    engine_json_fname = os.path.join(
        dir_path, "_run_on_acc_0_engine_layer_information.json"
    )
    profiling_json_fname = os.path.join(
        dir_path, "_run_on_acc_0_engine_engine_execution_profile.trace"
    )

    graphviz_is_installed = shutil.which("dot") is not None
    if not graphviz_is_installed:
        print("graphviz is required but it is not installed.\n")
        print("To install on Ubuntu:")
        print("sudo apt --yes install graphviz")
        exit()

    plan = trex.engine_plan.EnginePlan(
        engine_json_fname, profiling_file=profiling_json_fname
    )
    layer_node_formatter = trex.graphing.layer_type_formatter
    graph = trex.graphing.to_dot(plan, layer_node_formatter)
    output_format = "png"  # svg or jpg

    trex.graphing.render_dot(graph, engine_json_fname, output_format)
