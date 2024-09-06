import argparse
import itertools
import os

import numpy as np

from diffcbed.envs.dream4 import OnlyDAGDream4Environment
from graphs.graph_dream import Dream4Graph
from scripts.base_script import parse_args

# args = argparse.Namespace(scm_bias=0.0, noise_bias=0.0, old_er_logic=True)
# env = OnlyDAGDream4Environment(args)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

graph = Dream4Graph(yml_name="InSilicoSize10-Ecoli1")
