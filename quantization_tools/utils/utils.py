import torch
from tqdm import tqdm

from ..quantization import INT_TO_PRECISION
from ..quantization.utils import replace_module, find_layers
from ..quantization.layers import LinearQuantHub
