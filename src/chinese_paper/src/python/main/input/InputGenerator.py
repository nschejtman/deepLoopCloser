import math
from functools import partial
from glob import glob

from pathlib2 import Path

from input.CvInputParser import CvInputParser as Parser


def get_generator(file_pattern: str, shape: list):
    n_patches = shape[0]
    patch_size = int(math.sqrt(shape[1]))
    return partial(iteration, file_pattern, n_patches, patch_size)


def iteration(file_pattern: str, n_patches: int, patch_size: int):
    resolved_path = str(Path(file_pattern).resolve())
    files = glob(resolved_path)
    files.sort()

    parser = Parser(n_patches, patch_size)
    for file in files:
        yield parser.parse_from_path(file)
