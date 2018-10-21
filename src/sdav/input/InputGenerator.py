import math
from functools import partial
from glob import glob
import logging
from pathlib2 import Path

from src.sdav.input.CvInputParser import CvInputParser as Parser


def get_generator(file_pattern: str, shape: list):
    n_patches = shape[0]
    patch_size = int(math.sqrt(shape[1]))
    return partial(iteration, file_pattern, n_patches, patch_size)


def iteration(file_pattern: str, n_patches: int, patch_size: int):
    resolved_path = str(Path(file_pattern).resolve())
    files = glob(resolved_path)

    if len(files) == 0:
        logger = logging.getLogger()
        logger.error("Specified dataset is empty or could not find dataset")

    parser = Parser(n_patches, patch_size)
    for file in files:
        yield parser.parse_from_path(file)
