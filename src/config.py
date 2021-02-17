import os

from torch import cuda
from loguru import logger

_is_cuda_available = cuda.is_available()
logger.debug("is cuda available? {}", _is_cuda_available)
if _is_cuda_available:
    logger.debug("device count: {}", cuda.device_count())

DEVICE = "cuda" if _is_cuda_available else "cpu"
if os.getenv("DEVICE") is not None:
    # Accept override.
    DEVICE = os.getenv("DEVICE")
logger.info("using device: {}", DEVICE)

AIP_STORAGE_URI = os.getenv("AIP_STORAGE_URI")
IS_TEST = os.getenv("IS_TEST") == "true"
