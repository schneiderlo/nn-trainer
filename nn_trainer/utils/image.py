import numpy as np
from PIL import Image


def imread(path: str, dtype: np.dtype) -> np.array:
  """Read an image and store it as a numpy array. """
  return np.array(Image.open(path).convert(mode='L'), dtype=dtype)


def imsave(path: str, data: np.ndarray):
  """Save an image."""
  Image.fromarray(data).save(path)
