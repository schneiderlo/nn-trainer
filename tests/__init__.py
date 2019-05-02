import numpy as np
from PIL import Image

from nn_trainer.utils.tests import get_test_data


def get_lenna(np_dtype) -> np.ndarray:
  """Helper function to retrieve Lenna as a numpy array."""
  src_path = get_test_data('images', 'lenna.jpg')
  src = np.array(Image.open(src_path).convert(mode='L'), dtype=np_dtype)
  return src[..., np.newaxis]
