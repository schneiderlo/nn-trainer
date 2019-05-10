from os.path import join
import numpy as np
from shutil import copyfile

from nn_trainer.utils.tests import get_test_data
from nn_trainer.utils.image import imread


def get_lenna(np_dtype) -> np.ndarray:
  """Helper function to retrieve Lenna as a numpy array."""
  src_path = get_test_data('images', 'lenna.jpg')
  src = imread(src_path, dtype=np_dtype)
  return src[..., np.newaxis]


def stage_lenna(output_dir, np_dtype) -> [str, np.ndarray]:
  """Helper function to stage Lenna in a directory."""
  src_path = get_test_data('images', 'lenna.jpg')
  dst_path = join(output_dir, 'lenna.png')
  copyfile(src_path, dst_path)
  src = get_lenna(np_dtype)
  return src_path, src
