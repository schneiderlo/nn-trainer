import numpy as np
from PIL import Image
from typing import List, Type, Union


def imread(path: str, dtype: Type) -> np.array:
  """Read an image and store it as a numpy array. """
  return np.array(Image.open(path).convert(mode='L'), dtype=dtype)


def imsave(path: str, data: np.ndarray):
  """Save an image."""
  Image.fromarray(data).save(path)


class NPOperator(object):
  def __init__(self):
    pass

  def apply(self, data: np.ndarray) -> np.ndarray:
    """Apply a transformation to a numpy array.

    Args:
      data: A numpy array. It can either be a 4-D array or a 3-D array.

    Returns:
      The transformed array.
    """
    pass


def transform_data(
    data: np.ndarray,
    np_operators: List[NPOperator]) -> np.ndarray:
  """Apply a series of operations to a numpy array."""
  for op in np_operators:
    data = op.apply(data)
  return data


class ChunkerNPOp(NPOperator):
  """Slice an image into smaller sub-images.

  Remove the outer parts of an image but retain the central region of the image
  along each dimension. If we specify chunk_size = 2, this function
  returns 6 arrays as depicted in the below diagram.

       -------
      |XXYYZZ |
      |XXYYZZ |
      |AABBCC |
      |AABBCC |
       -------

  This function works on a single image (`image` is a 3-D Tensor).
  """

  def __init__(self, chunk_size: int, add_remaining: bool):
    """Initializer.

    Args:
      chunk_size: Size of the sub-arrays.
      add_remaining: Boolean indicating if
    """
    super(ChunkerNPOp, self).__init__()
    self._chunk_size = chunk_size
    self._add_remaining = add_remaining

  def apply(self, data: np.ndarray) -> np.ndarray:
    """Create chunks base on the input array."""
    rank = data.ndim
    if rank != 3:
      raise ValueError('`data` should either be a Tensor with rank = 3.'
                       'Had rank = {}.'.format(rank))

    img_h = data.shape[0]
    img_w = data.shape[1]
    chunk_size = self._chunk_size
    sub_arrays = [
      data[i:i + chunk_size, j:j + chunk_size, :]
      for i in range(0, img_h - chunk_size, chunk_size)
      for j in range(0, img_w - chunk_size, chunk_size)
    ]

    if self._add_remaining:
      if img_h % chunk_size != 0:
        sub_arrays += [
          data[i:i + chunk_size, img_w - chunk_size:img_w, :]
          for i in range(0, img_h - chunk_size, chunk_size)
        ]
      if img_w % chunk_size != 0:
        sub_arrays += [
          data[img_h - chunk_size:img_h, j:j + chunk_size, :]
          for j in range(0, img_w - chunk_size, chunk_size)
        ]

    return np.stack(sub_arrays, axis=0)


class CentralCropNPOp(NPOperator):
  """Crop the central region of the image(s).

  Remove the outer parts of an image but retain the central region of the image
  along each dimension. If we specify central_fraction = 0.5, this function
  returns the region marked with "X" in the below diagram.

       --------
      |        |
      |  XXXX  |
      |  XXXX  |
      |        |   where "X" is the central 50% of the image.
       --------

  This function works on either a single image (`image` is a 3-D Tensor), or a
  batch of images (`image` is a 4-D Tensor).
  """

  def __init__(self, central_fraction: float):
    """Initializer.

    Args:
      central_fraction: float (0, 1], fraction of size to crop.
    """
    super(CentralCropNPOp, self).__init__()
    if central_fraction <= 0.0 or central_fraction > 1.0:
      raise ValueError('central_fraction must be within (0, 1]')
    self._central_fraction = central_fraction

  def apply(self, data: np.ndarray) -> np.ndarray:
    """Apply the crop to the tensor."""
    if self._central_fraction == 1.0:
      return data
    rank = data.ndim
    if rank != 3 and rank != 4:
      raise ValueError('`data` should either be a Tensor with rank = 3 or '
                       'rank = 4. Had rank = {}.'.format(rank))
    if rank == 3:
      img_h = data.shape[0]
      img_w = data.shape[1]
    else:
      img_h = data.shape[1]
      img_w = data.shape[2]

    bbox_h_start = int((img_h - img_h * self._central_fraction) / 2)
    bbox_w_start = int((img_w - img_w * self._central_fraction) / 2)

    bbox_h_end = img_h - bbox_h_start
    bbox_w_end = img_w - bbox_w_start
    if rank == 3:
      bb_slice = (
        slice(bbox_h_start, bbox_h_end),
        slice(bbox_w_start, bbox_w_end),
        slice(None, None)
      )
    else:
      bb_slice = (
        slice(None, None),
        slice(bbox_h_start, bbox_h_end),
        slice(bbox_w_start, bbox_w_end),
        slice(None, None)
      )
    return data[bb_slice]


class CropBorderNPOp(NPOperator):
  """Crop the border of the image(s).

  Remove the outer parts of an image but retain the central region of the image
  along each dimension.

       --------
      |        |
      |  XXXX  |
      |  XXXX  |
      |        |
       --------

  This function works on either a single image (`image` is a 3-D Tensor), or a
  batch of images (`image` is a 4-D Tensor).
  """

  def __init__(self, padding: int):
    """Initializer.

    Args:
      padding: length of the inner area to crop.
    """
    super(CropBorderNPOp, self).__init__()
    self._padding = padding

  def apply(self, data: np.ndarray) -> np.ndarray:
    """Apply the crop to the tensor."""
    if self._padding == 0:
      return data
    rank = data.ndim
    if rank != 3 and rank != 4:
      raise ValueError('`data` should either be a Tensor with rank = 3 or '
                       'rank = 4. Had rank = {}.'.format(rank))
    if rank == 3:
      img_h = data.shape[0]
      img_w = data.shape[1]
    else:
      img_h = data.shape[1]
      img_w = data.shape[2]

    bbox_h_start = self._padding
    bbox_w_start = self._padding

    bbox_h_end = img_h - 2 * bbox_h_start
    bbox_w_end = img_w - 2 * bbox_w_start
    if rank == 3:
      bb_slice = (
        slice(bbox_h_start, bbox_h_end),
        slice(bbox_w_start, bbox_w_end),
        slice(None, None)
      )
    else:
      bb_slice = (
        slice(None, None),
        slice(bbox_h_start, bbox_h_end),
        slice(bbox_w_start, bbox_w_end),
        slice(None, None)
      )
    return data[bb_slice]
