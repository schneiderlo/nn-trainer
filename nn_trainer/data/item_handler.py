import tensorflow as tf
import numpy as np
from typing import Dict, List, Union


def _int_feature(value: Union[bool, int, List[Union[bool, int]]]) -> tf.train.Feature:
  """Return a int64_list from a bool, an enum, an integer."""
  if not isinstance(value, (tuple, list)):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value: float) -> tf.train.Feature:
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _bytes_feature(value: bytes) -> tf.train.Feature:
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy()
  if not isinstance(value, (tuple, list)):
    value = [value]
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


class ItemHandler(object):
  """Specifies the item-to-Features mapping for tf.parse_example.

  An ItemHandler both specifies a list of Features used for parsing an Example
  proto as well as a function that post-processes the results of Example
  parsing.
  """

  def __init__(self, key):
    """Initializes the ItemHandler.

    Args:
      keys: the name of the TensorFlow Example Feature.
    """
    self._key = key

  @property
  def key(self):
    return self._key

  def _deserialize_dict(self) -> Dict:
    """Format a dict to be used by `tf.parse_single_example`.

    Returns:
      A `dict` mapping feature keys to `FixedLenFeature` or
      `VarLenFeature` values.
    """
    pass

  def deserialize(self, record: str) -> Dict:
    """Deserialize a record and produce a Dictionary of `tf.Tensor`."""
    pass

  def serialize(self, data) -> Dict:
    """Serialize item into a key-value pair."""
    pass


class NPArrayHandler(ItemHandler):
  """An ItemHandler that take care of numpy array."""

  nptype_to_tftype = {
    np.uint8: tf.uint8,
    np.uint16: tf.uint16,
    # np.uint32: tf.uint32, Does not exist.
    np.float16: tf.float16,
    np.float32: tf.float32,
    np.float64: tf.float64,
  }

  def __init__(self, key, np_dtype):
    """Initializes the ImageHandler."""
    super(NPArrayHandler, self).__init__(key=key)
    self._tf_dtype = self.nptype_to_tftype[np_dtype]

  @property
  def tf_dtype(self):
    return self._tf_dtype

  def _height_key(self):
    """Shorcut to the key for height."""
    return self._key + '/height'

  def _width_key(self):
    """Shorcut to the key for width."""
    return self._key + '/width'

  def _depth_key(self):
    """Shorcut to the key for depth."""
    return self._key + '/depth'

  def _deserialize_dict(self) -> Dict:
    """Format a dict to be used by `tf.parse_single_example`.

    Returns:
      A `dict` mapping feature keys to `FixedLenFeature` or
      `VarLenFeature` values.
    """
    return {
      self._key: tf.io.FixedLenFeature([], tf.string),
      self._height_key(): tf.io.FixedLenFeature([], tf.int64),
      self._width_key(): tf.io.FixedLenFeature([], tf.int64),
      self._depth_key(): tf.io.FixedLenFeature([], tf.int64)
    }

  def serialize(self, data: np.ndarray) -> Dict:
    """Serialize item into a key-value pair."""
    if not isinstance(data, np.ndarray):
      raise ValueError('Serialization error. Data provided to '
                       'NPArrayHandler is not a numpy array.')
    height, width, depth = data.shape
    return {
      self._key: _bytes_feature(data.tostring()),
      self._height_key(): _int_feature(height),
      self._width_key(): _int_feature(width),
      self._depth_key(): _int_feature(depth)
    }


class TextHandler(ItemHandler):
  """An ItemHandler that take care of text."""

  def __init__(self, key):
    """Initializes the ImageHandler."""
    super(TextHandler, self).__init__(key=key)

  def _deserialize_dict(self) -> Dict:
    """Format a dict to be used by `tf.parse_single_example`.

    Returns:
      A `dict` mapping feature keys to `FixedLenFeature` or
      `VarLenFeature` values.
    """
    return {
      self._key: tf.io.FixedLenFeature([], tf.string),
    }

  def deserialize(self, record: str) -> Dict:
    """Deserialize a record and produce a Dictionary of `tf.Tensor`."""
    pass

  def serialize(self, data: str) -> Dict:
    """Serialize item into a key-value pair."""
    if not isinstance(data, str):
      raise ValueError('Serialization error. Data provided to '
                       'TextHandler is not string.')
    data = str.encode(data)
    return {
      self._key: _bytes_feature(data)
    }
