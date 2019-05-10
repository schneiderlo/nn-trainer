import tensorflow as tf
import numpy as np
import pandas as pd
from typing import Dict, List, Union

from nn_trainer.utils.image import imread


def _int_feature(value: Union[bool, int, List[Union[bool, int]]]) -> tf.train.Feature:
  """Return a int64_list from a bool, an enum, an integer."""
  if not isinstance(value, (tuple, list)):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value: Union[float, List[float]]) -> tf.train.Feature:
  """Returns a float_list from a float / double."""
  if not isinstance(value, (tuple, list)):
    value = [value]
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


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

  def deserialization_post_processing(self, parsed_example: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
    """Step to finalize the item deserialization.

    This function is typically run after a string record have been parsed to
    obtain the proper deserialized item.
    """
    return parsed_example

  def serialize(self, data) -> Dict:
    """Serialize item into a key-value pair."""
    pass

  def serialize_from_series(self, data: pd.Series):
    """Function used to convert a pd.Series to an `ItemHandler`."""
    raise NotImplementedError("ItemHandler.prepare_data cannot be used by base class ItemHandler.")


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
    self._np_dtype = np_dtype
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

  def deserialization_post_processing(self, parsed_example: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
    """Step to finalize the item deserialization.

    This function is typically run after a string record have been parsed to
    obtain the proper deserialized item.
    """
    np_key = self.key
    height = parsed_example[self._height_key()]
    width = parsed_example[self._width_key()]
    depth = parsed_example[self._depth_key()]
    parsed_example[np_key] = tf.io.decode_raw(
      bytes=parsed_example[np_key],
      out_type=self.tf_dtype
    )
    parsed_example[np_key] = tf.reshape(
      tensor=parsed_example[np_key],
      shape=[height, width, depth]
    )
    return parsed_example

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

  def serialize_from_series(self, data: pd.Series) -> Dict:
    """Load an image with its path extracted from a pandas Series."""
    src = imread(path=data[self.key], dtype=self._np_dtype)
    src = src[..., np.newaxis]
    return self.serialize(src)


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

  def serialize(self, data: str) -> Dict:
    """Serialize item into a key-value pair."""
    if not isinstance(data, str):
      raise ValueError('Serialization error. Data provided to '
                       'TextHandler is not string.')
    data = str.encode(data)
    return {
      self._key: _bytes_feature(data)
    }

  def serialize_from_series(self, data: pd.Series) -> Dict:
    """Load a text with its path extracted from a pandas Series."""
    return self.serialize(data[self.key])


class FloatMetricHandler(ItemHandler):
  """An ItemHandler that take care of metrics."""

  def __init__(self, key):
    """Initializes the ImageHandler."""
    super(FloatMetricHandler, self).__init__(key=key)

  def _deserialize_dict(self) -> Dict:
    """Format a dict to be used by `tf.parse_single_example`.

    Returns:
      A `dict` mapping feature keys to `FixedLenFeature` or
      `VarLenFeature` values.
    """
    return {
      self._key: tf.io.FixedLenFeature([], tf.float32),
    }

  def serialize(self, data: List[float]) -> Dict:
    """Serialize item into a key-value pair."""
    return {
      self._key: _float_feature(data)
    }

  def serialize_from_series(self, data: pd.Series) -> Dict:
    """Load a text with its path extracted from a pandas Series."""
    return self.serialize(data[self.key])
