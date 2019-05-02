import tensorflow as tf
from typing import Dict, List

from .item_handler import ItemHandler, NPArrayHandler


class DataModel(object):
  """Interface to serialize/deserialize objects to/from tfrecords."""

  def __init__(self, item_handlers: List[ItemHandler]):
    self._item_handlers = item_handlers

  def _deserialize_dict(self) -> Dict:
    """Format a dict to be used by `tf.parse_single_example`.

    Returns:
      A `dict` mapping feature keys to `FixedLenFeature` or
      `VarLenFeature` values.
    """
    dict_features = {}
    for item_handler in self._item_handlers:
      new_dict = item_handler._deserialize_dict()
      dict_features.update(new_dict)
    return dict_features

  def deserialize(self, record: str) -> Dict[str, tf.Tensor]:
    """Deserialize item into a dict of Tensors."""
    parsed_example = tf.io.parse_single_example(
      serialized=record,
      features=self._deserialize_dict()
    )
    # If an item is an image convert bytes to tf.Tensor.
    for item_handler in self._item_handlers:
      if isinstance(item_handler, NPArrayHandler):
        np_key = item_handler.key
        height = parsed_example[item_handler._height_key()]
        width = parsed_example[item_handler._width_key()]
        depth = parsed_example[item_handler._depth_key()]
        parsed_example[np_key] = tf.io.decode_raw(
          bytes=parsed_example[np_key],
          out_type=item_handler.tf_dtype
        )
        parsed_example[np_key] = tf.reshape(
          tensor=parsed_example[np_key],
          shape=[height, width, depth]
        )
    return parsed_example

  def serialize(self, data: Dict) -> tf.train.Example:
    """Serialize item into a `tf.train.Example`."""
    features = {}
    for item_handler in self._item_handlers:
      item_data = data[item_handler.key]
      item_serialized = item_handler.serialize(item_data)
      features.update(item_serialized)
    return tf.train.Example(
      features=tf.train.Features(
        feature=features
      )
    )

  def to_tfrecord(self, tfrecord_path: str, datas: List[Dict]):
    """Serialize of list of data model into a tfrecord file.

    Args:
      tfrecord_path: Path where the tfrecord is saved.
      datas: The list of datas to serialize the datas. It must respect
             the data model schema.
    """
    with tf.io.TFRecordWriter(tfrecord_path) as writer:
      for data in datas:
        serialized_data = self.serialize(data=data)
        serialized_data_str = serialized_data.SerializeToString()
        writer.write(serialized_data_str)
