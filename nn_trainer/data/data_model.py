import os
import tensorflow as tf
import pandas as pd
import numpy as np
from sys import getsizeof
from typing import Dict, List, Union
from tqdm import tqdm

from nn_trainer.data.item_handler import ItemHandler


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
    for item_handler in self._item_handlers:
      parsed_example = item_handler.deserialization_post_processing(parsed_example)
    return parsed_example

  def serialize(self, data: Union[Dict, pd.Series]) -> tf.train.Example:
    """Serialize item into a `tf.train.Example`."""

    def serialize_fn(handler: ItemHandler):
      if isinstance(data, dict):
        return handler.serialize(data=data[handler.key])
      elif isinstance(data, pd.Series):
        return handler.serialize_from_series(data=data)
      else:
        raise NameError('Data to serialize must either be a dict or a `pd.Series`.')

    features = {}
    for item_handler in self._item_handlers:
      item_serialized = serialize_fn(item_handler)
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

  def export_to_tfrecord(self,
                         dataframe: pd.DataFrame,
                         output_dir: str,
                         max_size: float):
    """Export a `pd.DataFrame` to one or more tfrecord files.

    This method map data extracted from a `pd.DataFrame` into a series of
    tfrecord files which size does not exceed a defined amount.

    Args:
      dataframe: A a `pd.DataFrame` containing items to be serialized.
      output_dir: Directory where the output tfrecord will be saved.
      max_size: Maximum size of a tfrecord.
    """
    df = dataframe.reset_index(drop=True)
    column_needed = [item_handler.key for item_handler in self._item_handlers]
    column_provided = dataframe.columns.values.tolist()
    if not set(column_needed).issubset(set(column_provided)):
      error_str = "Columns not found in DataFrame. Compulsory column: '{}', given: '{}".format(
        column_needed,
        column_provided
      )
      raise LookupError(error_str)

    # Estimate the number of files that will be created.
    first_row = df.iloc[0]
    serialize_row = self.serialize(first_row).SerializeToString()
    data_size_mb = getsizeof(serialize_row) * 1e-6
    num_items = len(df)
    expected_size_mb = num_items * data_size_mb

    num_shards = int(expected_size_mb // max_size) + 1
    print("A serialized data takes {:.2f} MB.".format(data_size_mb))
    print("The total expected size {:.2f} MB.\n".format(expected_size_mb))

    # Serialize data into a series of tfrecord files.
    os.makedirs(output_dir, exist_ok=True)
    splitted_index = np.array_split(range(0, num_items), num_shards)
    for j, indexes in enumerate(splitted_index):
      tfrecord_path = os.path.join(output_dir, 'shard_{}.tfr'.format(j))
      with tf.io.TFRecordWriter(tfrecord_path) as writer:
        for i in tqdm(indexes):
          row = df.iloc[i]
          serialized_data = self.serialize(data=row)
          serialized_data_str = serialized_data.SerializeToString()
          writer.write(serialized_data_str)
