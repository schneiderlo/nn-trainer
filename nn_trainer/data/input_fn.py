import os
import glob
import tensorflow as tf
from typing import List

from nn_trainer.data.data_model import DataModel


class InputPipeline(object):

  def __init__(self,
               data_model: DataModel,
               num_epochs: int,
               num_samples: int,
               batch_size: int,):
    self._data_model = data_model
    self._num_epochs = num_epochs
    self._num_samples = num_samples
    self._batch_size = batch_size
    self._autotune = tf.data.experimental.AUTOTUNE

  def create_dataset(self,
                     tfrecord_paths: List[str],
                     is_training: bool) -> tf.data.Dataset:
    """Create a `tf.Dataset` to be used either in training or in

    Args:
      tfrecord_paths: List of tfrecord files. For instance
                      `tfrecord_paths=['/my/dir/f1.tfr', '/my/dir/f2.tfr', ...]`
      is_training: Set to True if the dataset is used for training.
    Returns: A `tf.Dataset`.
    """
    filenames = tf.data.Dataset.from_tensor_slices(tfrecord_paths)
    dataset = tf.data.TFRecordDataset(
        filenames=filenames, num_parallel_reads=6)
    dataset = dataset.map(self._data_model.deserialize, num_parallel_calls=self._autotune)

    if is_training:
      dataset = dataset.shuffle(buffer_size=self._num_samples)
      num_repeat = self._num_epochs
    else:
      num_repeat = 1

    dataset = (dataset.repeat(count=num_repeat)
               .batch(batch_size=self._batch_size, drop_remainder=True)
               .prefetch(buffer_size=self._autotune))
    return dataset


def load_shards_from_dir(shard_directory: str):
  """Load shards to be used by tensorflow.

  Args:
    shard_directory: Directory containing tfrecords.
  Return: A list of paths to tfrecords.
  """
  list_filenames = glob.iglob(f"{shard_directory}/**/*.tfr", recursive=True)
  return [os.path.join(shard_directory, f)
          for f in list_filenames]


def get_number_samples(shard_directory: str) -> int:
  """Return the number of samples stored in a directory. """
  list_shards = load_shards_from_dir(shard_directory)

  def parse_record(record):
    return record

  num_samples = 0
  for tfrecords_path in list_shards:
    ds = tf.data.TFRecordDataset(tfrecords_path)
    ds = ds.map(parse_record)
    ds = ds.batch(1)

    for _ in ds:
        num_samples += 1
  return num_samples
