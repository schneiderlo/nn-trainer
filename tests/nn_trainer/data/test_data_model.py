import os
import numpy as np
import tensorflow as tf
import pytest

from tests import get_lenna
from nn_trainer.utils.tests import get_test_directory
from nn_trainer.data.item_handler import ItemHandler, NPArrayHandler, TextHandler
from nn_trainer.data.data_model import DataModel


NUMPY_DTYPE = [
  np.uint8,
  np.uint16,
  np.float16,
  np.float32,
  np.float64,
]


@pytest.mark.parametrize('np_dtype', NUMPY_DTYPE)
def test_serialization(request, np_dtype):
  src = get_lenna(np_dtype=np_dtype)
  data_model = DataModel(
    item_handlers=[
      NPArrayHandler('source', np_dtype)
    ]
  )
  serialized_data = data_model.serialize(data={'source': src})
  serialized_data_str = serialized_data.SerializeToString()
  parsed_example = data_model.deserialize(serialized_data_str)
  np.testing.assert_array_equal(src, parsed_example['source'].numpy())


@pytest.mark.parametrize('np_dtype', NUMPY_DTYPE)
def test_to_tfrecords(request, np_dtype):
  output_dir = get_test_directory(request)
  src = get_lenna(np_dtype=np_dtype)
  data_model = DataModel(
    item_handlers=[
      NPArrayHandler('source', np_dtype)
    ]
  )
  tfrecord_path = os.path.join(output_dir, './lenna.tfr')
  data_model.to_tfrecord(
    tfrecord_path=tfrecord_path,
    datas=[{'source': src}]
  )
  # Deserialize data and check data.
  dataset = tf.data.Dataset.from_tensor_slices([tfrecord_path])
  dataset = tf.data.TFRecordDataset(filenames=dataset)
  dataset = dataset.map(
    data_model.deserialize, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  item = next(iter(dataset))
  np.testing.assert_array_equal(src, item['source'].numpy())
  assert item['source/height'].numpy() == 729
  assert item['source/width'].numpy() == 740
  assert item['source/depth'].numpy() == 1
