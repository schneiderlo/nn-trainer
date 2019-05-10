import os
import numpy as np
import pandas as pd
import tensorflow as tf
import pytest

from tests import get_lenna, stage_lenna
from nn_trainer.utils.tests import get_test_directory, get_test_data
from nn_trainer.data.item_handler import ItemHandler, NPArrayHandler, TextHandler, \
  FloatMetricHandler
from nn_trainer.data.data_model import DataModel


NUMPY_DTYPE = [
  # np.uint8,
  # np.uint16,
  # np.float16,
  np.float32,
  # np.float64,
]


@pytest.mark.parametrize('np_dtype', NUMPY_DTYPE)
def test_serialization(request, np_dtype):
  src = get_lenna(np_dtype=np_dtype)
  data_model = DataModel(
    item_handlers=[
      NPArrayHandler('source', np_dtype),
      TextHandler('description')
    ]
  )
  serialized_data = data_model.serialize(data={
    'source': src,
    'description': 'my description',
  })
  serialized_data_str = serialized_data.SerializeToString()
  parsed_example = data_model.deserialize(serialized_data_str)
  np.testing.assert_array_equal(src, parsed_example['source'].numpy())
  assert parsed_example['description'].numpy() == b'my description'


@pytest.mark.parametrize('metric_value', [1, 0, 0.1])
@pytest.mark.parametrize('np_dtype', NUMPY_DTYPE)
def test_serialization_from_series(request, metric_value, np_dtype):
  output_dir = get_test_directory(request)
  lenna_path, src = stage_lenna(output_dir, np_dtype)
  metric_value = np.float32(metric_value)
  series = pd.Series(data={
    'source': lenna_path,
    'description': 'my description',
    'metric': metric_value
  })
  data_model = DataModel(
    item_handlers=[
      NPArrayHandler('source', np_dtype),
      FloatMetricHandler('metric'),
      TextHandler('description')
    ]
  )
  serialized_data = data_model.serialize(series)
  serialized_data_str = serialized_data.SerializeToString()
  parsed_example = data_model.deserialize(serialized_data_str)
  np.testing.assert_array_equal(src, parsed_example['source'].numpy())
  assert parsed_example['metric'].numpy() == metric_value
  assert parsed_example['description'].numpy() == b'my description'


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


@pytest.mark.parametrize('num_items', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# @pytest.mark.parametrize('num_items', [9])
def test_export_to_tfrecord(request, num_items):
  np_dtype = np.float32
  metric_value = 0.1
  output_dir = get_test_directory(request)
  lenna_path, src = stage_lenna(output_dir, np_dtype)
  metric_value = np.float32(metric_value)
  df = pd.DataFrame(data={
    'source': [lenna_path for _ in range(0, num_items)],
    'description': ['my description' for _ in range(0, num_items)],
    'metric': [metric_value for _ in range(0, num_items)],
  })
  data_model = DataModel(
    item_handlers=[
      NPArrayHandler('source', np_dtype),
      FloatMetricHandler('metric'),
      TextHandler('description')
    ]
  )

  data_model.export_to_tfrecord(
    output_dir=output_dir,
    dataframe=df,
    max_size=5
  )
