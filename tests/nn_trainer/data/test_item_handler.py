import numpy as np
import tensorflow as tf
import pytest

from tests import get_lenna
from nn_trainer.data.item_handler import ItemHandler, NPArrayHandler, TextHandler, FloatMetricHandler


NUMPY_DTYPE = [
  np.uint8,
  np.uint16,
  np.float16,
  np.float32,
  np.float64,
]


def get_example_proto(item_handler: ItemHandler, data) -> str:
  """Helper function to convert a data to a proto string."""
  serialize_item = item_handler.serialize(data=data)
  proto_example = tf.train.Example(
    features=tf.train.Features(
      feature=serialize_item
    )
  )
  return proto_example.SerializeToString()


@pytest.mark.parametrize('np_dtype', NUMPY_DTYPE)
def test_nparrayhandler(request, np_dtype):
  src = get_lenna(np_dtype=np_dtype)
  nparray_item = NPArrayHandler('source', np_dtype=np_dtype)
  proto_str = get_example_proto(item_handler=nparray_item, data=src)
  parsed_example = tf.io.parse_single_example(
    serialized=proto_str,
    features=nparray_item._deserialize_dict()
  )
  assert parsed_example['source/height'].numpy() == 729
  assert parsed_example['source/width'].numpy() == 740
  assert parsed_example['source/depth'].numpy() == 1


def test_texthandler(request):
  text_item = TextHandler('item-1')
  proto_str = get_example_proto(item_handler=text_item, data='Some texts to print.')
  parsed_example = tf.io.parse_single_example(
    serialized=proto_str,
    features=text_item._deserialize_dict()
  )
  assert parsed_example['item-1'].numpy().decode('utf-8') == 'Some texts to print.'


@pytest.mark.parametrize('metric_value', [10, 1, 0, 0.1, 0.01, 0.001])
def test_floatmetrichandler(request, metric_value):
  metric_value = np.float32(metric_value)
  item = FloatMetricHandler('item-1')
  proto_str = get_example_proto(item_handler=item, data=metric_value)
  parsed_example = tf.io.parse_single_example(
    serialized=proto_str,
    features=item._deserialize_dict()
  )
  assert parsed_example['item-1'].numpy() == metric_value
