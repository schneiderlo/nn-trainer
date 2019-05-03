import pytest
import numpy as np
import tensorflow as tf

from nn_trainer.utils.tests import get_test_data
from nn_trainer.data.data_model import DataModel
from nn_trainer.data.item_handler import NPArrayHandler
from nn_trainer.data.input_fn import InputPipeline


BATCH_SIZES = [1, 10, 100]


def get_test_dataset(
    num_epochs: int,
    batch_size: int,
    is_training: bool) -> tf.data.Dataset:
  """Helper function to get a `tf.data.Dataset` filled with Lenna."""
  tfrecord_path = get_test_data('tfrecords', 'lenna_float32.tfr')
  data_model = DataModel(
    item_handlers=[
      NPArrayHandler('source', np.float32)
    ]
  )
  ipl = InputPipeline(
    data_model=data_model,
    num_epochs=num_epochs * batch_size,
    num_samples=1,
    batch_size=batch_size
  )
  return ipl.create_dataset([tfrecord_path], is_training=is_training)


@pytest.mark.parametrize('batch_size', BATCH_SIZES)
def test_create_train_dataset(request, batch_size):
  dataset = get_test_dataset(
    num_epochs=2,
    batch_size=int(batch_size),
    is_training=True)
  for i, item in enumerate(dataset):
    lenna = item['source'].numpy()
    assert lenna.shape == (batch_size, 729, 740, 1)


@pytest.mark.parametrize('batch_size', BATCH_SIZES)
def test_create_eval_dataset(request, batch_size):
  dataset = get_test_dataset(
    num_epochs=1,
    batch_size=int(batch_size),
    is_training=False)
  for i, item in enumerate(dataset):
    lenna = item['source'].numpy()
    assert lenna.shape == (batch_size, 729, 740, 1)
