import tensorflow as tf
import pytest

from nn_trainer.data.data_augmentation import TFRandomOperator, RandomRotation90


TF_DTYPE = [
  tf.uint8,
  tf.uint16,
  tf.float16,
  tf.float32,
  tf.float64,
]


@pytest.mark.parametrize('tf_dtype', TF_DTYPE)
def test_random_rotation90(tf_dtype):
  source = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=tf_dtype)
  print(source)

  op = RandomRotation90(random_apply_ratio=1.)


