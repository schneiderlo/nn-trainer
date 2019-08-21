import tensorflow as tf
from abc import abstractmethod


class TFRandomOperator(object):
  """Base class from which every random operator are derived."""

  def __init__(self, random_apply_ratio: float, seed=None):
    """Initializer.

    Args:
      random_apply_ratio: The ratio with which the operator is triggered. This
                          value must lie in the interval [0, 1].
      seed: Used to create a random seed. See `tf.set_random_seed` for behavior.
            Only used if random_apply_ratio != 1.
    """
    self._random_apply_ratio = random_apply_ratio
    self._seed = seed

  def apply(self, src: tf.Tensor) -> tf.Tensor:
    """Apply the operator on a `tf.Tensor`.

    Args:
      src: source image (valid pixel array).
           Its shape must conform to the format [C, H, W].
    Return: transformed image, as a valid pixel array.
    """
    if self._random_apply_ratio == 1.:
      dst = self.apply_impl(src=src)
    else:
      uniform_random = tf.random.uniform([], 0, 1.0, seed=self._seed)
      random_apply_ratio = tf.convert_to_tensor(
          self._random_apply_ratio, dtype=tf.float32)
      apply_cond = tf.less(uniform_random, random_apply_ratio)
      dst = tf.cond(
        apply_cond,
        lambda: self.apply_impl(src=src),
        lambda: src)
    return dst

  @abstractmethod
  def apply_impl(self, src: tf.Tensor) -> tf.Tensor:
    """Actual implementation of the operator."""
    pass


class ConversionOp(TFRandomOperator):

  def __init__(self, output_tf_type) -> None:
    super(ConversionOp, self).__init__(random_apply_ratio=1., seed=None)
    self._output_tf_type = output_tf_type

  def apply_impl(self, src: tf.Tensor) -> tf.Tensor:
    return tf.cast(src, dtype=self._output_tf_type)


class ChangeRangeOp(TFRandomOperator):

  def __init__(self,
               initial_range_start: float,
               initial_range_end: float,
               final_range_start: float,
               final_range_end: float) -> None:
    super(ChangeRangeOp, self).__init__(random_apply_ratio=1., seed=None)
    self._initial_range_start = initial_range_start
    self._initial_range_end = initial_range_end
    self._final_range_start = final_range_start
    self._final_range_end = final_range_end

  def apply_impl(self, src: tf.Tensor) -> tf.Tensor:
    return src


class RandomRotation90(TFRandomOperator):
  """Operator that rotate an image randomly by 0°, 90°, 180° and 270°."""

  def __init__(self, random_apply_ratio: float,  seed=None) -> None:
    """Initializer.

    Args:
      k: A scalar integer. The number of times the
                   image is rotated by 90 degrees.
    """
    super(RandomRotation90, self).__init__(
      random_apply_ratio=random_apply_ratio, seed=seed)

  def apply_impl(self, image: tf.Tensor) -> tf.Tensor:
    k = tf.random.uniform(shape=[], minval=1, maxval=5, dtype=tf.int32)

    def _rot90():
      return tf.transpose(tf.reverse(image, [1]), [1, 0, 2])

    def _rot180():
      return tf.reverse(image, [0, 1])

    def _rot270():
      return tf.reverse(tf.transpose(image, [1, 0, 2]), [1])

    cases = [
      (tf.equal(k, 1), _rot90),
      (tf.equal(k, 2), _rot180),
      (tf.equal(k, 3), _rot270)
    ]

    return tf.case(cases, default=lambda: image, exclusive=True)

