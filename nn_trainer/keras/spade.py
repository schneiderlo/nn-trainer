import tensorflow as tf
from tensorflow.python import keras


class Spade(keras.Model):
  """SPatially-Adaptative (DE)normalization (SPADE)

  See https://arxiv.org/abs/1903.07291.
  """
  def __init__(self, kernel_depth: int, semantic_map: tf.Tensor):
    super(Spade, self).__init__(name='spade')
    self._semantic_map = semantic_map
    self._semantic_conv_128 = keras.layers.Conv2D(
      filters=128,
      kernel_size=3,
      padding='same',
      name='semantic_conv_128'
    )
    self._conv_k_pre_mul = keras.layers.Conv2D(
      filters=kernel_depth,
      kernel_size=3,
      padding='same',
      name='conv_k_pre_mul'
    )
    self._conv_k_pre_add = keras.layers.Conv2D(
      filters=kernel_depth,
      kernel_size=3,
      padding='same',
      name='conv_k_pre_add'
    )
    self._sync_batch_norm = keras.layers.BatchNormalization()

  def call(self, inputs: tf.Tensor, training=None, mask=None):
    input_shape = tf.shape(inputs)
    input_h = input_shape[1]
    input_w = input_shape[2]
    x_semantic = tf.image.resize(
      self._semantic_map,
      size=[input_h, input_w],
      method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )
    x_semantic = self._semantic_conv_128(x_semantic)
    x_semantic_pre_mul = self._conv_k_pre_mul(x_semantic)
    x_semantic_pre_add = self._conv_k_pre_add(x_semantic)

    x = self._sync_batch_norm(inputs, training=training)
    x = x_semantic_pre_mul * x
    x = x_semantic_pre_add + x
    return x


class SpadeResBlock(keras.Model):
  """SPADE residual block.

  See https://arxiv.org/abs/1903.07291.
  """
  def __init__(self, kernel_depth: int, semantic_map: tf.Tensor):
    super(SpadeResBlock, self).__init__(name='spade_resblock')
    self._semantic_map = semantic_map
    self._spade_1 = Spade(kernel_depth=kernel_depth, semantic_map=semantic_map)
    self._conv_1 = keras.layers.Conv2D(
      filters=kernel_depth,
      kernel_size=3,
      padding='same',
      name='conv_1'
    )
    self._spade_2 = Spade(kernel_depth=kernel_depth, semantic_map=semantic_map)
    self._conv_2 = keras.layers.Conv2D(
      filters=kernel_depth,
      kernel_size=3,
      padding='same',
      name='conv_2'
    )
    self._spade_3 = Spade(kernel_depth=kernel_depth, semantic_map=semantic_map)
    self._conv_3 = keras.layers.Conv2D(
      filters=kernel_depth,
      kernel_size=3,
      padding='same',
      name='conv_3'
    )

  def call(self, inputs: tf.Tensor, training=None, mask=None):
    x_left = self._spade_1(inputs, training=training, mask=mask)
    x_left = tf.nn.relu(x_left)
    x_left = self._conv_1(x_left)
    x_left = self._spade_2(x_left, training=training, mask=mask)
    x_left = tf.nn.relu(x_left)
    x_left = self._conv_2(x_left)

    x_right = self._spade_3(inputs, training=training, mask=mask)
    x_right = tf.nn.relu(x_right)
    x_right = self._conv_3(x_right)

    x = x_left + x_right

    return x


class SpadeGenerator(keras.Model):
  """SPADE Generator.

  See https://arxiv.org/abs/1903.07291.
  """
  def __init__(self, semantic_map: tf.Tensor, num_out_channels: int):
    super(SpadeGenerator, self).__init__(name='spade_generator')
    self._dense = keras.layers.Dense(1024 * 4 * 4)
    self._spade_resblock_1 = SpadeResBlock(kernel_depth=1024, semantic_map=semantic_map)
    self._spade_resblock_2 = SpadeResBlock(kernel_depth=1024, semantic_map=semantic_map)
    self._spade_resblock_3 = SpadeResBlock(kernel_depth=1024, semantic_map=semantic_map)
    self._spade_resblock_4 = SpadeResBlock(kernel_depth=512, semantic_map=semantic_map)
    self._spade_resblock_5 = SpadeResBlock(kernel_depth=256, semantic_map=semantic_map)
    self._spade_resblock_6 = SpadeResBlock(kernel_depth=128, semantic_map=semantic_map)
    self._spade_resblock_7 = SpadeResBlock(kernel_depth=64, semantic_map=semantic_map)
    self._last_conv = keras.layers.Conv2D(num_out_channels, kernel_size=3, padding='same')

  def _up_sample_by_2(self, inputs: tf.Tensor):
    input_shape = tf.shape(inputs)
    input_h = input_shape[1]
    input_w = input_shape[2]
    x = tf.image.resize(
      inputs,
      size=[input_h * 2, input_w * 2],
      method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )
    return x

  def call(self, inputs: tf.Tensor, training=None, mask=None):
    x = self._dense(inputs)
    x = tf.reshape(x, shape=[-1, 4, 4, 1024])
    x = self._spade_resblock_1(x)
    x = self._up_sample_by_2(x)
    x = self._spade_resblock_2(x)
    x = self._up_sample_by_2(x)
    x = self._spade_resblock_3(x)
    x = self._up_sample_by_2(x)
    x = self._spade_resblock_4(x)
    x = self._up_sample_by_2(x)
    x = self._spade_resblock_5(x)
    x = self._up_sample_by_2(x)
    x = self._spade_resblock_6(x)
    x = self._up_sample_by_2(x)
    x = self._spade_resblock_7(x)
    x = self._up_sample_by_2(x)
    x = self._last_conv(x)
    return x


# block = Spade(3, semantic_map=tf.zeros([1, 2, 3, 3]))
# print(block(tf.zeros([1, 2, 3, 3])))
# print([x.name for x in block.trainable_variables])
#
# block = SpadeResBlock(3, semantic_map=tf.zeros([1, 4, 6, 3]))
# print(block(tf.zeros([1, 2, 3, 3])))
# print([x.name for x in block.trainable_variables])


block = SpadeGenerator(num_out_channels=3, semantic_map=tf.zeros([2, 512, 512, 1]))
print(block(tf.zeros([2, 256])))
print([x.name for x in block.trainable_variables])

