import tensorflow as tf
from tensorflow.python import keras


class Spade(keras.Model):
  """SPatially-Adaptative (DE)normalization (SPADE)

  See https://arxiv.org/abs/1903.07291.
  """
  def __init__(self,
               kernel_depth: int,
               semantic_map: tf.Tensor):
    """Initializer.

    Args:
      kernel_depth: The number of channels of input tensor.
                    It is the output dimension of the SPADE.
      semantic_map: Semantic map.
    """
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
      name='gamma'
    )
    self._conv_k_pre_add = keras.layers.Conv2D(
      filters=kernel_depth,
      kernel_size=3,
      padding='same',
      name='beta'
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
    x_semantic = tf.nn.relu(x_semantic)
    gamma = self._conv_k_pre_mul(x_semantic)
    beta = self._conv_k_pre_add(x_semantic)

    x = self._sync_batch_norm(inputs, training=training)
    x = x * (1 + gamma) + beta
    return x


class SpadeResBlock(keras.Model):
  """SPADE residual block.

  See https://arxiv.org/abs/1903.07291.
  """
  def __init__(self, depth_in: int, depth_out: int, semantic_map: tf.Tensor):
    super(SpadeResBlock, self).__init__(name='spade_resblock')
    depth_mid = min(depth_in, depth_out)
    self._learn_shortcut = (depth_in != depth_out)
    self._spade_1 = Spade(kernel_depth=depth_in, semantic_map=semantic_map)
    self._conv_1 = keras.layers.Conv2D(
      filters=depth_mid,
      kernel_size=3,
      padding='same',
      name='conv_1'
    )
    self._spade_2 = Spade(kernel_depth=depth_mid, semantic_map=semantic_map)
    self._conv_2 = keras.layers.Conv2D(
      filters=depth_out,
      kernel_size=3,
      padding='same',
      name='conv_2'
    )
    if self._learn_shortcut:
      self._spade_shortcut = Spade(kernel_depth=depth_in, semantic_map=semantic_map)
      self._conv_shortcut = keras.layers.Conv2D(
        filters=depth_out,
        kernel_size=1,
        padding='same',
        name='conv_3',
        use_bias=False
      )

  @staticmethod
  def activation_fn(inputs: tf.Tensor):
    return tf.nn.leaky_relu(inputs, alpha=0.2)

  def call(self, inputs: tf.Tensor, training=None, mask=None):
    x_left = self._spade_1(inputs, training=training, mask=mask)
    x_left = self.activation_fn(x_left)
    x_left = self._conv_1(x_left)
    x_left = self._spade_2(x_left, training=training, mask=mask)
    x_left = self.activation_fn(x_left)
    x_left = self._conv_2(x_left)

    if self._learn_shortcut:
      x_shortcut = self._spade_shortcut(inputs, training=training, mask=mask)
      x_shortcut = self._conv_shortcut(x_shortcut)
    else:
      x_shortcut = inputs

    x = x_left + x_shortcut

    return x


class SpadeGenerator(keras.Model):
  """SPADE Generator.

  See https://arxiv.org/abs/1903.07291.
  """
  def __init__(self,
               semantic_map: tf.Tensor,
               z_dim: int,
               num_gen_filters: int = 64,
               num_out_channels: int = 1):
    super(SpadeGenerator, self).__init__(name='spade_generator')
    ngf = num_gen_filters
    self._dense = keras.layers.Dense(16 * ngf * 4 * 4, input_dim=z_dim)
    self._spade_resblock_1 = SpadeResBlock(
      depth_in=16 * ngf, depth_out=16 * ngf, semantic_map=semantic_map)
    self._spade_resblock_2 = SpadeResBlock(
      depth_in=16 * ngf, depth_out=16 * ngf, semantic_map=semantic_map)
    self._spade_resblock_3 = SpadeResBlock(
      depth_in=16 * ngf, depth_out=16 * ngf, semantic_map=semantic_map)
    self._spade_resblock_4 = SpadeResBlock(
      depth_in=16 * ngf, depth_out=8 * ngf, semantic_map=semantic_map)
    self._spade_resblock_5 = SpadeResBlock(
      depth_in=8 * ngf, depth_out=4 * ngf, semantic_map=semantic_map)
    self._spade_resblock_6 = SpadeResBlock(
      depth_in=4 * ngf, depth_out=2 * ngf, semantic_map=semantic_map)
    self._spade_resblock_7 = SpadeResBlock(
      depth_in=2 * ngf, depth_out=1 * ngf, semantic_map=semantic_map)
    self._last_conv = keras.layers.Conv2D(num_out_channels, kernel_size=3, padding='same')

  @staticmethod
  def _upsample_by_2(inputs: tf.Tensor):
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
    x = self._upsample_by_2(x)
    x = self._spade_resblock_2(x)
    x = self._upsample_by_2(x)
    x = self._spade_resblock_3(x)
    x = self._upsample_by_2(x)
    x = self._spade_resblock_4(x)
    x = self._upsample_by_2(x)
    x = self._spade_resblock_5(x)
    x = self._upsample_by_2(x)
    x = self._spade_resblock_6(x)
    x = self._upsample_by_2(x)
    x = self._spade_resblock_7(x)
    x = self._upsample_by_2(x)
    x = self._last_conv(x)
    x = tf.nn.tanh(x)
    return x


z_dim = 256
block = SpadeGenerator(z_dim=z_dim, num_out_channels=1, semantic_map=tf.ones([2, 1024, 1024, 1]))

block.build(input_shape=(2, z_dim))
print(block.summary())

