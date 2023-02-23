import keras.layers
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow_addons.layers import SpectralNormalization
from tensorflow.keras.layers import BatchNormalization
import functools
import GBlock
import DBlock
import LatentStack
import Mylayers


"""Generator implementation."""


class Generator(tf.keras.layers.Layer):
    """Generator for the proposed model."""

    def __init__(self, **kwargs):
        super(Generator, self).__init__(**kwargs)
        """Constructor.

        Args:
          lead_time: last lead time for the generator to predict. Default: 90 min.
          time_delta: time step between predictions. Default: 5 min.
        """
        self.cond_stack = ConditioningStack()
        self.sampler = Sampler()

    @tf.function
    def call(self, inputs, training):
        """Connect to a graph.

        Args:
          inputs: a batch of inputs on the shape [batch_size, time, h, w, 1].
        Returns:
          predictions: a batch of predictions in the form
            [batch_size, num_lead_times, h, w, 1].
            here is [16, 18, 256, 256, 1]
        """
        _, _, height, width, _ = inputs.shape.as_list()
        initial_states = self.cond_stack(inputs)
        predictions = self.sampler(initial_states, [height, width], training=training)
        return predictions

    def get_variables(self):
        """Get all variables of the module."""
        pass

    def get_config(self):
        config = super(Generator, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class ConditioningStack(tf.keras.layers.Layer):
    """Conditioning Stack for the Generator."""
    def __init__(self, **kwargs):
        super(ConditioningStack, self).__init__(**kwargs)
        self.block1 = DBlock.DBlock(output_channels=48, down_sample=True)
        self.conv_mix1 = SpectralNormalization(layers.Conv2D(filters=48, kernel_size=3, padding='same'))
        self.block2 = DBlock.DBlock(output_channels=96, down_sample=True)
        self.conv_mix2 = SpectralNormalization(layers.Conv2D(filters=96, kernel_size=3, padding='same'))
        self.block3 = DBlock.DBlock(output_channels=192, down_sample=True)
        self.conv_mix3 = SpectralNormalization(layers.Conv2D(filters=192, kernel_size=3, padding='same'))
        self.block4 = DBlock.DBlock(output_channels=384, down_sample=True)
        self.conv_mix4 = SpectralNormalization(layers.Conv2D(filters=384, kernel_size=3, padding='same'))

    @tf.function
    def call(self, inputs):
        # here inputs are 4 conditioning radar observations
        # Space to depth conversion of 256x256x1 radar to 128x128x4 hiddens.
        # functools.partial(func, /, *args, **keywords)
        # "functools.partial(func, /, *args, **keywords)" returns a new partial object which when called
        # will behave like "func" called with the positional arguments args and keyword arguments keywords.
        #  If more arguments are supplied to the call, they are appended to args.
        # If additional keyword arguments are supplied, they extend and override keywords.
        h0 = batch_apply(
            functools.partial(tf.nn.space_to_depth, block_size=2, data_format="NHWC"), inputs)

        # Down_sampling residual D Blocks.
        h1 = time_apply(self.block1, h0)
        h2 = time_apply(self.block2, h1)
        h3 = time_apply(self.block3, h2)
        h4 = time_apply(self.block4, h3)

        # Spectrally normalized convolutions, followed by rectified linear units.
        # the 4 output of each residual block are concatenated across the channel dimension
        init_state_1 = self.mixing_layer(h1, self.conv_mix1)
        init_state_2 = self.mixing_layer(h2, self.conv_mix2)
        init_state_3 = self.mixing_layer(h3, self.conv_mix3)
        init_state_4 = self.mixing_layer(h4, self.conv_mix4)

        # Return a stack of conditioning representations of size 64x64x48, 32x32x96,
        # 16x16x192 and 8x8x384.
        return init_state_1, init_state_2, init_state_3, init_state_4

    def mixing_layer(self, inputs, conv_block):
        # Convert from [batch_size, time, h, w, c] -> [batch_size, h, w, c * time]
        # then perform convolution on the output while preserving number of c.
        stacked_inputs = layers.Concatenate(axis=-1)(tf.unstack(inputs, axis=1))
        return layers.ReLU()(conv_block(stacked_inputs))

    def get_config(self):
        config = super(ConditioningStack, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class Sampler(tf.keras.layers.Layer):
    """Sampler for the Generator."""
    def __init__(self, lead_time=90, time_delta=5, **kwargs):
        super(Sampler, self).__init__(**kwargs)
        self.lead_time = lead_time
        self.time_delta = time_delta

        self.num_predictions = lead_time // time_delta
        self.latent_stack = LatentStack.LatentCondStack()

        self.conv_gru4 = Mylayers.ConvGRUCell()
        self.conv4 = SpectralNormalization(layers.Conv2D(filters=768, kernel_size=1, padding='same'))
        self.gblock4 = GBlock.GBlock(output_channels=768)
        self.g_up_block4 = GBlock.UpSampleGBlock(output_channels=384)

        self.conv_gru3 = Mylayers.ConvGRUCell()
        self.conv3 = SpectralNormalization(layers.Conv2D(filters=384, kernel_size=1, padding='same'))
        self.gblock3 = GBlock.GBlock(output_channels=384)
        self.g_up_block3 = GBlock.UpSampleGBlock(output_channels=192)

        self.conv_gru2 = Mylayers.ConvGRUCell()
        self.conv2 = SpectralNormalization(layers.Conv2D(filters=192, kernel_size=1, padding='same'))
        self.gblock2 = GBlock.GBlock(output_channels=192)
        self.g_up_block2 = GBlock.UpSampleGBlock(output_channels=96)

        self.conv_gru1 = Mylayers.ConvGRUCell()
        self.conv1 = SpectralNormalization(layers.Conv2D(filters=96, kernel_size=1, padding='same'))
        self.gblock1 = GBlock.GBlock(output_channels=96)
        self.g_up_block1 = GBlock.UpSampleGBlock(output_channels=48)

        self.bn = BatchNormalization(scale=False)
        self.output_conv = SpectralNormalization(layers.Conv2D(filters=4, kernel_size=1, padding='same'))

    @tf.function
    def call(self, initial_states, resolution, training):
        init_state_1, init_state_2, init_state_3, init_state_4 = initial_states
        batch_size = init_state_1.shape.as_list()[0]
        # Latent conditioning stack.
        z = self.latent_stack(batch_size, resolution)
        z = tf.expand_dims(z, axis=1)
        multiple = tf.constant([1, self.num_predictions, 1, 1, 1])
        z = tf.tile(z, multiples=multiple)
        #  * is an operator defined in Python for its primitive sequence types and an integer to
        #  concatenate the sequence with itself that number of times.
        # hs is a list with _num_predictions elements, each element is a tensor with shape[B, W, H, C]
        # hs = [z] * self._num_predictions
        hs = z
        # Here, RNN returns N-D tensor with shape [batch_size, output_size],
        # where output_size could be a high dimension tensor shape.
        # Layer 4 (bottom-most).

        rnn1 = layers.RNN(cell=self.conv_gru4,
                          return_sequences=True, unroll=True)
        hs = rnn1(inputs=hs, initial_state=init_state_4)
        hs = tf.unstack(hs, axis=1)
        hs = [self.conv4(h) for h in hs]
        hs = [self.gblock4(h, training=training) for h in hs]
        hs = [self.g_up_block4(h, training=training) for h in hs]
        hs = tf.stack(hs, axis=1)

        # Layer 3.
        rnn2 = layers.RNN(cell=self.conv_gru3,
                          return_sequences=True, unroll=True)
        hs = rnn2(inputs=hs, initial_state=init_state_3)
        hs = tf.unstack(hs, axis=1)
        hs = [self.conv3(h) for h in hs]
        hs = [self.gblock3(h, training=training) for h in hs]
        hs = [self.g_up_block3(h, training=training) for h in hs]
        hs = tf.stack(hs, axis=1)

        # Layer 2.
        rnn3 = layers.RNN(cell=self.conv_gru2,
                          return_sequences=True, unroll=True)
        hs = rnn3(inputs=hs, initial_state=init_state_2)
        hs = tf.unstack(hs, axis=1)
        hs = [self.conv2(h) for h in hs]
        hs = [self.gblock2(h, training=training) for h in hs]
        hs = [self.g_up_block2(h, training=training) for h in hs]
        hs = tf.stack(hs, axis=1)

        # Layer 1 (top-most)
        rnn4 = layers.RNN(cell=self.conv_gru1,
                          return_sequences=True, unroll=True)
        hs = rnn4(inputs=hs, initial_state=init_state_1)
        hs = tf.unstack(hs, axis=1)
        hs = [self.conv1(h) for h in hs]
        hs = [self.gblock1(h, training=training) for h in hs]
        hs = [self.g_up_block1(h, training=training) for h in hs]

        # Output layer.
        hs = [layers.ReLU()(self.bn(h, training=training)) for h in hs]
        hs = [self.output_conv(h) for h in hs]
        hs = [tf.nn.depth_to_space(h, 2, data_format='NHWC') for h in hs]
        hs = tf.stack(hs, axis=1)
        return hs

    def get_config(self):
        config = super(Sampler, self).get_config()
        config.update({"lead_time": self.lead_time,
                       "time_delta": self.time_delta,
                       })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.function
def time_apply(func, inputs):
    """Apply function func on each element of inputs along the time axis."""
    return ApplyAlongAxis1(func, axis=1)(inputs)


@tf.function
def batch_apply(func, inputs):
    """Apply function func on each element of inputs along the batch axis."""
    return ApplyAlongAxis1(func, axis=0)(inputs)


class ApplyAlongAxis1(object):
    """Layer for applying an operation on each element, along a specified axis."""

    def __init__(self, operation, axis=0):
        """Constructor."""
        self._operation = operation
        self._axis = axis

    @tf.function
    def __call__(self, *args):
        """Apply the operation to each element of args along the specified axis."""
        # Unpacks the given dimension of a rank-R tensor into rank-(R-1) tensors.
        # Unpacks tensors from value by chipping it along the axis dimension.
        # axis=0 means from 1st dim, along rows.
        # here *args are query, key and value.
        # This returns a list0 with length equals number of *args.
        # Each element in list0 is a list1 of with length equals Batch of each arg for tensor shape[B,H,W,C]
        # Each element in list1 is a tensor with shape[H,W,C]
        # return a list including 1 list which includes 16 tensor elements with shape(B, H, W, C]
        split_inputs = [tf.unstack(arg, axis=self._axis) for arg in args]

        # If a single iterable is passed, zip() returns an iterator of tuples with each tuple having only one element.
        # The * operator can be used in conjunction with zip() to unzip the list.
        # The zip() function returns an iterator of tuples based on the iterable objects.
        res = [self._operation(x[0]) for x in zip(*split_inputs)]
        # Stacks a list of rank-R tensors into one rank-(R+1) tensor.
        return tf.stack(res, axis=self._axis)
