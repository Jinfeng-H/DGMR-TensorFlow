import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, activations
from tensorflow_addons.layers import SpectralNormalization


# Used in Attention.py
class ApplyAlongAxis:
    """Layer for applying an operation on each element, along a specified axis."""

    def __init__(self, operation, axis=0):
        """Constructor."""
        self._operation = operation
        self._axis = axis

    def __call__(self, *args):
        """Apply the operation to each element of args along the specified axis."""
        # Unpacks the given dimension of a rank-R tensor into rank-(R-1) tensors.
        # Unpacks tensors from value by chipping it along the axis dimension.
        # axis=0 means from 1st dim, along rows.
        # here *args are query, key and value.
        # This returns a list0 with length equals number of *args.
        # Each element in list0 is a list1 of with length equals Batch of each arg for tensor shape[B,H,W,C]
        # Each element in list1 is a tensor with shape[H,W,C]
        # here num is batch_size1=2
        # https://stackoverflow.com/questions/71950234/tensorflow-io-valueerror-cannot-infer-argument-num-from-shape-none-none-n
        # https://stackoverflow.com/questions/45404056/tf-unstack-with-dynamic-shape
        split_inputs = [tf.unstack(arg, num=1, axis=self._axis) for arg in args]

        # If a single iterable is passed, zip() returns an iterator of tuples with each tuple having only one element.
        # The * operator can be used in conjunction with zip() to unzip the list.
        # The zip() function returns an iterator of tuples based on the iterable objects.
        res = [self._operation(x1, x2, x3) for (x1, x2, x3) in zip(*split_inputs)]
        # Stacks a list of rank-R tensors into one rank-(R+1) tensor.
        return tf.stack(res, axis=self._axis)


# ConvGRUCell:Used in Generator
class ConvGRUCell(tf.keras.layers.Layer):
    """A ConvGRU implementation."""
    """inputs, prev_state should be tensors with shape [B, H, W, C]
    the output shape is a tensor with the same shape the prev_state.shape
    returns a tuple with two elements(out, new_state)"""

    def __init__(self, kernel_size=3,
                 activation=activations.relu,  # the last activation
                 recurrent_activation=activations.sigmoid,
                 use_bias=False,
                 kernel_initializer='Ones',
                 recurrent_initializer='Ones',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 **kwargs):
        """Constructor.
        Args:
          kernel_size: kernel size of the convolutions. Default: 3.
          A state_size attribute. This can be a single integer (single state) in which case it is the size of the
          _state_size: recurrent state. This can also be a list/tuple of integers (one size per state). The state_size
          can also be TensorShape or tuple/list of TensorShape, to represent high dimension state.
          _state_size will be changed during the call.
        """
        super(ConvGRUCell, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.activation = activation
        self.recurrent_activation = recurrent_activation
        self.use_bias = use_bias
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.recurrent_initializer = tf.keras.initializers.get(recurrent_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.kernel_regularizer = kernel_regularizer
        self.recurrent_regularizer = recurrent_regularizer
        self.bias_regularizer = bias_regularizer
        self.kernel_constraint = kernel_constraint
        self.recurrent_constraint = recurrent_constraint
        self.bias_constraint = bias_constraint
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.state_size = None
        # self.kernel = None
        # Read gate of the GRU.
        self.spectral_norm1 = SpectralNormalization(CustomConv2d_last_in(kernel_size=self.kernel_size))
        self.spectral_norm2 = SpectralNormalization(CustomConv2d_last_in(kernel_size=self.kernel_size))
        self.spectral_norm3 = SpectralNormalization(CustomConv2d_last_in(kernel_size=self.kernel_size))

    @tf.function
    def call(self, inputs, states):
        # here states is a tuple with one element(tensor), this tensor is the initial state.
        self.state_size = tf.TensorShape(states[0].shape.as_list()[1:])
        h_tm1 = states[0]
        xh = tf.concat([inputs, h_tm1], axis=-1)

        read_gate_conv = self.spectral_norm1
        read_gate = self.recurrent_activation(read_gate_conv(xh))

        # Update gate of the GRU.
        update_gate_conv = self.spectral_norm2
        update_gate = self.recurrent_activation(update_gate_conv(xh))

        # Gate the inputs.
        # a * b is element-wise multiplication of 2 tensors.
        gated_input = tf.concat([inputs, read_gate * h_tm1], axis=-1)

        # Gate the cell and state / outputs.
        output_conv = self.spectral_norm3
        ht = self.activation(output_conv(gated_input))
        # previous and candidate state mixed by update gate
        out = update_gate * h_tm1 + (1. - update_gate) * ht
        new_state = out
        return out, new_state

    def get_config(self):
        config = super(ConvGRUCell, self).get_config()
        config.update({"kernel_size": self.kernel_size,
                       "use_bias": self.use_bias,
                       "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
                       "recurrent_initializer": keras.initializers.serialize(self.recurrent_initializer),
                       "bias_initializer": keras.initializers.serialize(self.bias_initializer),
                       "kernel_constraint": self.kernel_constraint,
                       "recurrent_constraint": self.recurrent_constraint,
                       "bias_constraint": self.bias_constraint,
                       "dropout": self.dropout,
                       "recurrent_dropout": self.recurrent_dropout,
                       # "state_size": self.state_size,
                       # "kernel": self.kernel,
                       })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# This customized layer is used in ConvGRUCell
# customize convolution layers with multiple inputs
# use last input's input_channels from several inputs as filters size.
class CustomConv2d_last_in(tf.keras.layers.Layer):
    def __init__(self, kernel_size, padding="SAME", strides=(1, 1), activation=None, use_bias=True,
                 kernel_initializer="glorot_uniform",
                 bias_initializer="zeros", **kwargs):
        super(CustomConv2d_last_in, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.strides = strides
        self.activation = activation
        self.use_bias = use_bias
        self.padding = padding
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)

    def build(self, input_shape):
        input_channels = input_shape[-1]//3
        self.kernel = self.add_weight(name='GRU_kernel',
                                      shape=(self.kernel_size, self.kernel_size, input_channels, input_channels),
                                      initializer=self.kernel_initializer,
                                      regularizer=None,
                                      constraint=None,
                                      dtype='float32',
                                      trainable=True)

        if self.use_bias:
            self.bias = self.add_weight(name='GRU_bias',
                                        shape=(input_channels,),
                                        initializer=self.bias_initializer,
                                        regularizer=None,
                                        constraint=None,
                                        dtype='float32',
                                        trainable=True)
        else:
            self.bias = None

    @tf.function
    def call(self, inputs):
        x = tf.nn.conv2d(inputs, filters=self.kernel, strides=self.strides, padding=self.padding)
        if self.use_bias:
            x = x + self.bias
        return x

    def get_config(self):
        config = super(CustomConv2d_last_in, self).get_config()
        config.update({"kernel_size": self.kernel_size,
                       "strides": self.strides,
                       "activation": self.activation,
                       "padding": self.padding,
                       "use_bias": self.use_bias,
                       "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
                       "bias_initializer": keras.initializers.serialize(self.bias_initializer),
                       })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# This customized layer is used in DBlock
# customize convolution layers with single input
# use input_channels as filters size.
class CustomConv2d_in(tf.keras.layers.Layer):
    def __init__(self, kernel_size, padding="SAME", strides=(1, 1), use_bias=True,
                 kernel_initializer="glorot_uniform",
                 bias_initializer="zeros",
                 **kwargs):
        super(CustomConv2d_in, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.padding = padding
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.strides = strides
        self.use_bias = use_bias

    def build(self, input_shape):
        input_channels = input_shape[-1]
        self.kernel = self.add_weight(name='kernel',
                                      shape=(self.kernel_size, self.kernel_size, input_channels, input_channels),
                                      initializer=self.kernel_initializer,
                                      regularizer=None,
                                      constraint=None,
                                      dtype='float32',
                                      trainable=True)

        if self.use_bias:
            self.bias = self.add_weight(name='bias',
                                        shape=(input_channels,),
                                        initializer=self.bias_initializer,
                                        regularizer=None,
                                        constraint=None,
                                        dtype='float32',
                                        trainable=True)
        else:
            self.bias = None

    @tf.function
    def call(self, inputs):
        x = tf.nn.conv2d(inputs, filters=self.kernel, strides=self.strides, padding=self.padding)
        if self.use_bias:
            x = x + self.bias
        return x

    def get_config(self):
        config = super(CustomConv2d_in, self).get_config()
        config.update({"kernel_size": self.kernel_size,
                       "padding": self.padding,
                       "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
                       "bias_initializer": keras.initializers.serialize(self.bias_initializer),
                       "strides": self.strides,
                       "use_bias": self.use_bias
                       })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# This customized layer is used in LBlock
# customize convolution layers with single input
# use difference between output_channels and input_channels as filters size.
class CustomConv2d_dif_o_i(tf.keras.layers.Layer):
    def __init__(self, output_channels, kernel_size, padding="SAME", strides=(1, 1),
                 activation=None, use_bias=True,
                 kernel_initializer="glorot_uniform",
                 bias_initializer="zeros", **kwargs):
        super(CustomConv2d_dif_o_i, self).__init__(**kwargs)
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.strides = strides
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)

    def build(self, input_shape):
        input_channels = input_shape[-1]
        filter_num = self.output_channels - input_channels
        self.kernel = self.add_weight(name='kernel',
                                      shape=(self.kernel_size, self.kernel_size, input_channels, filter_num),
                                      initializer=self.kernel_initializer,
                                      regularizer=None,
                                      constraint=None,
                                      dtype='float32',
                                      trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(name='bias',
                                        shape=(filter_num,),
                                        initializer=self.bias_initializer,
                                        regularizer=None,
                                        constraint=None,
                                        dtype='float32',
                                        trainable=True)

    @tf.function
    def call(self, inputs):
        x = tf.nn.conv2d(input=inputs, filters=self.kernel,
                         strides=self.strides, padding=self.padding)
        if self.use_bias:
            x = x + self.bias
        return x

    def get_config(self):
        config = super(CustomConv2d_dif_o_i, self).get_config()
        config.update({"output_channels": self.output_channels,
                       "kernel_size": self.kernel_size,
                       "padding": self.padding, "strides": self.strides,
                       "activation": self.activation,
                       "use_bias": self.use_bias,
                       "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
                       "bias_initializer": keras.initializers.serialize(self.bias_initializer),
                       })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

