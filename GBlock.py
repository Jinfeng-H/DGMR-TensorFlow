import tensorflow as tf
from tensorflow import keras
from tensorflow_addons.layers import SpectralNormalization
from tensorflow.keras import layers


# The second residual block doubles the input's spatial resolution with
# nearest neighbour interpolation and halves its channels
# GBlock and UpSampleGBlock used in Generator
class UpSampleGBlock(keras.layers.Layer):
    def __init__(self, output_channels, **kwargs):
        super(UpSampleGBlock, self).__init__(**kwargs)
        self.output_channels = output_channels
        # for channels first, set axis=1; here is channels last, axis=-1,which is default.
        self.bn1 = layers.BatchNormalization(scale=False, name='GBlock_up_BN1')
        self.bn2 = layers.BatchNormalization(scale=False, name='GBlock_up_BN2')
        self.activation = layers.ReLU()
        # interpolation = "nearest" in the paper
        self.up_sample = layers.UpSampling2D(size=(2, 2), interpolation="nearest")
        self.conv1 = SpectralNormalization(layers.Conv2D(filters=output_channels, kernel_size=1, strides=(1, 1), padding="same"))
        self.conv2 = SpectralNormalization(layers.Conv2D(filters=output_channels, kernel_size=3, strides=(1, 1), padding="same"))
        self.conv3 = SpectralNormalization(layers.Conv2D(filters=output_channels, kernel_size=3, strides=(1, 1), padding="same"))

    @tf.function
    def call(self, inputs, training):
        # data_format: A string, channels_last (default) corresponds to inputs with shape
        # (batch_size, height, width, channels)
        input_channels = inputs.shape.as_list()[-1]

        x1 = self.up_sample(inputs)
        x1 = self.conv1(x1)

        x2 = self.bn1(inputs, training=training)
        x2 = self.activation(x2)
        x2 = self.up_sample(x2)
        x2 = self.conv2(x2)
        x2 = self.bn2(x2, training=training)
        x2 = self.activation(x2)
        x2 = self.conv3(x2)
        return x1 + x2

    def get_config(self):
        config = super(UpSampleGBlock, self).get_config()
        config.update({"output_channels": self.output_channels})
        return config


class GBlock(keras.layers.Layer):
    def __init__(self, output_channels,
                 **kwargs):
        super(GBlock, self).__init__(**kwargs)
        # for channels first, set axis=1; here is channels last, axis=-1,which is default.
        self.output_channels = output_channels
        self.bn1 = layers.BatchNormalization(scale=False, name='GBlock_BN1')
        self.bn2 = layers.BatchNormalization(scale=False, name='GBlock_BN2')
        self.activation = layers.ReLU()
        # SpectralNormalization is a method to stabilize the training of GANs.
        self.conv1 = SpectralNormalization(layers.Conv2D(filters=output_channels, kernel_size=1, strides=(1, 1), padding="same"))
        self.conv2 = SpectralNormalization(layers.Conv2D(filters=output_channels, kernel_size=3, strides=(1, 1), padding="same"))
        self.conv3 = SpectralNormalization(layers.Conv2D(filters=output_channels, kernel_size=3, strides=(1, 1), padding="same"))

    @tf.function
    def call(self, inputs, training):
        # data_format: A string, channels_last (default) corresponds to inputs with shape
        # (batch_size, height, width, channels)
        input_channels = inputs.shape.as_list()[-1]
        x1 = self.conv1(inputs)

        x2 = self.bn1(inputs, training=training)
        x2 = self.activation(x2)
        x2 = self.conv2(x2)
        x2 = self.bn2(x2, training=training)
        x2 = self.activation(x2)
        x2 = self.conv3(x2)
        return x1 + x2

    def get_config(self):
        config = super(GBlock, self).get_config()
        config.update({"output_channels": self.output_channels})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)