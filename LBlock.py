import tensorflow as tf
from tensorflow import keras
from tensorflow_addons.layers import SpectralNormalization
from tensorflow.keras import layers
import Mylayers


# LBlock: Used in LatentStack.py
class LBlock(keras.layers.Layer):
    """
    Residual block for the Latent Stack.
    L-Block for increasing the number of channels in the input
    """
    def __init__(self, output_channels, **kwargs):
        super(LBlock, self).__init__(**kwargs)
        self.output_channels = output_channels
        self.activation = layers.ReLU()
        self.conv1 = SpectralNormalization(Mylayers.CustomConv2d_dif_o_i(output_channels=output_channels, kernel_size=1))
        self.conv2 = SpectralNormalization(layers.Conv2D(filters=output_channels, kernel_size=3, strides=(1, 1), padding="same"))
        self.conv3 = SpectralNormalization(layers.Conv2D(filters=output_channels, kernel_size=3, strides=(1, 1), padding="same"))

    """Constructor for the L blocks of the DVD-GAN.

    Args:
      output_channels: Integer number of channels in convolution operations in
        the main branch, and number of channels in the output of the block.
      kernel_size: Integer kernel size of the convolutions. Default: 3.
      conv: TF module. Default: layers.Conv2D.
      activation: Activation before the conv. layers. Default: tf.nn.relu.
    """
    @tf.function
    def call(self, inputs):
        """Build the LBlock.

        Args:
          inputs: a tensor with a complete observation [N, H, W, input_channels]

        Returns:
          A tensor with shape [b, H, W, output_channels=2*input_channels]
          A tensor with discriminator loss scalars [B].

        # Stack of two conv. layers and non_linearity that increase the number of channels.
        """
        x0 = inputs
        input_channels = inputs.shape.as_list()[-1]
        if input_channels < self.output_channels:
            x1 = self.conv1(inputs)
            x1 = layers.Concatenate(axis=-1)([inputs, x1])
        else:
            x1 = inputs

        x2 = self.activation(x0)
        x2 = self.conv2(x2)
        x2 = self.activation(x2)
        x2 = self.conv3(x2)
        # Residual connection.
        return x1 + x2

    def get_config(self):
        config = super(LBlock, self).get_config()
        config.update({"output_channels": self.output_channels})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)