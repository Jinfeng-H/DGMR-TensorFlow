import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, activations
from tensorflow_addons.layers import SpectralNormalization
import Mylayers


# data_format: A string, channels_last (default) corresponds to inputs with shape (batch_size, height, width, channels)
# DBlock used in Discriminator.py, Generator.py
class DBlock(tf.keras.layers.Layer):
    def __init__(self, output_channels,
                 pooling_2d=True,
                 conv_2d=True,
                 pre_activation=True,
                 down_sample=True,
                 **kwargs):
        super(DBlock, self).__init__(**kwargs)
        self.output_channels = output_channels
        self.pre_activation = pre_activation
        self.down_sample = down_sample
        self.conv_2d = conv_2d
        self.conv2 = SpectralNormalization(Mylayers.CustomConv2d_in(kernel_size=2))
        # in convolution 2d and 3d layers, default stride is 1.
        if conv_2d:
            self.conv1 = SpectralNormalization(layers.Conv2D(filters=output_channels, kernel_size=1, padding="same"))
            self.conv3 = SpectralNormalization(layers.Conv2D(filters=output_channels, kernel_size=3, padding="same"))

        else:
            self.conv1 = SpectralNormalization(layers.Conv3D(filters=output_channels, kernel_size=1, padding="same"))
            self.conv3 = SpectralNormalization(layers.Conv3D(filters=output_channels, kernel_size=3, padding="same"))
        # in convolution 2d and 3d average pooling layers, default size is 2.
        self.pooling_2d = pooling_2d

    """
    Constructor for the D blocks of the DVD-GAN.
      Args:
        output_channels: Integer number of channels in the second convolution, and
          number of channels in the residual 1x1 convolution module.
        down_sample: Boolean: shall we use the average pooling layer?
        pre_activation: Boolean: shall we apply pre-activation to inputs?
        conv: TF module, either layers.Conv2D or a wrapper with spectral normalisation.
        pooling: Average pooling layer. Default: layers.down_sample_avg_pool.
        activation: Activation at optional pre_activation and first conv layers.
      """

    @tf.function
    def call(self, inputs):
        """Build the DBlock.
        Args:
          inputs: a tensor with a complete observation [b, H, W, input_channels]

        Returns:
          A tensor with shape [b, H/2, W/2, output_channels]
          A tensor with discriminator loss scalars [b].
        """
        x0 = inputs
        # 1×1 convolution.
        x1 = self.conv1(x0)

        # 3×3 convolution.
        # Pre-activation.
        if self.pre_activation:
            x0 = layers.ReLU()(x0)
        # First 3×3 convolution.
        x2 = self.conv2(x0)
        #x2 = self.conv2(x0)
        x2 = layers.ReLU()(x2)
        # Second 3×3 convolution.
        x2 = self.conv3(x2)
        # Down_sampling.
        if self.down_sample:
            if self.pooling_2d:
                x1 = layers.AveragePooling2D(pool_size=(2, 2))(x1)
                x2 = layers.AveragePooling2D(pool_size=(2, 2))(x2)
            else:
                x1 = layers.AveragePooling3D(pool_size=(2, 2, 2))(x1)
                x2 = layers.AveragePooling3D(pool_size=(2, 2, 2))(x2)
        return x1 + x2

    def get_config(self):
        config = super(DBlock, self).get_config()
        config.update({"output_channels": self.output_channels,
                       "pre_activation": self.pre_activation,
                       "down_sample": self.down_sample,
                       "conv_2d": self.conv_2d,
                       "pooling_2d": self.pooling_2d,
                       })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

