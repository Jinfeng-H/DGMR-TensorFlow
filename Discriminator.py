import tensorflow as tf
from tensorflow.keras import layers, activations
import DBlock
# from tensorflow_addons.layers import SpectralNormalization
from tensorflow.keras.layers import BatchNormalization

"""Discriminator implementation."""
'''
By subclassing tf.Module instead of object any tf.Variable or tf.Module instances assigned to object properties can 
be collected using the variables, trainable_variables or submodules property. refer to tf.Module
'''


class Discriminator(tf.keras.layers.Layer):
    """Discriminator."""
    def __init__(self, num_spatial_frames=8, temporal_crop_ratio=2,
                 num_conditioning_frames=4, **kwargs):
        super(Discriminator, self).__init__()
        """Constructor."""
        # Number of random time steps for the spatial discriminator.
        # The spatial discriminator picks uniformly at random 8 out of 18 lead times.
        self.num_spatial_frames = num_spatial_frames
        # Input size ratio with respect to crop size for the temporal discriminator.
        self.temporal_crop_ratio = temporal_crop_ratio
        # As the input is the whole sequence of the event (including conditioning
        # frames), the spatial discriminator needs to pick only the t > T+0.
        self.num_conditioning_frames = num_conditioning_frames
        self.spatial_discriminator = SpatialDiscriminator()
        self.temporal_discriminator = TemporalDiscriminator()

    @tf.function
    def call(self, frames, training):
        """Build the discriminator.

        Args:
          frames: a tensor with a complete observation [b, 22, 256, 256, 1].

        Returns:
          A tensor with discriminator loss scalars with shape [b, 2, 1]. one for spatial and one for temporal
        """
        b, t, h, w, c = frames.shape

        # Prepare the frames for spatial discriminator: pick 8 random time steps out of 18 lead time steps
        # and down sample from 256x256 to 128x128.
        # this returns a numpy array of indexes of lead times.
        target_frames_sel = tf.range(self.num_conditioning_frames, t)
        # Stacks a list of rank-R tensors into one rank-(R+1) tensor.
        # tf.stack([x,y,z],axis=0), stack by rows.
        # permutation returns tf.tensor with shape (b*_num_spatial_frames), dtype is int32.
        permutation = tf.stack([
            tf.random.shuffle(target_frames_sel)[:self.num_spatial_frames]
            for _ in range(b)
        ], 0)
        # Gather slices from params axis according to indices.
        # indices must be an integer tensor of any dimension (often 1-D)
        # Using batch_dims=1 is equivalent to having an outer loop over the first axis of frames and permutation
        # which means gather correspondent rows
        frames_for_sd = tf.gather(frames, permutation, batch_dims=1)
        # data_format='channels_last' is default.
        frames_for_sd = layers.AveragePooling3D((1, 2, 2), (1, 2, 2),
                                                data_format='channels_last')(frames_for_sd)
        # Compute the average spatial discriminator score for each of 8 picked time steps.
        sd_out = self.spatial_discriminator(frames_for_sd, training=training)

        # Prepare the frames for temporal discriminator: choose the offset of a
        # random crop of size 128x128 out of 256x256 and pick full sequence samples.
        cr = self.temporal_crop_ratio
        h_offset = tf.random.uniform([], 0, (cr - 1) * (h // cr), dtype=tf.dtypes.int32)
        w_offset = tf.random.uniform([], 0, (cr - 1) * (w // cr), dtype=tf.dtypes.int32)
        zero_offset = tf.zeros_like(w_offset)
        # this return a tensor with shape(5,), numpy=array([ 0,  0, X, X,  0])
        begin_tensor = tf.stack(
            [zero_offset, zero_offset, h_offset, w_offset, zero_offset], -1)
        size_tensor = tf.constant([b, t, h // cr, w // cr, c])
        # Extracts a slice from a tensor.
        # This operation extracts a slice of size 'size_tensor' from a tensor 'frames'
        # at the location specified by 'begin_tensor'.
        frames_for_td = tf.slice(frames, begin_tensor, size_tensor)
        frames_for_td.set_shape([b, t, h // cr, w // cr, c])

        # Compute the average temporal discriminator score over length 5 sequences.
        td_out = self.temporal_discriminator(frames_for_td, training=training)
        concat_outputs = layers.Concatenate(axis=1)([sd_out, td_out])

        return concat_outputs

    def get_config(self):
        return {"num_spatial_frames": self.num_spatial_frames,
                "temporal_crop_ratio": self.temporal_crop_ratio,
                "num_conditioning_frames": self.num_conditioning_frames,
                }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


'''
    def get_config(self):
        config = super(Discriminator, self).get_config()
        config.update({"num_spatial_frames": self._num_spatial_frames,
                       "temporal_crop_ratio": self._temporal_crop_ratio,
                       "num_conditioning_frames": self._num_conditioning_frames,
                       "spatial_discriminator": self._spatial_discriminator,
                       "temporal_discriminator": self._temporal_discriminator
                       })
        return config'''


class SpatialDiscriminator(tf.keras.layers.Layer):
    """Spatial Discriminator."""
    def __init__(self, **kwargs):
        super(SpatialDiscriminator, self).__init__(**kwargs)
        self.dblock1 = DBlock.DBlock(output_channels=48, pre_activation=False)
        self.dblock2 = DBlock.DBlock(output_channels=96)
        self.dblock3 = DBlock.DBlock(output_channels=192)
        self.dblock4 = DBlock.DBlock(output_channels=384)
        self.dblock5 = DBlock.DBlock(output_channels=768)
        self.dblock6 = DBlock.DBlock(output_channels=768, down_sample=False)
        # self._output_layer = SpectralNormalization(layers.Dense(1, activation=None))
        self.output_layer = layers.Dense(1, activation=None)
        self.bn = BatchNormalization()

    @tf.function
    def call(self, frames, training):
        """Build the spatial discriminator.

        Args: here n=8=_num_spatial_frames
          frames: a tensor with a complete observation [b, n, 128, 128, 1].

        Returns:
          A tensor with discriminator loss scalars with shape [b, 1, 1].
        """
        b, n, h, w, c = frames.shape.as_list()

        # Process each of the n inputs independently.
        frames = tf.reshape(frames, [b * n, h, w, c])

        # Space-to-depth stacking from 128x128x1 to 64x64x4.
        frames = tf.nn.space_to_depth(frames, block_size=2, data_format='NHWC')

        # Five residual D Blocks to halve the resolution of the image and double
        # the number of channels.
        y = self.dblock1(frames)
        y = self.dblock2(y)
        y = self.dblock3(y)
        y = self.dblock4(y)
        y = self.dblock5(y)

        # One more D Block without down_sampling or increase in number of channels.
        y = self.dblock6(y)

        # Sum-pool the representations and feed to a spectrally normalized linear layer.
        y = tf.math.reduce_sum(activations.relu(y), axis=[1, 2])
        y = self.bn(y, training=training)
        output = self.output_layer(y)

        # Take the sum across the t samples. Note: we apply the ReLU to
        # (1 - score_real) and (1 + score_generated) in the loss.
        output = tf.reshape(output, [b, n, -1])
        output = tf.math.reduce_sum(output, keepdims=True, axis=1)
        return output

    def get_config(self):
        config = super(SpatialDiscriminator, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class TemporalDiscriminator(tf.keras.layers.Layer):
    """Spatial Discriminator."""

    def __init__(self, **kwargs):
        super(TemporalDiscriminator, self).__init__(**kwargs)
        self.dblock1 = DBlock.DBlock(output_channels=48, conv_2d=False,
                                      pooling_2d=False,
                                      pre_activation=False)
        self.dblock2 = DBlock.DBlock(output_channels=96, conv_2d=False,
                                      pooling_2d=False)
        self.dblock3 = DBlock.DBlock(output_channels=192)
        self.dblock4 = DBlock.DBlock(output_channels=384)
        self.dblock5 = DBlock.DBlock(output_channels=768)
        self.dblock6 = DBlock.DBlock(output_channels=768, down_sample=False)
        self.bn = BatchNormalization()
        self.output_layer = layers.Dense(1, activation=None)
        # self._output_layer = SpectralNormalization(layers.Dense(1, activation=None))

    @tf.function
    def call(self, frames, training):
        """Build the temporal discriminator.

        Args:
          frames: a tensor with a complete observation [b, ts, 128, 128, 1]. here ts=22.

        Returns:
          A tensor with discriminator loss scalars with shape [b, 1, 1].
        """
        b, ts, hs, ws, cs = frames.shape.as_list()

        # Process each of the ti inputs independently.
        frames = tf.reshape(frames, [b * ts, hs, ws, cs])

        # Space-to-depth stacking from 128x128x1 to 64x64x4.
        frames = tf.nn.space_to_depth(frames, block_size=2, data_format='NHWC')
        _m, hm, wm, cm = frames.shape.as_list()

        # Stack back to sequences of length ti.
        frames = tf.reshape(frames, [b, -1, hm, wm, cm])

        # Two residual 3D Blocks to halve the resolution of the image, double
        # the number of channels, and reduce the number of time steps.
        y = self.dblock1(frames)
        y = self.dblock2(y)

        # Get t < ts, h, w, and c, as we have down_sampled in 3D.
        _, t, h, w, c = y.shape.as_list()

        # Process each of the t images independently.
        # b t h w c -> (b x t) h w c
        # where -1 will be inferred.
        y = tf.reshape(y, [-1] + [h, w, c])

        # Three residual D Blocks to halve the resolution of the image and double
        # the number of channels.
        y = self.dblock3(y)
        y = self.dblock4(y)
        y = self.dblock5(y)

        # One more D Block without down_sampling or increase in number of channels.
        y = self.dblock6(y)

        # Sum-pool the representations and feed to spectrally normalized lin. layer.
        y = tf.math.reduce_sum(activations.relu(y), axis=[1, 2])
        y = self.bn(y, training=training)
        output = self.output_layer(y)

        # Take the sum across the t samples. Note: we apply the ReLU to
        # (1 - score_real) and (1 + score_generated) in the loss.
        output = tf.reshape(output, [b, t, 1])
        scores = tf.math.reduce_sum(output, keepdims=True, axis=1)
        return scores

    def get_config(self):
        config = super(TemporalDiscriminator, self).get_config()
        '''
        config.update({"dblock1": self.dblock1, "dblock2": self.dblock2,
                       "dblock3": self.dblock3, "dblock4": self.dblock4,
                       "dblock5": self.dblock5, "dblock6": self.dblock6,
                       "output_layer": self.output_layer,
                       "bn": self.bn,
                       })
                       '''
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
