import tensorflow as tf
from tensorflow.keras import layers, activations
import My_Metrics_Eva
import Discriminator
import csv
import Generator
import _DP_ParseTfrecords
import Write_netcdf

"""Pseudocode for the training loop, assuming the UK data.

This code presents, as clearly as possible, the algorithmic logic behind
the generative method. It does not include some control dependencies and
initialization ops that are specific to the hardware architecture on which it is
run as well as specific dataset storage choices.
"""


@tf.function
def get_data_batch(inputs, num_input_frames, num_target_frames):

    """Returns data batch.

    This function should return a pair of (input sequence, target unroll sequence)
    of image frames for a given batch size, with the following dimensions:
    batch_inputs is tensor with size [batch_size, 4, 256, 256, 1],
    batch_targets is tensor with size [batch_size, 18, 256, 256, 1].

    Args:
      batch_size: The batch size, int.

    Returns:
      batch_inputs:
      batch_targets: Data for training.
    """
    '''
    batch_inputs = tf.random.normal(shape=[16, 4, 256, 256, 1])
    batch_targets = tf.random.normal(shape=[16, 18, 256, 256, 1])
    '''
    batch_inputs = inputs[:, -num_target_frames-num_input_frames: -num_target_frames, :, :, :]
    batch_targets = inputs[:, -num_target_frames:, :, :, :]
    return batch_inputs, batch_targets


class DGMR_Train(tf.keras.Model):
    def __init__(self, num_target_frames=18, num_input_frames=4, latent_numbers=1, dis_steps=2,
                 **kwargs):
        super(DGMR_Train, self).__init__(**kwargs)
        """Pseudocode of training loop for the generative method."""
        self.num_target_frames = num_target_frames
        self.num_input_frames = num_input_frames
        self.latent_numbers = latent_numbers
        self.dis_steps = dis_steps
        self.generator_obj = Generator.Generator()
        # the discriminator combines the spatial and temporal discriminators.
        self.discriminator_obj = Discriminator.Discriminator()

        # implement the call method
    @tf.function
    def call(self, inputs, *args, **kwargs):
        print(f'inputs is {inputs.shape}')
        return inputs

    def compile(self, disc_optimizer, gen_optimizer,
                disc_loss_fun, grid_cell_reg_fun, gen_disc_loss_fun):
        super(DGMR_Train, self).compile()
        self.disc_optimizer = disc_optimizer
        self.gen_optimizer = gen_optimizer
        self.loss_disc_fun = disc_loss_fun
        self.grid_cell_reg_fun = grid_cell_reg_fun
        self.loss_gen_disc_fun = gen_disc_loss_fun
        self.disc_loss = tf.keras.metrics.Mean(name='disc_loss')
        self.gen_loss = tf.keras.metrics.Mean(name='gen_loss')
        self.auc = tf.keras.metrics.AUC(num_thresholds=200, name='AUC')
        # self.auc_5 = tf.keras.metrics.AUC(name='AUC_5')
        # self.auc_05 = tf.keras.metrics.AUC(name='AUC_05')
        self.auc_acc_30 = tf.keras.metrics.AUC(name='AUC_acc_30')
        self.auc_acc_60 = tf.keras.metrics.AUC(name='AUC_acc_60')
        self.auc_acc_90 = tf.keras.metrics.AUC(name='AUC_acc_90')
        self.mse_ori = tf.keras.metrics.MeanSquaredError(name='mse_ori')
        self.mse = tf.keras.metrics.Mean(name='MSE')
        self.mse_sd = tf.keras.metrics.Mean(name='MSE_sd')
        self.mse_acc_30 = tf.keras.metrics.Mean(name='MSE_acc_30')
        self.mse_acc_30_sd = tf.keras.metrics.Mean(name='mse_acc_30_sd')
        self.mse_acc_60 = tf.keras.metrics.Mean(name='MSE_acc_60')
        self.mse_acc_60_sd = tf.keras.metrics.Mean(name='mse_acc_60_sd')
        self.mse_acc_90 = tf.keras.metrics.Mean(name='MSE_acc_90')
        self.mse_acc_90_sd = tf.keras.metrics.Mean(name='mse_acc_90_sd')
        self.mse_extreme = tf.keras.metrics.Mean(name='MSE_extreme')
        self.mse_extreme_sd = tf.keras.metrics.Mean(name='MSE_extreme_sd')
        self.mse_large = tf.keras.metrics.Mean(name='MSE_large')
        self.mse_large_sd = tf.keras.metrics.Mean(name='MSE_large_sd')
        self.mse_medium = tf.keras.metrics.Mean(name='MSE_medium')
        self.mse_medium_sd = tf.keras.metrics.Mean(name='MSE_medium_sd')
        self.mse_small = tf.keras.metrics.Mean(name='MSE_small')
        self.mse_small_sd = tf.keras.metrics.Mean(name='MSE_small_sd')
        self.mse_5 = tf.keras.metrics.Mean(name='MSE_5')
        self.mse_5_sd = tf.keras.metrics.Mean(name='MSE_5_sd')
        self.mse_05 = tf.keras.metrics.Mean(name='MSE_05')
        self.mse_05_sd = tf.keras.metrics.Mean(name='MSE_05_sd')
        self.nse = tf.keras.metrics.Mean(name='NSE')
        self.nse_sd = tf.keras.metrics.Mean(name='NSE_sd')
        self.pcc = tf.keras.metrics.Mean(name='PCC')
        self.pcc_sd = tf.keras.metrics.Mean(name='PCC_sd')
        self.pcc_extreme = tf.keras.metrics.Mean(name='PCC_extreme')
        self.pcc_extreme_sd = tf.keras.metrics.Mean(name='PCC_extreme_sd')
        self.pcc_large = tf.keras.metrics.Mean(name='PCC_large')
        self.pcc_large_sd = tf.keras.metrics.Mean(name='PCC_large_sd')
        self.pcc_medium = tf.keras.metrics.Mean(name='PCC_medium')
        self.pcc_medium_sd = tf.keras.metrics.Mean(name='PCC_medium_sd')
        self.pcc_small = tf.keras.metrics.Mean(name='PCC_small')
        self.pcc_small_sd = tf.keras.metrics.Mean(name='PCC_small_sd')
        self.pcc_5 = tf.keras.metrics.Mean(name='PCC_5')
        self.pcc_5_sd = tf.keras.metrics.Mean(name='PCC_5_sd')
        self.pcc_05 = tf.keras.metrics.Mean(name='PCC_05')
        self.pcc_05_sd = tf.keras.metrics.Mean(name='PCC_05_sd')
        self.csi_extreme = tf.keras.metrics.Mean(name='CSI_extreme')
        self.csi_large = tf.keras.metrics.Mean(name='CSI_large')
        self.csi_medium = tf.keras.metrics.Mean(name='CSI_medium')
        self.csi_small = tf.keras.metrics.Mean(name='CSI_small')
        self.csi_5 = tf.keras.metrics.Mean(name='CSI_5')
        self.csi_05 = tf.keras.metrics.Mean(name='CSI_05')
        self.precision_ori = tf.keras.metrics.Precision(name='precision_ori')
        self.precision_extreme = tf.keras.metrics.Mean(name='precision_extreme')
        self.precision_large = tf.keras.metrics.Mean(name='precision_large')
        self.precision_medium = tf.keras.metrics.Mean(name='precision_medium')
        self.precision_small = tf.keras.metrics.Mean(name='precision_small')
        self.precision_5 = tf.keras.metrics.Mean(name='precision_5')
        self.precision_05 = tf.keras.metrics.Mean(name='precision_05')
        self.recall_ori = tf.keras.metrics.Recall(name='recall_ori')
        self.recall_extreme = tf.keras.metrics.Mean(name='recall_extreme')
        self.recall_large = tf.keras.metrics.Mean(name='recall_large')
        self.recall_medium = tf.keras.metrics.Mean(name='recall_medium')
        self.recall_small = tf.keras.metrics.Mean(name='recall_small')
        self.recall_5 = tf.keras.metrics.Mean(name='recall_5')
        self.recall_05 = tf.keras.metrics.Mean(name='recall_05')
        self.mse_time = tf.keras.metrics.MeanTensor(name='MSE_time', shape=(18,))
        self.mse_time_sd = tf.keras.metrics.MeanTensor(name='MSE_time_sd', shape=(18,))
        self.pcc_time = tf.keras.metrics.MeanTensor(name='PCC_time', shape=(18,))
        self.pcc_time_sd = tf.keras.metrics.MeanTensor(name='PCC_time_sd', shape=(18,))
        self.csi_extreme_time = tf.keras.metrics.MeanTensor(name='csi_extreme_time', shape=(18,))
        self.csi_large_time = tf.keras.metrics.MeanTensor(name='csi_large_time', shape=(18,))
        self.csi_medium_time = tf.keras.metrics.MeanTensor(name='csi_medium_time', shape=(18,))
        self.csi_small_time = tf.keras.metrics.MeanTensor(name='csi_small_time', shape=(18,))
        self.csi_5_time = tf.keras.metrics.MeanTensor(name='csi_5_time', shape=(18,))
        self.csi_05_time = tf.keras.metrics.MeanTensor(name='csi_05_time', shape=(18,))
        self.precision_extreme_time = tf.keras.metrics.MeanTensor(name='precision_extreme_time', shape=(18,))
        self.precision_large_time = tf.keras.metrics.MeanTensor(name='precision_large_time', shape=(18,))
        self.precision_medium_time = tf.keras.metrics.MeanTensor(name='precision_medium_time', shape=(18,))
        self.precision_small_time = tf.keras.metrics.MeanTensor(name='precision_small_time', shape=(18,))
        self.precision_5_time = tf.keras.metrics.MeanTensor(name='precision_5_time', shape=(18,))
        self.precision_05_time = tf.keras.metrics.MeanTensor(name='precision_05_time', shape=(18,))
        self.recall_extreme_time = tf.keras.metrics.MeanTensor(name='recall_extreme_time', shape=(18,))
        self.recall_large_time = tf.keras.metrics.MeanTensor(name='recall_large_time', shape=(18,))
        self.recall_medium_time = tf.keras.metrics.MeanTensor(name='recall_medium_time', shape=(18,))
        self.recall_small_time = tf.keras.metrics.MeanTensor(name='recall_small_time', shape=(18,))
        self.recall_5_time = tf.keras.metrics.MeanTensor(name='recall_5_time', shape=(18,))
        self.recall_05_time = tf.keras.metrics.MeanTensor(name='recall_05_time', shape=(18,))

    @property
    def metrics(self):
        return [self.disc_loss, self.gen_loss, self.auc, self.auc_acc_30, self.auc_acc_60, self.auc_acc_90, self.mse_ori,
                self.mse, self.mse_sd, self.mse_acc_30, self.mse_acc_30_sd, self.mse_acc_60, self.mse_acc_60_sd,
                self.mse_acc_90, self.mse_acc_90_sd, self.mse_extreme, self.mse_extreme_sd, self.mse_large,
                self.mse_large_sd, self.mse_medium, self.mse_medium_sd, self.mse_small, self.mse_small_sd,
                self.mse_5, self.mse_5_sd, self.mse_05, self.mse_05_sd, self.nse, self.nse_sd, self.pcc,
                self.pcc_sd, self.pcc_extreme, self.pcc_extreme_sd, self.pcc_large, self.pcc_large_sd, self.pcc_medium,
                self.pcc_medium_sd, self.pcc_small, self.pcc_small_sd, self.pcc_5, self.pcc_5_sd, self.pcc_05,
                self.pcc_05_sd, self.csi_extreme, self.csi_large, self.csi_medium, self.csi_small,
                self.csi_5, self.csi_05, self.precision_ori, self.precision_extreme, self.precision_large,
                self.precision_medium, self.precision_small, self.precision_5, self.precision_05, self.recall_ori,
                self.recall_extreme, self.recall_large, self.recall_medium, self.recall_small,
                self.recall_5, self.recall_05, self.mse_time, self.mse_time_sd, self.pcc_time, self.pcc_time_sd,
                self.csi_extreme_time, self.csi_large_time, self.csi_medium_time, self.csi_small_time,
                self.csi_5_time , self.csi_05_time, self.precision_extreme_time, self.precision_large_time,
                self.precision_medium_time, self.precision_small_time, self.precision_5_time , self.precision_05_time,
                self.recall_extreme_time, self.recall_large_time, self.recall_medium_time, self.recall_small_time,
                self.recall_5_time, self.recall_05_time]

    @tf.function
    def train_step(self, inputs):
        # Unpack the data. Its structure depends on your model and on what you pass to `fit()`.
        """"
        if len(inputs_seq) == 2:
            inputs, sample_weight = inputs_seq
        else:
            sample_weight = None
            inputs = inputs_seq"""
        sample_weight = None
        '''
        dataset = _DP_ParseTfrecords.reader(split="train", variant="random_crops_256")
        row = next(iter(dataset.batch(self._batch_size)))
        radar_frames = row["radar_frames"]
        print(f'radar_frames type is {type(radar_frames)}')
        print(f'radar_frames is {next(iter(row.values())).shape[0]}')
        print(f'radar_frames shape is {radar_frames.shape}')'''
        # layers.Input(shape=(24, 256, 256, 1), batch_size=16)
        # input_frames = tf.math.maximum(inputs, 0.)
        # input_frames = tf.expand_dims(input_frames, 0)
        # input_frames = tf.tile(input_frames, multiples=[1, 1, 1, 1, 1])
        # print(f'input_frames shape is {input_frames.shape}')
        batch_inputs, batch_targets = get_data_batch(inputs, self.num_input_frames, self.num_target_frames)
        batch_predictions = self.generator_obj(batch_inputs, training=True)
        # gen_sequence and real_sequence are tensors with shape [batch_size, num_input_frames + num_target_frames, H, W, C]
        gen_sequence1 = layers.concatenate([batch_inputs, batch_predictions], axis=1)
        real_sequence = layers.concatenate([batch_inputs, batch_targets], axis=1)
        # Concatenate the real and generated samples along the batch dimension
        # concat_inputs is tensor with shape [2*batch_size, num_input_frames + num_target_frames, H, W, C]
        concat_inputs = layers.concatenate([real_sequence, gen_sequence1], axis=0)

        # train discriminator 2 steps and then train the generator for 1 step
        for _ in tf.range(self.dis_steps):
            with tf.GradientTape() as tape:
                # concat_outputs is tensor with shape [2*batch_size, 2, 1]
                concat_outputs = self.discriminator_obj(concat_inputs, training=True)
                # And split back to scores for real and generated samples
                # score_real and score_generated are tensors with shape [batch_size, 2, 1]
                score_real, score_generated = tf.split(concat_outputs, 2, axis=0)
                # loss is tensor with a scalar
                disc_losses = self.loss_disc_fun(score_generated, score_real)
                # make sure .variables or trainable weights
            grads = tape.gradient(disc_losses, self.discriminator_obj.trainable_variables)
            self.disc_optimizer.apply_gradients(zip(grads, self.discriminator_obj.trainable_variables))

        # Train the generator (note that we should *not* update the weights of the discriminator)!
        with tf.GradientTape() as tape:
            num_samples_per_input = self.latent_numbers
            generated_samples = [
                self.generator_obj(batch_inputs, training=True) for _ in range(num_samples_per_input)]
            gen_sequence2 = [layers.concatenate([batch_inputs, x], axis=1) for x in generated_samples]
            gen_sequence2_concat_input = layers.concatenate(gen_sequence2, axis=0)
            # the grid_cell_reg_input shape is (num_samples_per_input, B, self._num_target_frames, 256, 256, 1)
            grid_cell_reg_input = tf.stack(generated_samples, axis=0)
            grid_cell_reg = self.grid_cell_reg_fun(grid_cell_reg_input, batch_targets)
            gen_disc_loss = self.loss_gen_disc_fun(gen_sequence2_concat_input)
            gen_losses = gen_disc_loss + 20.0 * grid_cell_reg
        grads = tape.gradient(gen_losses, self.generator_obj.trainable_weights)
        self.gen_optimizer.apply_gradients(zip(grads, self.generator_obj.trainable_variables))
        # Update metrics
        # Similarly, we call self.compiled_metrics.update_state(y, y_pred) to update the state of the metrics that were
        # passed in compile(), and we query results from self.metrics at the end to retrieve their current value.
        b, t, h, w, c = batch_targets.shape.as_list()
        # reshape inputs shape for tf.image.central_crop.
        batch_targets_re = tf.reshape(batch_targets, [b * t, h, w, c])
        batch_predictions_re = tf.reshape(batch_predictions, [b * t, h, w, c])
        # crop to use the central 64 grids for the evaluation. Since we know the full crop is 256*256, so ratio=0.25
        # or we need to calculate the ratio.
        batch_targets_re_crop = tf.image.central_crop(batch_targets_re, 0.5)
        batch_predictions_re_crop = tf.image.central_crop(batch_predictions_re, 0.5)
        _, h_re, w_re, _ = batch_targets_re_crop.shape.as_list()
        # reshape back for later calculation
        batch_targets_crop = tf.reshape(batch_targets_re_crop, [b, t, h_re, w_re, c])
        batch_predictions_crop = tf.reshape(batch_predictions_re_crop, [b, t, h_re, w_re, c])
        # calculate accumulation every 30mins
        batch_targets_crop_30, batch_targets_crop_60, batch_targets_crop_90 = \
            tf.split(batch_targets_crop, 3, axis=1)
        batch_predictions_crop_30, batch_predictions_crop_60, batch_predictions_crop_90 = \
            tf.split(batch_predictions_crop, 3, axis=1)
        batch_targets_crop_acc_30 = tf.math.multiply(
            tf.math.reduce_mean(batch_targets_crop_30, axis=1, keepdims=True), tf.constant(0.5))
        batch_targets_crop_acc_60 = tf.math.multiply(
            tf.math.reduce_mean(batch_targets_crop_60, axis=1, keepdims=True), tf.constant(0.5))
        batch_targets_crop_acc_90 = tf.math.multiply(
            tf.math.reduce_mean(batch_targets_crop_90, axis=1, keepdims=True), tf.constant(0.5))
        batch_predictions_crop_acc_30 = tf.math.multiply(
            tf.math.reduce_mean(batch_predictions_crop_30, axis=1, keepdims=True), tf.constant(0.5))
        batch_predictions_crop_acc_60 = tf.math.multiply(
            tf.math.reduce_mean(batch_predictions_crop_60, axis=1, keepdims=True), tf.constant(0.5))
        batch_predictions_crop_acc_90 = tf.math.multiply(
            tf.math.reduce_mean(batch_predictions_crop_90, axis=1, keepdims=True), tf.constant(0.5))
        # mask out Nans, here is the grid cells with number of -1
        crop_non_nan = tf.where(batch_targets == -1., 0., 1.)
        crop_non_nan_30, crop_non_nan_60, crop_non_nan_90 = tf.split(crop_non_nan, 3, axis=1)
        my_metrics_eva__ = My_Metrics_Eva
        mse, mse_sd, mse_time, mse_time_sd = my_metrics_eva__.mse_metric(batch_targets_crop, batch_predictions_crop, None, None, crop_non_nan)
        mse_acc_30, mse_acc_30_sd, _, _ = my_metrics_eva__.mse_metric(batch_targets_crop_acc_30, batch_predictions_crop_acc_30, None, None,
                                                           crop_non_nan_30)
        mse_acc_60, mse_acc_60_sd, _, _ = my_metrics_eva__.mse_metric(batch_targets_crop_acc_60, batch_predictions_crop_acc_60, None, None,
                                                           crop_non_nan_30)
        mse_acc_90, mse_acc_90_sd, _, _ = my_metrics_eva__.mse_metric(batch_targets_crop_acc_90, batch_predictions_crop_acc_90, None, None,
                                                           crop_non_nan_60)
        mse_extreme, mse_extreme_sd, _, _ = my_metrics_eva__.mse_metric(batch_targets_crop, batch_predictions_crop, 50., None, crop_non_nan)
        mse_large, mse_large_sd, _, _ = my_metrics_eva__.mse_metric(batch_targets_crop, batch_predictions_crop, 10.,
                                                         50., crop_non_nan)
        mse_medium, mse_medium_sd, _, _ = my_metrics_eva__.mse_metric(batch_targets_crop, batch_predictions_crop, 2.,
                                                           10., crop_non_nan)
        mse_small, mse_small_sd, _, _ = my_metrics_eva__.mse_metric(batch_targets_crop, batch_predictions_crop, None,
                                                         2., crop_non_nan)
        mse_5, mse_5_sd, _, _ = my_metrics_eva__.mse_metric(batch_targets_crop, batch_predictions_crop, 5.,
                                                 None, crop_non_nan)
        mse_05, mse_05_sd, _, _ = my_metrics_eva__.mse_metric(batch_targets_crop, batch_predictions_crop, 0.5,
                                                   None, crop_non_nan)
        nse, nse_sd = my_metrics_eva__.nse_metric(batch_targets_crop, batch_predictions_crop, crop_non_nan)
        pcc, pcc_sd, pcc_time, pcc_time_sd = my_metrics_eva__.pcc_metric(batch_targets_crop, batch_predictions_crop, None, None, crop_non_nan)
        pcc_extreme, pcc_extreme_sd, _, _ = my_metrics_eva__.pcc_metric(batch_targets_crop, batch_predictions_crop,
                                                             50., None, crop_non_nan)
        pcc_large, pcc_large_sd, _, _ = my_metrics_eva__.pcc_metric(batch_targets_crop, batch_predictions_crop,
                                                         10., 50., crop_non_nan)
        pcc_medium, pcc_medium_sd, _, _ = my_metrics_eva__.pcc_metric(batch_targets_crop, batch_predictions_crop,
                                                           None, None, crop_non_nan)
        pcc_small, pcc_small_sd, _, _ = my_metrics_eva__.pcc_metric(batch_targets_crop, batch_predictions_crop,
                                                         2., 10., crop_non_nan)
        pcc_5, pcc_5_sd, _, _ = my_metrics_eva__.pcc_metric(batch_targets_crop, batch_predictions_crop,
                                                 5., None, crop_non_nan)
        pcc_05, pcc_05_sd, _, _ = my_metrics_eva__.pcc_metric(batch_targets_crop, batch_predictions_crop,
                                                   0.5, None, crop_non_nan)
        csi_extreme, recall_extreme, precision_extreme, csi_time_extreme, recall_time_extreme, precision_time_extreme =\
            my_metrics_eva__.csi_precison_recall_metric(batch_targets_crop, batch_predictions_crop, 50., None,
                                             crop_non_nan)
        csi_large, recall_large, precision_large, csi_time_large, recall_time_large, precision_time_large = \
            my_metrics_eva__.csi_precison_recall_metric(batch_targets_crop, batch_predictions_crop, 10., 50.,
                                             crop_non_nan)
        csi_medium, recall_medium, precision_medium, csi_time_medium, recall_time_medium, precision_time_medium =\
            my_metrics_eva__.csi_precison_recall_metric(batch_targets_crop, batch_predictions_crop, 2., 10.,
                                             crop_non_nan)
        csi_small, recall_small, precision_small, csi_time_small, recall_time_small, precision_time_small = \
            my_metrics_eva__.csi_precison_recall_metric(batch_targets_crop, batch_predictions_crop, None, 2.,
                                             crop_non_nan)
        csi_5, recall_5, precision_5, csi_5_time, recall_5_time, precision_5_time = \
            my_metrics_eva__.csi_precison_recall_metric(batch_targets_crop, batch_predictions_crop, 5., None,
                                                        crop_non_nan)
        csi_05, recall_05, precision_05, csi_05_time, recall_05_time, precision_05_time = \
            my_metrics_eva__.csi_precison_recall_metric(batch_targets_crop, batch_predictions_crop, 0.5, None,
                                                        crop_non_nan)
        self.disc_loss.update_state(disc_losses)
        self.gen_loss.update_state(gen_losses)
        batch_targets_crop_scale = tf.keras.layers.Rescaling(scale=1./100)(batch_targets_crop)
        batch_predictions_crop_scale = tf.keras.layers.Rescaling(scale=1. / 100)(batch_predictions_crop)
        batch_targets_crop_acc_30_scale = tf.keras.layers.Rescaling(scale=1. / 100)(batch_targets_crop_acc_30)
        batch_predictions_crop_acc_30_scale = tf.keras.layers.Rescaling(scale=1. / 100)(batch_predictions_crop_acc_30)
        batch_targets_crop_acc_60_scale = tf.keras.layers.Rescaling(scale=1. / 100)(batch_targets_crop_acc_60)
        batch_predictions_crop_acc_60_scale = tf.keras.layers.Rescaling(scale=1. / 100)(batch_predictions_crop_acc_60)
        batch_targets_crop_acc_90_scale = tf.keras.layers.Rescaling(scale=1. / 100)(batch_targets_crop_acc_90)
        batch_predictions_crop_acc_90_scale = tf.keras.layers.Rescaling(scale=1. / 100)(batch_predictions_crop_acc_90)
        self.auc.update_state(batch_targets_crop_scale, batch_predictions_crop_scale)
        self.auc_acc_30.update_state(batch_targets_crop_acc_30_scale, batch_predictions_crop_acc_30_scale)
        self.auc_acc_60.update_state(batch_targets_crop_acc_60_scale, batch_predictions_crop_acc_60_scale)
        self.auc_acc_90.update_state(batch_targets_crop_acc_90_scale, batch_predictions_crop_acc_90_scale)
        self.mse_ori.update_state(batch_targets_crop, batch_predictions_crop)
        self.mse.update_state(mse)
        self.mse_sd.update_state(mse_sd)
        self.mse_time.update_state(mse_time)
        self.mse_time_sd.update_state(mse_time_sd)
        self.mse_acc_30.update_state(mse_acc_30)
        self.mse_acc_30_sd.update_state(mse_acc_30_sd)
        self.mse_acc_60.update_state(mse_acc_60)
        self.mse_acc_60_sd.update_state(mse_acc_60_sd)
        self.mse_acc_90.update_state(mse_acc_90)
        self.mse_acc_90_sd.update_state(mse_acc_90_sd)
        self.mse_extreme.update_state(mse_extreme)
        self.mse_extreme_sd.update_state(mse_extreme_sd)
        self.mse_large.update_state(mse_large)
        self.mse_large_sd.update_state(mse_large_sd)
        self.mse_medium.update_state(mse_medium)
        self.mse_medium_sd.update_state(mse_medium_sd)
        self.mse_small.update_state(mse_small)
        self.mse_small_sd.update_state(mse_small_sd)
        self.mse_5.update_state(mse_5)
        self.mse_5_sd.update_state(mse_5_sd)
        self.mse_05.update_state(mse_05)
        self.mse_05_sd.update_state(mse_05_sd)
        self.nse.update_state(nse)
        self.nse_sd.update_state(nse_sd)
        self.pcc.update_state(pcc)
        self.pcc_sd.update_state(pcc_sd)
        self.pcc_extreme.update_state(pcc_extreme)
        self.pcc_extreme_sd.update_state(pcc_extreme_sd)
        self.pcc_large.update_state(pcc_large)
        self.pcc_large_sd.update_state(pcc_large_sd)
        self.pcc_medium.update_state(pcc_medium)
        self.pcc_medium_sd.update_state(pcc_medium_sd)
        self.pcc_small.update_state(pcc_small)
        self.pcc_small_sd.update_state(pcc_small_sd)
        self.pcc_5.update_state(pcc_5)
        self.pcc_5_sd.update_state(pcc_5_sd)
        self.pcc_05.update_state(pcc_05)
        self.pcc_05_sd.update_state(pcc_05_sd)
        self.pcc_time.update_state(pcc_time)
        self.pcc_time_sd.update_state(pcc_time_sd)
        self.csi_extreme.update_state(csi_extreme)
        self.csi_large.update_state(csi_large)
        self.csi_medium.update_state(csi_medium)
        self.csi_small.update_state(csi_small)
        self.csi_5.update_state(csi_5)
        self.csi_05.update_state(csi_05)
        self.precision_ori.update_state(batch_targets_crop, batch_predictions_crop)
        self.precision_extreme.update_state(precision_extreme)
        self.precision_large.update_state(precision_large)
        self.precision_medium.update_state(precision_medium)
        self.precision_small.update_state(precision_small)
        self.precision_5.update_state(precision_5)
        self.precision_05.update_state(precision_05)
        self.recall_ori.update_state(batch_targets_crop, batch_predictions_crop)
        self.recall_extreme.update_state(recall_extreme)
        self.recall_large.update_state(recall_large)
        self.recall_medium.update_state(recall_medium)
        self.recall_small.update_state(recall_small)
        self.recall_5.update_state(recall_5)
        self.recall_05.update(recall_05)
        self.csi_extreme_time.update_state(csi_time_extreme)
        self.csi_large_time.update_state(csi_time_large)
        self.csi_medium_time.update_state(csi_time_medium)
        self.csi_small_time.update_state(csi_time_small)
        self.csi_5_time.update_state(csi_5_time)
        self.csi_05_time.update_state(csi_05_time)
        self.precision_extreme_time.update_state(precision_time_extreme)
        self.precision_large_time.update_state(precision_time_large)
        self.precision_medium_time.update_state(precision_time_medium)
        self.precision_small_time.update_state(precision_time_small)
        self.precision_5_time.update_state(precision_5_time)
        self.precision_05_time.update_state(precision_05_time)
        self.recall_extreme_time.update_state(recall_time_extreme)
        self.recall_large_time.update_state(recall_time_large)
        self.recall_medium_time.update_state(recall_time_medium)
        self.recall_small_time.update_state(recall_time_small)
        self.recall_5_time.update_state(recall_5_time)
        self.recall_05_time.update_state(recall_05_time)
        return {m.name: m.result() for m in self.metrics}

    # This is test_step for evaluation
    @tf.function
    def test_step(self, inputs):
        sample_weight = None
        inputs_seq = inputs["radar_frames"]
        print(f'type inputs_seq is {type(inputs_seq)}')
        time_stamp_1 = inputs["end_timestamp_seq"]
        print(f'type time_stamp_1 is {time_stamp_1}')
        batch_inputs, batch_targets = get_data_batch(inputs_seq, self.num_input_frames, self.num_target_frames)
        num_samples_per_input = self.latent_numbers # this is a parameter that we need to tune
        print(f'num_samples_per_input {num_samples_per_input}')
        # this is another method to do ensemble prediction, this size
        ''''
        generated_sample_sum = tf.zeros([1, 18, 256, 256, 1])
        for _ in tf.range(num_samples_per_input):
            generated_sample_temp = self.generator_obj(batch_inputs, training=False)
            generated_sample_sum = tf.math.add(generated_sample_sum, generated_sample_temp)'''
        # this is the method used in the pseudocode
        generated_samples = [
            self.generator_obj(batch_inputs, training=False) for _ in range(num_samples_per_input)]
        generated_sample_sum = tf.math.add_n(generated_samples)
        # this is the calculation
        batch_predictions = tf.math.divide_no_nan(generated_sample_sum, num_samples_per_input)
        Write_netcdf.write_netcdf(time_stamp_1, batch_predictions)
        # gen_loss calculation
        gen_sequence1 = layers.concatenate([batch_inputs, batch_predictions], axis=1)
        print(f'type gen_sequence1 is {type(gen_sequence1)}')
        print(f' gen_sequence1 is {gen_sequence1}')
        grid_cell_reg = self.grid_cell_reg_fun(batch_predictions, batch_targets)
        gen_disc_loss = self.loss_gen_disc_fun(gen_sequence1)
        gen_losses = gen_disc_loss + 20.0 * grid_cell_reg
        # dic_loss calculation
        real_sequence = layers.concatenate([batch_inputs, batch_targets], axis=1)
        concat_inputs = layers.concatenate([real_sequence, gen_sequence1], axis=0)
        concat_outputs = self.discriminator_obj(concat_inputs, training=False)
        score_real, score_generated = tf.split(concat_outputs, 2, axis=0)
        disc_losses = self.loss_disc_fun(score_generated, score_real)
        # calculation of metric, just use predictions, t=18
        b, t, h, w, c = batch_targets.shape.as_list()
        # reshape inputs shape for tf.image.central_crop.
        batch_targets_re = tf.reshape(batch_targets, [b * t, h, w, c])
        batch_predictions_re = tf.reshape(batch_predictions, [b * t, h, w, c])
        # crop to use the central 64 grids for the evaluation. Since we know the full crop is 256*256, so ratio=0.25
        # or we need to calculate the ratio.
        batch_targets_re_crop = tf.image.central_crop(batch_targets_re, 0.5)
        batch_predictions_re_crop = tf.image.central_crop(batch_predictions_re, 0.5)
        _, h_re, w_re, _ = batch_targets_re_crop.shape.as_list()
        # reshape back for later calculation
        batch_targets_crop = tf.reshape(batch_targets_re_crop, [b, t, h_re, w_re, c])
        batch_predictions_crop = tf.reshape(batch_predictions_re_crop, [b, t, h_re, w_re, c])
        # calculate accumulation every 30mins
        batch_targets_crop_30, batch_targets_crop_60, batch_targets_crop_90 = \
            tf.split(batch_targets_crop, 3, axis=1)
        batch_predictions_crop_30, batch_predictions_crop_60, batch_predictions_crop_90 = \
            tf.split(batch_predictions_crop, 3, axis=1)
        batch_targets_crop_acc_30 = tf.math.multiply(
            tf.math.reduce_mean(batch_targets_crop_30, axis=1, keepdims=True), tf.constant(0.5))
        batch_targets_crop_acc_60 = tf.math.multiply(
            tf.math.reduce_mean(batch_targets_crop_60, axis=1, keepdims=True), tf.constant(0.5))
        batch_targets_crop_acc_90 = tf.math.multiply(
            tf.math.reduce_mean(batch_targets_crop_90, axis=1, keepdims=True), tf.constant(0.5))
        batch_predictions_crop_acc_30 = tf.math.multiply(
            tf.math.reduce_mean(batch_predictions_crop_30, axis=1, keepdims=True), tf.constant(0.5))
        batch_predictions_crop_acc_60 = tf.math.multiply(
            tf.math.reduce_mean(batch_predictions_crop_60, axis=1, keepdims=True), tf.constant(0.5))
        batch_predictions_crop_acc_90 = tf.math.multiply(
            tf.math.reduce_mean(batch_predictions_crop_90, axis=1, keepdims=True), tf.constant(0.5))
        # mask out Nans, here is the grid cells with number of -1
        crop_non_nan = tf.where(batch_targets_crop == -1., 0., 1.)
        crop_non_nan_30, crop_non_nan_60, crop_non_nan_90 = tf.split(crop_non_nan, 3, axis=1)
        my_metrics_eva__ = My_Metrics_Eva
        mse, mse_sd, mse_time, mse_time_sd = my_metrics_eva__.mse_metric(batch_targets_crop, batch_predictions_crop,
                                                              None, None, crop_non_nan)
        mse_acc_30, mse_acc_30_sd, _, _ = my_metrics_eva__.mse_metric(batch_targets_crop_acc_30,
                                                           batch_predictions_crop_acc_30, None, None,
                                                           crop_non_nan_30)
        mse_acc_60, mse_acc_60_sd, _, _ = my_metrics_eva__.mse_metric(batch_targets_crop_acc_60,
                                                           batch_predictions_crop_acc_60, None, None,
                                                           crop_non_nan_30)
        mse_acc_90, mse_acc_90_sd, _, _ = my_metrics_eva__.mse_metric(batch_targets_crop_acc_90,
                                                           batch_predictions_crop_acc_90, None, None,
                                                           crop_non_nan_60)
        mse_extreme, mse_extreme_sd, _, _ = my_metrics_eva__.mse_metric(batch_targets_crop, batch_predictions_crop, 50.,
                                                             None, crop_non_nan)
        mse_large, mse_large_sd, _, _ = my_metrics_eva__.mse_metric(batch_targets_crop, batch_predictions_crop, 10.,
                                                         50., crop_non_nan)
        mse_medium, mse_medium_sd, _, _ = my_metrics_eva__.mse_metric(batch_targets_crop, batch_predictions_crop, 2.,
                                                           10., crop_non_nan)
        mse_small, mse_small_sd, _, _ = my_metrics_eva__.mse_metric(batch_targets_crop, batch_predictions_crop, None,
                                                         2., crop_non_nan)
        mse_5, mse_5_sd, _, _ = my_metrics_eva__.mse_metric(batch_targets_crop, batch_predictions_crop, 5.,
                                                 None, crop_non_nan)
        mse_05, mse_05_sd, _, _ = my_metrics_eva__.mse_metric(batch_targets_crop, batch_predictions_crop, 0.5,
                                                   None, crop_non_nan)
        nse, nse_sd = my_metrics_eva__.nse_metric(batch_targets_crop, batch_predictions_crop, crop_non_nan)
        pcc, pcc_sd, pcc_time, pcc_time_sd = my_metrics_eva__.pcc_metric(batch_targets_crop, batch_predictions_crop,
                                                              None, None, crop_non_nan)
        pcc_extreme, pcc_extreme_sd, _, _ = my_metrics_eva__.pcc_metric(batch_targets_crop, batch_predictions_crop,
                                                             50., None, crop_non_nan)
        pcc_large, pcc_large_sd, _, _ = my_metrics_eva__.pcc_metric(batch_targets_crop, batch_predictions_crop,
                                                         10., 50., crop_non_nan)
        pcc_medium, pcc_medium_sd, _, _ = my_metrics_eva__.pcc_metric(batch_targets_crop, batch_predictions_crop,
                                                           2., 10., crop_non_nan)
        pcc_small, pcc_small_sd, _, _ = my_metrics_eva__.pcc_metric(batch_targets_crop, batch_predictions_crop,
                                                         None, 2., crop_non_nan)
        pcc_5, pcc_5_sd, _, _ = my_metrics_eva__.pcc_metric(batch_targets_crop, batch_predictions_crop,
                                                 5., None, crop_non_nan)
        pcc_05, pcc_05_sd, _, _ = my_metrics_eva__.pcc_metric(batch_targets_crop, batch_predictions_crop,
                                                   0., None, crop_non_nan)
        #csi, recall, precision, csi_time, recall_time, precision_time = my_metrics_eva__.csi_precison_recall_metric(
        #    batch_targets_crop, batch_predictions_crop, None, None, crop_non_nan)
        csi_extreme, recall_extreme, precision_extreme, csi_time_extreme, recall_time_extreme, precision_time_extreme = \
            my_metrics_eva__.csi_precison_recall_metric(batch_targets_crop, batch_predictions_crop, 50., None,
                                             crop_non_nan)
        csi_large, recall_large, precision_large, csi_time_large, recall_time_large, precision_time_large = \
            my_metrics_eva__.csi_precison_recall_metric(batch_targets_crop, batch_predictions_crop, 10., 50.,
                                             crop_non_nan)
        csi_medium, recall_medium, precision_medium, csi_time_medium, recall_time_medium, precision_time_medium = \
            my_metrics_eva__.csi_precison_recall_metric(batch_targets_crop, batch_predictions_crop, 2., 10.,
                                             crop_non_nan)
        csi_small, recall_small, precision_small, csi_time_small, recall_time_small, precision_time_small = \
            my_metrics_eva__.csi_precison_recall_metric(batch_targets_crop, batch_predictions_crop, None, 2.,
                                             crop_non_nan)
        csi_5, recall_5, precision_5, csi_5_time, recall_5_time, precision_5_time = my_metrics_eva__.csi_precison_recall_metric(
            batch_targets_crop, batch_predictions_crop, 5., None, crop_non_nan
        )
        csi_05, recall_05, precision_05, csi_05_time, recall_05_time, precision_05_time = my_metrics_eva__.csi_precison_recall_metric(
            batch_targets_crop, batch_predictions_crop, 0.5, None, crop_non_nan
        )
        self.disc_loss.update_state(disc_losses)
        self.gen_loss.update_state(gen_losses)
        self.auc.update_state(batch_targets_crop, batch_predictions_crop)
        self.auc_acc_30.update_state(batch_targets_crop_acc_30, batch_predictions_crop_acc_30)
        self.auc_acc_60.update_state(batch_targets_crop_acc_60, batch_predictions_crop_acc_60)
        self.auc_acc_90.update_state(batch_targets_crop_acc_90, batch_predictions_crop_acc_90)
        self.mse_ori.update_state(batch_targets_crop, batch_predictions_crop)
        self.mse.update_state(mse)
        self.mse_sd.update_state(mse_sd)
        self.mse_time.update_state(mse_time)
        self.mse_time_sd.update_state(mse_time_sd)
        self.mse_acc_30.update_state(mse_acc_30)
        self.mse_acc_30_sd.update_state(mse_acc_30_sd)
        self.mse_acc_60.update_state(mse_acc_60)
        self.mse_acc_60_sd.update_state(mse_acc_60_sd)
        self.mse_acc_90.update_state(mse_acc_90)
        self.mse_acc_90_sd.update_state(mse_acc_90_sd)
        self.mse_extreme.update_state(mse_extreme)
        self.mse_extreme_sd.update_state(mse_extreme_sd)
        self.mse_large.update_state(mse_large)
        self.mse_large_sd.update_state(mse_large_sd)
        self.mse_medium.update_state(mse_medium)
        self.mse_medium_sd.update_state(mse_medium_sd)
        self.mse_small.update_state(mse_small)
        self.mse_small_sd.update_state(mse_small_sd)
        self.mse_5.update_state(mse_5)
        self.mse_5_sd.update_state(mse_5_sd)
        self.mse_05.update_state(mse_05)
        self.mse_05_sd.update_state(mse_05_sd)
        self.nse.update_state(nse)
        self.nse_sd.update_state(nse_sd)
        self.pcc.update_state(pcc)
        self.pcc_sd.update_state(pcc_sd)
        self.pcc_extreme.update_state(pcc_extreme)
        self.pcc_extreme_sd.update_state(pcc_extreme_sd)
        self.pcc_large.update_state(pcc_large)
        self.pcc_large_sd.update_state(pcc_large_sd)
        self.pcc_medium.update_state(pcc_medium)
        self.pcc_medium_sd.update_state(pcc_medium_sd)
        self.pcc_small.update_state(pcc_small)
        self.pcc_small_sd.update_state(pcc_small_sd)
        self.pcc_5.update_state(pcc_5)
        self.pcc_5_sd.update_state(pcc_5_sd)
        self.pcc_05.update_state(pcc_05)
        self.pcc_05_sd.update_state(pcc_05_sd)
        self.pcc_time.update_state(pcc_time)
        self.pcc_time_sd.update_state(pcc_time_sd)
        # self.csi.update_state(csi)
        self.csi_extreme.update_state(csi_extreme)
        self.csi_large.update_state(csi_large)
        self.csi_medium.update_state(csi_medium)
        self.csi_small.update_state(csi_small)
        self.csi_5.update_state(csi_5)
        self.csi_05.update_state(csi_05)
        self.precision_ori.update_state(batch_targets_crop, batch_predictions_crop)
        # self.precision.update_state(precision)
        self.precision_extreme.update_state(precision_extreme)
        self.precision_large.update_state(precision_large)
        self.precision_medium.update_state(precision_medium)
        self.precision_small.update_state(precision_small)
        self.precision_5.update_state(precision_5)
        self.precision_05.update_state(precision_05)
        self.recall_ori.update_state(batch_targets_crop, batch_predictions_crop)
        #self.recall.update_state(recall)
        self.recall_extreme.update_state(recall_extreme)
        self.recall_large.update_state(recall_large)
        self.recall_medium.update_state(recall_medium)
        self.recall_small.update_state(recall_small)
        self.recall_5.update_state(recall_5)
        self.recall_05.update_state(recall_05)
        self.csi_extreme_time.update_state(csi_time_extreme)
        self.csi_large_time.update_state(csi_time_large)
        self.csi_medium_time.update_state(csi_time_medium)
        self.csi_small_time.update_state(csi_time_small)
        self.csi_5_time.update_state(csi_5_time)
        self.csi_05_time.update_state(csi_05_time)
        self.precision_extreme_time.update_state(precision_time_extreme)
        self.precision_large_time.update_state(precision_time_large)
        self.precision_medium_time.update_state(precision_time_medium)
        self.precision_small_time.update_state(precision_time_small)
        self.precision_5_time.update_state(precision_5_time)
        self.precision_05_time.update_state(precision_05_time)
        self.recall_extreme_time.update_state(recall_time_extreme)
        self.recall_large_time.update_state(recall_time_large)
        self.recall_medium_time.update_state(recall_time_medium)
        self.recall_small_time.update_state(recall_time_small)
        self.recall_5_time.update_state(recall_5_time)
        self.recall_05_time.update_state(recall_05_time)
        '''
        re_list_name = ['batch_targets_crop', 'batch_predictions_crop',
                        'batch_targets_crop_acc_30', 'batch_targets_crop_acc_60',
                        'batch_targets_crop_acc_90', 'batch_predictions_crop_acc_30',
                        'batch_predictions_crop_acc_60', 'batch_predictions_crop_acc_90']
        re_list = [batch_targets_crop, batch_predictions_crop,
                   batch_targets_crop_acc_30,
        batch_targets_crop_acc_60,
        batch_targets_crop_acc_90,
        batch_predictions_crop_acc_30,
        batch_predictions_crop_acc_60,
        batch_predictions_crop_acc_90]'''
        re_list_name = ['batch_targets_crop', 'batch_predictions_crop']
        re_list_1 = tf.math.multiply(batch_targets_crop, crop_non_nan)
        re_list_2 = tf.math.multiply(batch_predictions_crop, crop_non_nan)
        print(f're_list1 {re_list_1.shape}')
        re_list = [re_list_1, re_list_2]
        # return {m.name: m.result() for m in self.metrics}
        return {key: value for key, value in zip(re_list_name, re_list)}

    # This part is incomplete.
    def predict_step(self, inputs):
        batch_inputs, batch_targets = get_data_batch(inputs, self.num_input_frames, self.num_target_frames)
        batch_predictions = self.generator_obj(batch_inputs)

        gen_sequence1 = layers.concatenate([batch_inputs, batch_predictions], axis=1)
        real_sequence = layers.concatenate([batch_inputs, batch_targets], axis=1)
        concat_inputs = layers.concatenate([real_sequence, gen_sequence1], axis=0)

        concat_outputs = self.discriminator_obj(concat_inputs)
        score_real, score_generated = tf.split(concat_outputs, 2, axis=0)
        disc_losses = self.loss_disc_fun(score_generated, score_real)

        num_samples_per_input = self._latent_numbers
        generated_samples = [
            self.generator_obj(batch_inputs) for _ in range(num_samples_per_input)]
        gen_sequence2 = [layers.concatenate([batch_inputs, x], axis=1) for x in generated_samples]
        gen_sequence2_concat_input = layers.concatenate(gen_sequence2, axis=0)
        # the grid_cell_reg_input shape is (num_samples_per_input, B, self._num_target_frames, 256, 256, 1)
        grid_cell_reg_input = tf.stack(generated_samples, axis=0)
        grid_cell_reg = self.grid_cell_reg_fun(grid_cell_reg_input, batch_targets)
        gen_disc_loss = self.loss_gen_disc_fun(gen_sequence2_concat_input)
        gen_losses = gen_disc_loss + 20.0 * grid_cell_reg

        self.disc_loss.update_state(disc_losses)
        self.gen_loss.update_state(gen_losses)
        self.metric_mse.update_state(real_sequence, gen_sequence1)
        self.metric_nse.update_state(real_sequence, gen_sequence1)

        return {m.name: m.result() for m in self.metrics}

    def get_config(self):
        return {"num_target_frames": self.num_target_frames,
                "num_input_frames": self.num_input_frames,
                "latent_numbers": self.latent_numbers,
                }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


