import tensorflow as tf
from tensorflow.keras import layers, activations
import My_Metrics_Train
import Discriminator
import Generator
import _DP_ParseTfrecords

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
      batch_inputs: shape=[16, 4, 256, 256, 1]
      batch_targets: Data for training, shape=[16, 18, 256, 256, 1]
    """
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
        self.auc = tf.keras.metrics.AUC(name='auc')
        self.metric_mse = tf.keras.metrics.Mean(name='MSE')
        self.metric_nse = tf.keras.metrics.Mean(name='NSE')
        self.metric_nse_ori = tf.keras.metrics.Mean(name='NSE_ori')
        self.metric_pcc = tf.keras.metrics.Mean(name='PCC')
        self.metric_pcc_ori = tf.keras.metrics.Mean(name='PCC_ori')
        self.metric_csi = tf.keras.metrics.Mean(name='CSI')
        self.metric_precision = tf.keras.metrics.Mean(name='Precision')
        self.metric_recall = tf.keras.metrics.Mean(name='Recall')
        self.metric_csi_s = tf.keras.metrics.Mean(name='CSI_Small')
        self.metric_csi_m = tf.keras.metrics.Mean(name='CSI_Medium')
        self.metric_csi_l = tf.keras.metrics.Mean(name='CSI_Large')
        self.metric_csi_e = tf.keras.metrics.Mean(name='CSI_Extreme')
        self.metric_precision_ori = tf.keras.metrics.Precision(name='Precision_Ori')
        self.metric_recall_ori = tf.keras.metrics.Recall(name='Recall_Ori')
        self.metric_recall_e = tf.keras.metrics.Mean(name='Recall_Extreme')
        self.metric_precision_e = tf.keras.metrics.Mean(name='Precision_Extreme')
        self.metric_recall_l = tf.keras.metrics.Mean(name='Recall_Large')
        self.metric_precision_l = tf.keras.metrics.Mean(name='Precision_Large')
        self.metric_recall_m = tf.keras.metrics.Mean(name='Recall_Medium')
        self.metric_precision_m = tf.keras.metrics.Mean(name='Precision_Medium')
        self.metric_recall_s = tf.keras.metrics.Mean(name='Recall_Small')
        self.metric_precision_s = tf.keras.metrics.Mean(name='Precision_Small')

    @property
    def metrics(self):
        return [self.disc_loss, self.gen_loss, self.metric_mse, self.metric_nse, self.metric_nse_ori, self.auc,
                self.metric_pcc, self.metric_pcc_ori, self.metric_csi, self.metric_csi_s, self.metric_csi_m,
                self.metric_csi_l, self.metric_csi_e, self.metric_recall_ori, self.metric_precision_ori,
                self.metric_recall, self.metric_precision, self.metric_recall_e,
                self.metric_precision_e, self.metric_recall_l, self.metric_precision_l,
                self.metric_recall_m, self.metric_precision_m, self.metric_recall_s,
                self.metric_precision_s]

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
        inputs = inputs["radar_frames"]
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
        # actual train samples are 120k, since batch_size=2, so num_train=60k
        num_train = tf.constant(60000.)
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
        batch_predictions = tf.reshape(batch_predictions, [b * t, h, w, c])
        # crop to use the central 64 grids for the evaluation. Since we know the full crop is 256*256, so ratio=0.25
        # or we need to calculate the ratio.
        batch_targets_re_crop = tf.image.central_crop(batch_targets_re, 0.5)
        batch_predictions_re_crop = tf.image.central_crop(batch_predictions, 0.5)
        _, h_re, w_re, _ = batch_targets_re_crop.shape.as_list()
        # reshape back for later calculation
        batch_targets_crop = tf.reshape(batch_targets_re_crop, [b, t, h_re, w_re, c])
        batch_predictions_crop = tf.reshape(batch_predictions_re_crop, [b, t, h_re, w_re, c])
        # mask out Nans, here is the grid cells with number of -1
        crop_non_nan = tf.where(batch_targets_crop == -1., 0., 1.)
        mse = My_Metrics_Train.mse_metric(batch_targets_crop, batch_predictions_crop, None, None, crop_non_nan)
        nse = My_Metrics_Train.nse_metric(batch_targets_crop, batch_predictions_crop, crop_non_nan)
        nse_ori = My_Metrics_Train.nse_metric(batch_targets_crop, batch_predictions_crop, crop_non_nan)
        pcc = My_Metrics_Train.pcc_metric(batch_targets_crop, batch_predictions_crop, None, None, crop_non_nan)
        pcc_ori = My_Metrics_Train.pcc_ori_metric(batch_targets_crop, batch_predictions_crop, crop_non_nan)
        csi, recall, precision = \
            My_Metrics_Train.csi_precison_recall_metric(batch_targets_crop, batch_predictions_crop, None, None,
                                                        crop_non_nan)
        csi_extreme, recall_extreme, precision_extreme = \
            My_Metrics_Train.csi_precison_recall_metric(batch_targets_crop, batch_predictions_crop, 50., None,
                                                        crop_non_nan)
        csi_large, recall_large, precision_large = \
            My_Metrics_Train.csi_precison_recall_metric(batch_targets_crop, batch_predictions_crop, 10., 50.,
                                                        crop_non_nan)
        csi_medium, recall_medium, precision_medium = \
            My_Metrics_Train.csi_precison_recall_metric(batch_targets_crop, batch_predictions_crop, 2., 10.,
                                                        crop_non_nan)
        csi_small, recall_small, precision_small = \
            My_Metrics_Train.csi_precison_recall_metric(batch_targets_crop, batch_predictions_crop, None, 2.,
                                                        crop_non_nan)
        self.disc_loss.update_state(disc_losses)
        self.gen_loss.update_state(gen_losses)
        self.auc.update_state(batch_targets_crop, batch_predictions_crop, sample_weight=None)
        self.metric_precision_ori.update_state(batch_targets_crop, batch_predictions_crop, sample_weight=None)
        self.metric_recall_ori.update_state(batch_targets_crop, batch_predictions_crop, sample_weight=None)
        self.metric_mse.update_state(mse)
        self.metric_nse.update_state(nse)
        self.metric_nse.update_state(nse_ori)
        self.metric_pcc.update_state(pcc)
        self.metric_pcc_ori.update_state(pcc_ori)
        self.metric_csi.update_state(csi)
        self.metric_csi_s.update_state(csi_small)
        self.metric_csi_m.update_state(csi_medium)
        self.metric_csi_l.update_state(csi_large)
        self.metric_csi_e.update_state(csi_extreme)
        self.metric_recall.update_state(recall)
        self.metric_precision.update_state(precision)
        self.metric_recall_e.update_state(recall_extreme)
        self.metric_precision_e.update_state(precision_extreme)
        self.metric_recall_l.update_state(recall_large)
        self.metric_precision_l.update_state(precision_large)
        self.metric_recall_m.update_state(recall_medium)
        self.metric_precision_m.update_state(precision_medium)
        self.metric_recall_s.update_state(recall_small)
        self.metric_precision_s.update_state(precision_small)
        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def test_step(self, inputs_seq):
        # Unpack the data. Its structure depends on your model and on what you pass to `fit()`.
        '''if len(inputs_seq) == 2:
            inputs, sample_weight = inputs_seq
        else:
            sample_weight = None
            inputs = inputs_seq
            '''
        sample_weight = None
        # actual train samples are 14820, since batch_size=2, so num_train=7410
        inputs_seq = inputs_seq["radar_frames"]
        batch_inputs, batch_targets = get_data_batch(inputs_seq, self.num_input_frames, self.num_target_frames)
        batch_predictions = self.generator_obj(batch_inputs, training=False)
        gen_sequence1 = layers.concatenate([batch_inputs, batch_predictions], axis=1)
        real_sequence = layers.concatenate([batch_inputs, batch_targets], axis=1)
        concat_inputs = layers.concatenate([real_sequence, gen_sequence1], axis=0)

        concat_outputs = self.discriminator_obj(concat_inputs, training=False)
        score_real, score_generated = tf.split(concat_outputs, 2, axis=0)
        disc_losses = self.loss_disc_fun(score_generated, score_real)

        num_samples_per_input = self.latent_numbers
        # for loop here instead of tf.stack.
        generated_samples = [
            self.generator_obj(batch_inputs, training=False) for _ in range(num_samples_per_input)]
        gen_sequence2 = [layers.concatenate([batch_inputs, x], axis=1) for x in generated_samples]
        gen_sequence2_concat_input = layers.concatenate(gen_sequence2, axis=0)
        # the grid_cell_reg_input shape is (num_samples_per_input, B, self._num_target_frames, 256, 256, 1)
        grid_cell_reg_input = tf.stack(generated_samples, axis=0)
        grid_cell_reg = self.grid_cell_reg_fun(grid_cell_reg_input, batch_targets)
        gen_disc_loss = self.loss_gen_disc_fun(gen_sequence2_concat_input)
        gen_losses = gen_disc_loss + 20.0 * grid_cell_reg

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
        # mask out Nans, here is the grid cells with number of -1
        crop_non_nan = tf.where(batch_targets_crop == -1., 0., 1.)
        mse = My_Metrics_Train.mse_metric(batch_targets_crop, batch_predictions_crop, None, None, crop_non_nan)
        nse = My_Metrics_Train.nse_metric(batch_targets_crop, batch_predictions_crop, crop_non_nan)
        nse_ori = My_Metrics_Train.nse_metric(batch_targets_crop, batch_predictions_crop, crop_non_nan)
        pcc = My_Metrics_Train.pcc_metric(batch_targets_crop, batch_predictions_crop, None, None, crop_non_nan)
        pcc_ori = My_Metrics_Train.pcc_ori_metric(batch_targets_crop, batch_predictions_crop, crop_non_nan)
        csi, recall, precision = \
            My_Metrics_Train.csi_precison_recall_metric(batch_targets_crop, batch_predictions_crop, None, None,
                                                        crop_non_nan)
        csi_extreme, recall_extreme, precision_extreme = \
            My_Metrics_Train.csi_precison_recall_metric(batch_targets_crop, batch_predictions_crop, 50., None,
                                                        crop_non_nan)
        csi_large, recall_large, precision_large = \
            My_Metrics_Train.csi_precison_recall_metric(batch_targets_crop, batch_predictions_crop, 10., 50.,
                                                        crop_non_nan)
        csi_medium, recall_medium, precision_medium = \
            My_Metrics_Train.csi_precison_recall_metric(batch_targets_crop, batch_predictions_crop, 2., 10.,
                                                        crop_non_nan)
        csi_small, recall_small, precision_small = \
            My_Metrics_Train.csi_precison_recall_metric(batch_targets_crop, batch_predictions_crop, None, 2.,
                                                        crop_non_nan)
        self.disc_loss.update_state(disc_losses)
        self.gen_loss.update_state(gen_losses)
        self.auc.update_state(batch_targets_crop, batch_predictions_crop, sample_weight=None)
        self.metric_precision_ori.update_state(batch_targets_crop, batch_predictions_crop, sample_weight=None)
        self.metric_recall_ori.update_state(batch_targets_crop, batch_predictions_crop, sample_weight=None)
        self.metric_mse.update_state(mse)
        self.metric_nse.update_state(nse)
        self.metric_nse.update_state(nse_ori)
        self.metric_pcc.update_state(pcc)
        self.metric_pcc_ori.update_state(pcc_ori)
        self.metric_csi.update_state(csi)
        self.metric_csi_s.update_state(csi_small)
        self.metric_csi_m.update_state(csi_medium)
        self.metric_csi_l.update_state(csi_large)
        self.metric_csi_e.update_state(csi_small)
        self.metric_recall.update_state(recall)
        self.metric_precision.update_state(precision)
        self.metric_recall_e.update_state(recall_extreme)
        self.metric_precision_e.update_state(precision_extreme)
        self.metric_recall_l.update_state(recall_large)
        self.metric_precision_l.update_state(precision_large)
        self.metric_recall_m.update_state(recall_medium)
        self.metric_precision_m.update_state(precision_medium)
        self.metric_recall_s.update_state(recall_small)
        self.metric_precision_s.update_state(precision_small)
        return {m.name: m.result() for m in self.metrics}

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


