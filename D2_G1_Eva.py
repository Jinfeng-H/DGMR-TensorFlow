import tensorflow as tf
from tensorflow.keras import layers
import DGMR_D2_G1_Eva
from tensorflow.keras.optimizers import Adam
import _DP_ParseTfrecords
import csv
import datetime

''''
# https://forums.developer.nvidia.com/t/could-not-create-cudnn-handle-cudnn-status-alloc-failed/108261/3
# This part is cited from above link to solve the error "Could not create cudnn handle: CUDNN_STATUS_ALLOC_FAILED"
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)
'''
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.config.run_functions_eagerly(True)

# tf.debugging.set_log_device_placement(True)
# tf.config.set_soft_device_placement(True)

# set global strategy for mixed precision, reducing consuming memory.
# https://www.tensorflow.org/guide/mixed_precision#%E8%AE%BE%E7%BD%AE_dtype_%E7%AD%96%E7%95%A5
# policy = tf.keras.mixed_precision.Policy('mixed_float16')
# tf.keras.mixed_precision.set_global_policy(policy)

@tf.function
def loss_hinge_disc(score_generated, score_real):
    """Discriminator hinge loss."""
    l1 = layers.ReLU()(1. - score_real)
    # Computes the mean of elements across dimensions of a tensor.
    # If axis is None, all dimensions are reduced, and a tensor with a single element is returned.
    loss = tf.math.reduce_mean(l1)
    l2 = layers.ReLU()(1. + score_generated)
    loss += tf.math.reduce_mean(l2)
    return loss


@tf.function
def loss_hinge_gen(score_generated):
    """Generator hinge loss."""
    loss = -tf.math.reduce_mean(score_generated)
    return loss


@tf.function
def grid_cell_regularizer(generated_samples, batch_targets):
    """Grid cell regularizer.

    Args:
      generated_samples: Tensor of size [n_samples, batch_size, 18, 256, 256, 1].
      batch_targets: Tensor of size [batch_size, 18, 256, 256, 1].

    Returns:
      loss: A tensor of shape [batch_size].
    """
    gen_mean = tf.math.reduce_mean(generated_samples, axis=0)
    weights = tf.clip_by_value(batch_targets, 0.0, 24.0)
    loss = tf.math.reduce_mean(tf.math.abs(gen_mean - batch_targets) * weights)
    return loss


def train():
    # when batch_size1 changed, we need to change argument of num in tf.unstack in Mylayers.ApplyAlongAxis
    batch_size1 = 1

    # The following part is reading data without sampling
    print(f'let us start')
    dataset_test_2013 = _DP_ParseTfrecords.reader_test(split="test", variant="random_crops_256")
    print(f'parse reader finished.')
    radar_frames_test = dataset_test_2013.batch(batch_size1, drop_remainder=True)

    print(f'let us compile!')
    gan = DGMR_D2_G1_Eva.DGMR_Train()
    gan.built = True
    gan.compile(disc_optimizer=Adam(learning_rate=0.0002, beta_1=0., beta_2=0.999, epsilon=1e-08, global_clipnorm=100.),
                gen_optimizer=Adam(learning_rate=0.00005, beta_1=0., beta_2=0.999, epsilon=1e-08, global_clipnorm=100.),
                disc_loss_fun=loss_hinge_disc,
                grid_cell_reg_fun=grid_cell_regularizer,
                gen_disc_loss_fun=loss_hinge_gen,
                # List of metrics to monitor, we have to add sample weights here
                # metrics=[metrics.RootMeanSquaredError(), My_Metrics.PCCMetric()]
                )

    checkpoint_path = 'D://Jinfeng/v6_modules_get_config/00_D2_G1_Train/ckpt_/4656_58.12.ckpt'
    gan.load_weights(checkpoint_path)
    print('weights loaded successfully!')
    history_eva = gan.evaluate(radar_frames_test, batch_size=batch_size1, verbose=2)
    print('Jinfeng, Well done!!!')

    metric_names = []
    metric_values = []
    for k, v in history_eva.history:
        metric_names.append(k)
        metric_values.append(v)
    metrics_info = [metric_names, metric_values]
    metrics_info_zip = zip(*metrics_info)
    
    with open(
            'D://Jinfeng/v6_modules_get_config/00 making tfrecords and statistics/eva_DGMR_Test_of_200_samples_4656_write_netcdf.csv',
            'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['evaluation metrics of ensemble predictions from DGMR Test'])
        for k in history_eva:
            writer.writerow([k])
        f.close()

    print('Jinfeng, Well done!!!')
    print(f'hisroty_eva is {history_eva}')
    '''
        gan.compile(disc_optimizer=Adam(learning_rate=2E-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
                gen_optimizer=Adam(learning_rate=5E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
                disc_loss_fun=loss_hinge_disc,
                grid_cell_reg_fun=grid_cell_regularizer,
                gen_disc_loss_fun=loss_hinge_gen,
                # List of metrics to monitor, we have to add sample weights here
                # metrics=[metrics.RootMeanSquaredError(), My_Metrics.PCCMetric()]
                )'''

    # The EarlyStopping callback is executed via the on_epoch_end trigger for training.
    # A model.fit() training loop will check at end of every epoch whether monitor="val_loss" is no longer decreasing
    '''
    # this callback was abandoned because of disc_loss quickly decreases to zero.
    callbacks1 = tf.keras.callbacks.EarlyStopping(
        # Stop training when `val_loss` is no longer improving
        monitor='disc_loss',
        # "no longer improving" being defined as "no better than 1e-2 less"
        min_delta=1e-2,
        # "no longer improving" being further defined as "for at least 2 epochs"
        patience=6,
        verbose=1)
'''

    callbacks2 = tf.keras.callbacks.EarlyStopping(
        # Stop training when `val_loss` is no longer improving
        monitor='gen_loss',
        # "no longer improving" being defined as "no better than 1e-2 less"
        min_delta=1e-4,
        # "no longer improving" being further defined as "for at least 2 epochs"
        patience=50,
        verbose=1)

    callbacks3 = tf.keras.callbacks.ModelCheckpoint(
        # Path where to save the model
        filepath="D://Jinfeng/03_training_D2_G1/ckpt/{epoch:02d}_{disc_loss:0.2f}.ckpt",
        save_best_only=True,  # Only save a model if `monitor=val_loss` has improved.
        monitor='val_disc_loss',
        save_weights_only=True,
        save_freq='epoch',
        verbose=1)

    callbacks4 = tf.keras.callbacks.ModelCheckpoint(
        # Path where to save the model
        filepath="D://Jinfeng/03_training_D2_G1/ckpt/{epoch:02d}_{gen_loss:0.2f}.ckpt",
        save_best_only=True,  # Only save a model if `monitor=val_loss` has improved.
        monitor='val_gen_loss',
        save_weights_only=True,
        save_freq='epoch',
        mode='auto',
        verbose=1)
    
    # log_dir is the path of the folder where you need to store the logs. 
    # To launch the TensorBoard you need to execute the following command:
    # tensorboard --logdir=/full_path_to_your_logs
    callbacks5 = tf.keras.callbacks.TensorBoard(
        log_dir="D://Jinfeng/03_training_D2_G1/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
        # log_dir=os.path.join('D://Jinfeng/05_training', 'logs'),
        # How often to log histogram visualizations, frequency (in epochs). If set to 0, histograms won't be computed.
        # Validation data (or split) must be specified for histogram visualizations.
        # whether to visualize graph in TensorBoard. The log file can be quite large when write_graph is set to True.
        write_graph=True,
        histogram_freq=1,
        # whether to write model weights to visualize as image in TensorBoard.
        write_images=True,
        update_freq='epoch',  # How often to write logs (default: once per epoch)
        )

    # this callback logs the training details in a CSV file.
    # The logged parameters are epoch, accuracy, loss, val_accuracy, and val_loss.
    # pass accuracy as a metric while compiling the model, otherwise will get an execution error.
    # append defines whether or not to append to an existing file, or write in a new file instead.
    callbacks6 = tf.keras.callbacks.CSVLogger(
        filename='M://00_training/train_D2_G1.csv',
        separator=',',
        append=True)

    '''
    callbacks7 = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_gen_loss",
        factor=0.95,
        patience=10,
        verbose=0,
        mode="auto",
        min_delta=0.0001,
        cooldown=0,
        min_lr=0, )'''
    # To limit the execution time, we only train on 100 batches. You can train on
    # the entire dataset. You will need about 20 epochs to get nice results.

'''
    gan.fit(radar_frames_train,
            batch_size=batch_size1, epochs=10000, verbose=2,
            callbacks=[callbacks3, callbacks4, callbacks5, callbacks6],
            validation_data=(radar_frames_val, ), validation_freq=2,
            )'''



    # print(history.history) validation_data=(val_x, val_targets)


if __name__ == "__main__":
    train()
