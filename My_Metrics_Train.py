import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt



@tf.function
def mask_fun(y_true_crop, y_pred_crop, t1, t2, crop_non_nan):
    if t1 is None and t2 is not None:
        mask_t = tf.where(y_true_crop < t2, 1., 0.)
        mask_p = tf.where(y_pred_crop < t2, 1., 0.)
    elif t1 is not None and t2 is None:
        mask_t = tf.where(y_true_crop >= t1, 1., 0.)
        mask_p = tf.where(y_pred_crop >= t1, 1., 0.)
    elif t1 is not None and t2 is not None:
        mask_t = tf.where(y_true_crop >= t1, y_true_crop, 0.)
        mask_t = tf.where(mask_t < t2, 1., 0.)
        mask_p = tf.where(y_pred_crop >= t1, y_pred_crop, 0.)
        mask_p = tf.where(mask_p < t2, 1., 0.)
    else:
        mask_t = tf.ones_like(y_true_crop, dtype=tf.float32)
        mask_p = tf.ones_like(y_pred_crop, dtype=tf.float32)
    mask_t = tf.math.multiply(mask_t, crop_non_nan)
    mask_p = tf.math.multiply(mask_p, crop_non_nan)
    tp = tf.math.logical_and(tf.math.equal(mask_t, 1.), tf.math.equal(mask_p, 1.))
    tp = tf.cast(tp, tf.int32)
    fp = tf.math.logical_and(tf.math.equal(mask_t, 0.), tf.math.equal(mask_p, 1.))
    fp = tf.cast(fp, tf.int32)
    fn = tf.math.logical_and(tf.math.equal(mask_t, 1.), tf.math.equal(mask_p, 0.))
    fn = tf.cast(fn, tf.int32)
    return mask_t, mask_p, tp, fp, fn


@tf.function
def mse_metric(y_true_crop, y_pred_crop, t1, t2, crop_non_nan, sample_weight=None):
    # calculation of the metric
    mask_t, _, _, _, _ = mask_fun(y_true_crop, y_pred_crop, t1, t2, crop_non_nan)
    mask_t_batch = tf.math.reduce_sum(mask_t, axis=[1, 2, 3, 4])
    y_true_crop = tf.math.multiply(y_true_crop, mask_t)
    y_pred_crop = tf.math.multiply(y_pred_crop, mask_t)
    mse_crop = tf.math.squared_difference(y_true_crop, y_pred_crop)
    mse_crop_batch = tf.math.reduce_sum(mse_crop, axis=[1, 2, 3, 4])
    mse_batch = tf.math.divide_no_nan(mse_crop_batch, mask_t_batch)
    if sample_weight is not None:
        # get the sample weights with the sum
        sample_weight = tf.cast(sample_weight, dtype=tf.float32)
        mse_batch = tf.math.multiply(mse_batch, sample_weight)
    mse = tf.math.reduce_mean(mse_batch)
    return mse


@tf.function
def pcc_ori_metric(y_true_crop, y_pred_crop, crop_non_nan, sample_weight=None):
    y_mask_nan_batch = tf.math.reduce_sum(crop_non_nan, axis=[1, 2, 3, 4])
    y_true_mean = tf.math.reduce_mean(y_true_crop, axis=[1, 2, 3, 4], keepdims=True)
    y_true_std = tf.math.reduce_std(y_true_crop, axis=[1, 2, 3, 4], keepdims=True)
    y_pred_mean = tf.math.reduce_mean(y_pred_crop, axis=[1, 2, 3, 4], keepdims=True)
    y_pred_std = tf.math.reduce_std(y_pred_crop, axis=[1, 2, 3, 4], keepdims=True)
    y_true_new_temp = tf.math.add(y_true_crop, -y_true_mean)
    y_pred_new_temp = tf.math.add(y_pred_crop, -y_pred_mean)
    y_true_new = tf.math.divide_no_nan(y_true_new_temp, y_true_std)
    y_pred_new = tf.math.divide_no_nan(y_pred_new_temp, y_pred_std)
    pcc_matrix_temp = tf.math.multiply(y_true_new, y_pred_new)
    pcc_matrix = tf.math.multiply(pcc_matrix_temp, crop_non_nan)
    pcc_matrix_temp_batch = tf.math.reduce_sum(pcc_matrix, axis=[1, 2, 3, 4])
    pcc_matrix_batch = tf.math.divide_no_nan(pcc_matrix_temp_batch, y_mask_nan_batch)
    if sample_weight is not None:
        sample_weight = tf.cast(sample_weight, dtype=tf.float32)
        pcc_matrix_batch = tf.math.multiply(pcc_matrix_batch, sample_weight)
    pcc = tf.math.reduce_mean(pcc_matrix_batch)
    return pcc

@tf.function
def pcc_metric(y_true_crop, y_pred_crop, t1, t2, crop_non_nan, sample_weight=None):
    mask_t, _, _, _, _ = mask_fun(y_true_crop, y_pred_crop, t1, t2, crop_non_nan)
    mask_t_batch_dim = tf.math.reduce_sum(mask_t, axis=[1, 2, 3, 4], keepdims=True)
    y_true_crop = tf.math.multiply(y_true_crop, mask_t)
    y_pred_crop = tf.math.multiply(y_pred_crop, mask_t)
    y_true_mean_batch_dim = tf.math.divide_no_nan(
        tf.math.reduce_sum(y_true_crop, axis=[1, 2, 3, 4], keepdims=True), mask_t_batch_dim)
    y_pred_mean_batch_dim = tf.math.divide_no_nan(
        tf.math.reduce_sum(y_pred_crop, axis=[1, 2, 3, 4], keepdims=True), mask_t_batch_dim)
    y_true_std_batch_dim = tf.math.sqrt(tf.math.divide_no_nan(
        tf.math.reduce_sum(
            tf.math.multiply(tf.math.squared_difference(y_true_crop, y_true_mean_batch_dim), mask_t),
            axis=[1, 2, 3, 4], keepdims=True)
        , mask_t_batch_dim))
    y_pred_std_batch_dim = tf.math.sqrt(tf.math.divide_no_nan(
        tf.math.reduce_sum(
            tf.math.multiply(tf.math.squared_difference(y_pred_crop, y_pred_mean_batch_dim), mask_t),
            axis=[1, 2, 3, 4], keepdims=True)
        , mask_t_batch_dim))
    pcc_crop = tf.math.multiply(
        tf.math.divide_no_nan(tf.math.subtract(y_true_crop, y_true_mean_batch_dim), y_true_std_batch_dim),
        tf.math.divide_no_nan(tf.math.subtract(y_pred_crop, y_pred_mean_batch_dim), y_pred_std_batch_dim),
    )
    pcc_crop = tf.math.multiply(pcc_crop, mask_t)
    pcc_batch_dim = tf.math.divide_no_nan(
        tf.math.reduce_sum(pcc_crop, axis=[1, 2, 3, 4], keepdims=True), mask_t_batch_dim)
    if sample_weight is not None:
        sample_weight = tf.cast(sample_weight, dtype=tf.float32)
        pcc_batch_dim = tf.math.multiply(pcc_batch_dim, sample_weight)
    pcc = tf.math.reduce_mean(pcc_batch_dim)
    return pcc


@tf.function
def nse_metric(y_true_crop, y_pred_crop, crop_non_nan, sample_weight=None):
    crop_non_nan_batch = tf.math.reduce_sum(crop_non_nan, axis=[1, 2, 3, 4])
    y_true_crop = tf.math.multiply(y_true_crop, crop_non_nan)
    y_pred_crop = tf.math.multiply(y_pred_crop, crop_non_nan)
    y_true_mean_time = tf.math.reduce_mean(y_true_crop, axis=1, keepdims=True)
    true_pred = tf.math.squared_difference(y_true_crop, y_pred_crop)
    true_mean = tf.math.squared_difference(y_true_crop, y_true_mean_time)
    true_mean = tf.math.multiply(true_mean, crop_non_nan)
    true_pred = tf.math.reduce_sum(true_pred, axis=1, keepdims=True)
    true_mean = tf.math.reduce_sum(true_mean, axis=1, keepdims=True)
    nse_crop = tf.math.subtract(1., tf.math.divide_no_nan(true_pred, true_mean))
    nse_crop = tf.math.multiply(nse_crop, crop_non_nan)
    nse_batch = tf.math.divide_no_nan(
        tf.math.reduce_sum(nse_crop, axis=[1, 2, 3, 4]), crop_non_nan_batch)
    if sample_weight is not None:
        sample_weight = tf.cast(sample_weight, dtype=tf.float32)
        nse_batch = tf.math.multiply(nse_batch, sample_weight)
    nse = tf.math.reduce_mean(nse_batch)
    return nse


@tf.function
def nse_ori_metric(y_true_crop, y_pred_crop, crop_non_nan, sample_weight=None):
    y_true_mean = tf.math.reduce_mean(y_true_crop, axis=[1, 2, 3, 4], keepdims=True)
    y_square_difference = tf.math.squared_difference(y_true_crop, y_pred_crop)
    y_square_difference = tf.math.multiply(y_square_difference, crop_non_nan)
    y_square_difference_mean = tf.math.reduce_mean(y_square_difference, axis=[1, 2, 3, 4])
    y_true_square_difference = tf.math.squared_difference(y_true_crop, y_true_mean)
    y_true_square_difference = tf.math.multiply(y_true_square_difference, crop_non_nan)
    y_true_square_difference_mean = tf.math.reduce_mean(y_true_square_difference, axis=[1, 2, 3, 4])
    nse_matrix_temp = tf.math.divide_no_nan(y_square_difference_mean, y_true_square_difference_mean)
    nse_matrix_batch = tf.math.add(tf.constant(1.0), -nse_matrix_temp)
    if sample_weight is not None:
        sample_weight = tf.cast(sample_weight, dtype=tf.float32)
        nse_matrix_batch = tf.math.multiply(nse_matrix_batch, sample_weight)
    nse_ori = tf.math.reduce_mean(nse_matrix_batch)
    return  nse_ori


def csi_precison_recall_metric(y_true_crop, y_pred_crop, t1, t2, crop_non_nan):
    _, _, tp, fp, fn = mask_fun(y_true_crop, y_pred_crop, t1, t2, crop_non_nan)
    tp_sum = tf.math.reduce_sum(tp, axis=[1, 2, 3, 4])
    fp_sum = tf.math.reduce_sum(fp, axis=[1, 2, 3, 4])
    fn_sum = tf.math.reduce_sum(fn, axis=[1, 2, 3, 4])
    tp_fp_fn = tf.math.add_n([tp_sum, fp_sum, fn_sum])
    tp_fp_fn = tf.cast(tp_fp_fn, dtype=tf.float32)
    tp_fn = tf.math.add(tp_sum, fn_sum)
    tp_fn = tf.cast(tp_fn, dtype=tf.float32)
    tp_fp = tf.math.add(tp_sum, fp_sum)
    tp_fp = tf.cast(tp_fp, dtype=tf.float32)
    tp_sum = tf.cast(tp_sum, dtype=tf.float32)
    csi = tf.math.divide_no_nan(tp_sum, tp_fp_fn)
    csi = tf.cast(csi, dtype=tf.float32)
    csi = tf.math.reduce_mean(csi)
    recall = tf.math.divide_no_nan(tp_sum, tp_fn)
    recall = tf.cast(recall, dtype=tf.float32)
    recall = tf.math.reduce_mean(recall)
    precision = tf.math.divide_no_nan(tp_sum, tp_fp)
    precision = tf.cast(precision, dtype=tf.float32)
    precision = tf.math.reduce_mean(precision)
    return csi, recall, precision

