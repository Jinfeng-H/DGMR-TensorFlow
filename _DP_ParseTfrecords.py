import os
import tensorflow as tf
import numpy as np
import pandas as pd


DATASET_ROOT_DIR = "M://01_dataset"

_FEATURES = {name: tf.io.FixedLenFeature([], dtype)
             for name, dtype in [
                 ('radar', tf.string), ('prob_seq', tf.float32),
                 ('average_seq', tf.float32),
                 ('ul_x_seq', tf.int64), ('ul_y_seq', tf.int64),
                 ('lr_x_seq', tf.int64), ('lr_y_seq', tf.int64),
                 ('end_timestamp_seq', tf.int64),
             ]}

_SHAPE_BY_SPLIT_VARIANT = {
    ('test_steps_be', 'random_crops_256'): (24, 256, 256, 1),
    ("test_parse", "random_crops_256"): (24, 256, 256, 1),
    ("train", "random_crops_256"): (24, 256, 256, 1),
    ("validation", "random_crops_256"): (24, 256, 256, 1),
    ("test", "random_crops_256"): (24, 256, 256, 1),
}

_MM_PER_HOUR_INCREMENT = 1.
_MAX_MM_PER_HOUR = 200.
_INT16_MASK_VALUE = -1


@tf.function
def parse_and_preprocess_row(row, split, variant, data_type):
    # 2005, 2006 data_type=tf.float64
    # 2008, 2009, 2012, 2013 and data from STEPS_BE data_type=tf.float32
    result = tf.io.parse_example(row, _FEATURES)
    shape = _SHAPE_BY_SPLIT_VARIANT[(split, variant)]
    radar_bytes = result.pop("radar")
    radar_int16 = tf.reshape(tf.io.decode_raw(radar_bytes, data_type), shape)
    mask = tf.not_equal(radar_int16, _INT16_MASK_VALUE)
    radar = tf.cast(radar_int16, tf.float32) * _MM_PER_HOUR_INCREMENT
    radar = tf.clip_by_value(
      radar, _INT16_MASK_VALUE * _MM_PER_HOUR_INCREMENT, _MAX_MM_PER_HOUR)
    result["radar_frames"] = radar
    result["radar_mask"] = mask
    return result


@tf.function
def reader(split, variant="random_crops_256", shuffle_files=False):
    # This part is used for parsing tain, validatin.
    if split == "train":
        df_train_2005 = pd.read_csv(
            'D://Jinfeng/v6_modules_get_config/00 making tfrecords and statistics/00_train_sample_30.csv', header=0)
        shard_paths_temp_2005 = df_train_2005.iloc[:, 0].values
        print('Load train finished!')
    elif split == "validation":
        df_val_2005 = pd.read_csv(
            'D://Jinfeng/v6_modules_get_config/00 making tfrecords and statistics/00_validation_sample_30.csv',
            header=0)
        shard_paths_temp_2005 = df_val_2005.iloc[:, 0].values
        print('Load validation finished!')

    shard_paths_2005 = np.array([])
    for i in shard_paths_temp_2005:
        temp_path_2005 = os.path.join(DATASET_ROOT_DIR, split, variant, i)
        shard_paths_2005 = np.append(shard_paths_2005, temp_path_2005)

    print(f'len(shard_paths_2005) is {len(shard_paths_2005)}')
    shards_dataset_2005 = tf.data.Dataset.from_tensor_slices(shard_paths_2005)

    if shuffle_files:
        shards_dataset_2005 = shards_dataset_2005.shuffle(buffer_size=len(shard_paths_2005))

    dataset_2005 = shards_dataset_2005.interleave(lambda x: tf.data.TFRecordDataset(x, compression_type=""),
                                                  num_parallel_calls=tf.data.AUTOTUNE, deterministic=not shuffle_files)\
        .map(lambda row: parse_and_preprocess_row(row, split, variant, data_type=tf.float64), num_parallel_calls=tf.data.AUTOTUNE)
    return dataset_2005


def reader_test(split, variant="random_crops_256", shuffle_files=False):
    shard_paths_2013=np.array([])
    if split == "test":
        #df_test_2013 = pd.read_csv(
        #    'D://Jinfeng/v6_modules_get_config/00 making tfrecords and statistics/00_test_a_flood.csv',
        #    header=0)
        df_test_2013 = pd.read_csv('D://Jinfeng/v6_modules_get_config/00 making tfrecords and statistics/00_test_steps_be.csv',
            header=0)
        shard_paths_temp_2013 = df_test_2013.iloc[:, 0].values
        for m in shard_paths_temp_2013:
            temp_path_2013 = os.path.join(DATASET_ROOT_DIR, split, variant, m)
            shard_paths_2013 = np.append(shard_paths_2013, temp_path_2013)
        print(f'Load test finished! lenhth is {len(shard_paths_2013)}')
    elif split == "test_parse":
        df_test_2013 = pd.read_csv(
            'D://Jinfeng/v6_modules_get_config/00 making tfrecords and statistics/00_test_2006.csv', header=0)
        shard_paths_temp_2013 = df_test_2013.iloc[:, 0].values
        for m in shard_paths_temp_2013:
            temp_path_2013 = os.path.join(DATASET_ROOT_DIR, split, variant, m)
            shard_paths_2013 = np.append(shard_paths_2013, temp_path_2013)
        print('Load test finished!')
    shards_dataset_2013 = tf.data.Dataset.from_tensor_slices(shard_paths_2013)
    if shuffle_files:
        shards_dataset_2013 = shards_dataset_2013.shuffle(buffer_size=len(shard_paths_2013))
    dataset_2013 = shards_dataset_2013.interleave(lambda x: tf.data.TFRecordDataset(x, compression_type=""),
                                                  num_parallel_calls=tf.data.AUTOTUNE, deterministic=not shuffle_files)\
        .map(lambda row: parse_and_preprocess_row(row, split, variant, data_type=tf.float32), num_parallel_calls=tf.data.AUTOTUNE)
    # dataset_all = tf.concat([dataset_2005, dataset_2012], 0)
    return dataset_2013


@tf.function
def reader_test_parse(split, variant="random_crops_256", shuffle_files=False):
    # This part is used for test_parse_tfrecords_right.py
    # configurate the path of the dataset, 路径下面所有"*.tfrecord"的文件
    shards_glob = os.path.join(DATASET_ROOT_DIR, split, variant, "*.tfrecord")
    # 'tf.io.gfile.glob' Returns a list of files' names that match the given pattern in the bracket,
    # which is  "*.tfrecord.gz".
    shard_paths = tf.io.gfile.glob(shards_glob)
    print(f'len(shard_paths) is {len(shard_paths)}')
    shards_dataset = tf.data.Dataset.from_tensor_slices(shard_paths)

    if shuffle_files:
        shards_dataset = shards_dataset.shuffle(buffer_size=len(shard_paths))
    return (shards_dataset
            .interleave(lambda x: tf.data.TFRecordDataset(x, compression_type=""),
                        num_parallel_calls=tf.data.AUTOTUNE,
                        deterministic=not shuffle_files)
            .map(lambda row: parse_and_preprocess_row(row, split, variant, data_type=tf.float32),
                 num_parallel_calls=tf.data.AUTOTUNE))

