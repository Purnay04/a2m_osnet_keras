import os
from functools import partial
from absl import flags
from absl.flags import FLAGS

import tensorflow as tf

from .image_preprocessing import preprocess_image, resize_and_rescale_image

IMG_HEIGHT = 128
IMG_WIDTH = 64


def decode_jpeg(image_buffer, scope=None):
    with tf.name_scope('decode_jpeg') as scope:
        image = tf.image.decode_jpeg(image_buffer, channels=3)
        image = tf.image.convert_image_dtype(image, dtype = tf.float32)
        
        return image
    
def _parse_fn(example_serialized, is_training):
    feature_map = {
        'image/encoded': tf.io.FixedLenFeature([], dtype=tf.string,
                                           default_value=''),
        'image/class/label': tf.io.FixedLenFeature([], dtype= tf.int64,
                                                default_value=-1),
        'image/class/text': tf.io.FixedLenFeature([], dtype=tf.string,
                                               default_value=''),
    }       
    parsed = tf.io.parse_single_example(example_serialized, feature_map)
    image = decode_jpeg(parsed['image/encoded'])
    if FLAGS.data_aug:
        image = preprocess_image(image, IMG_HEIGHT, IMG_WIDTH, is_training=is_training)
    else:
        image = resize_and_rescale_image(image, IMG_HEIGHT, IMG_WIDTH)
    label = tf.one_hot(parsed['image/class/label'] - 1, 1501, dtype=tf.float32)
    return (image, label)

def get_dataset(tfrecords_dir, subset, batch_size):
    files = tf.io.matching_files(os.path.join(tfrecords_dir, "%s-*" % subset))
    shards = tf.data.Dataset.from_tensor_slices(files)
    shards = shards.shuffle(tf.cast(tf.shape(files)[0], tf.int64)) 
    shards = shards.repeat()
    dataset = shards.interleave(tf.data.TFRecordDataset, cycle_length=4)
    dataset = dataset.shuffle(buffer_size = 8192)
    parser = partial(
        _parse_fn, is_training=True if subset == 'train' else False)
    dataset = dataset.apply(
        tf.data.experimental.map_and_batch(
            map_func = parser,
            batch_size = batch_size,
            num_parallel_calls= FLAGS.num_data_workers))
    dataset = dataset.prefetch(batch_size)
    return dataset               