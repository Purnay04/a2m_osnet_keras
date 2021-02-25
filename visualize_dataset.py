from absl import app,logging,flags
from absl.flags import FLAGS

import cv2
import tensorflow as tf
import os
import numpy as np

flags.DEFINE_string('shards_dir','./Market-1501-v15.09.15/shards', 'Path to the shards of dataset')
flags.DEFINE_boolean('is_val', False, 'specify this if you want to visualize validation dataset')

def _parse_fn(example_serialized):
    feature_map = {
        'image/encoded': tf.io.FixedLenFeature([], dtype=tf.string,
                                           default_value=''),
        'image/class/label': tf.io.FixedLenFeature([], dtype= tf.int64,
                                                default_value=-1),
        'image/class/text': tf.io.FixedLenFeature([], dtype=tf.string,
                                               default_value=''),
    }
    parsed = tf.io.parse_single_example(example_serialized, feature_map)
    return parsed

def main(_argv):
    if FLAGS.is_val:
        path = os.path.join(FLAGS.shards_dir, "validation-*")
       
    else:
        path = os.path.join(FLAGS.shards_dir, "train-*")   

    files = tf.io.matching_files(path)
    shards = tf.data.Dataset.from_tensor_slices(files)
    shards = shards.shuffle(tf.cast(tf.shape(files)[0], tf.int64))
    shards = shards.repeat()
    dataset = shards.interleave(tf.data.TFRecordDataset, cycle_length = 4)
    dataset = dataset.shuffle(buffer_size = 8192)
    parsed_dataset = dataset.map(_parse_fn)
    
    for data in parsed_dataset.take(1):
        print(data['image/class/label'].numpy())
        print(data['image/class/text'].numpy())
        #print(data['image/encoded'].numpy())
        image = cv2.imdecode(np.frombuffer(data['image/encoded'].numpy(), np.uint8), -1)
        cv2.imshow('img', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
   
if __name__ == '__main__':
    app.run(main)