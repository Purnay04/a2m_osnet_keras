from datetime import datetime
import os
import random
import sys
import threading
import six
from tqdm import tqdm

import numpy as np
import tensorflow as tf

from absl import flags, logging, app
from absl.flags import FLAGS


# Training Images: 28841
# Testing Images: 15905
flags.DEFINE_integer('train_shards', 32, 'Enter number of trining shards')
flags.DEFINE_integer('val_shards', 32,'Enter number of validation shards')
flags.DEFINE_string('train_dataset','./Market-1501-v15.09.15/training', 'Enter trian dataset path')
flags.DEFINE_string('val_dataset','./Market-1501-v15.09.15/bounding_box_test', 'Enter test dataset path')
flags.DEFINE_string('classes', './data/classes.txt', 'Enter classes file path')
flags.DEFINE_string('output_path','./Market-1501-v15.09.15/shards', 'Enter Output path to store shards')
flags.DEFINE_integer('NUM_THREADS', 4, 'Enter the thread count')
#flags.DEFINE_integer('num_classes', 1501, 'specify number of classes')

#NUM_THREADS = 4 # Number of threads to preprocess the images.

def _int64_feature(value):
    """wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _float_feature(value):
    """wrapper for inserting float features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list = tf.train.FloatList(value=value))

def _bytes_feature(value):
    """ wrapper for inserting bytes features into Example proto. """
    if six.PY3 and isinstance(value, six.text_type):
        value = six.binary_type(value, encoding='utf-8')
    return tf.train.Feature(bytes_list = tf.train.BytesList(value=[value]))


def _convert_to_example(filename,
                        image_buffer,
                        label,
                        class_name,
                        height,
                        width):
    """Build an Example proto for an example.

    Args:
        filename (string): path to an image file
        image_buffer (string): JPEG encoding of RGB image
        label (integer): identifier for the ground truth for the network
        class_name (string): human redable label
        height (integer): image height in pixels
        width (integer): image width in pixels

    Returns:
        [Example]: example proto of Example class.
    """ 
    example = tf.train.Example(features = tf.train.Features(feature={
        'image/height': _int64_feature(height),
        'image/width': _int64_feature(width),
        'image/class/label': _int64_feature(label),
        'image/class/text': _bytes_feature(class_name),
        'image/filename': _bytes_feature(os.path.basename(filename)),
        'image/encoded': _bytes_feature(image_buffer)}))
    return example

class ImageCoder(object):
    """Helper class that provides Tensorflow image coding utilities."""
    
    #def __init__(self):
        # create a single session to run all image coding calls.
        #self._sess = tf.compat.v1.Session()
        
        # Initializes function that decodes RGB JPEG data.
        #self._decode_jpeg_data = tf.compat.v1.placeholder(dtype=tf.string)
        #self.decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)
        
    def decode_jpeg(self, image_data):
        image = tf.image.decode_jpeg(image_data, channels=3)
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image

def _process_image(filename, coder):
    # Read the image file
    with tf.io.gfile.GFile(filename, 'rb') as f:
        image_data = f.read()
    
    # Decode the RGB JPEG.
    image = coder.decode_jpeg(image_data)
    
    # Check that image converted to RGB
    assert len(image.shape) == 3
    height = image.shape[0]
    width = image.shape[1]
    assert image.shape[2] == 3
    
    return image_data, height, width

def _process_image_files_batch(coder, thread_index, ranges, name,
                               filenames, labels, classes, num_shards):
    # Each thread produces N shards where N = int(num_shards / num_threads).
    # For instance, if num_shards = 32, and the num_threads = 4. then the first
    # thread would produce shards [0, 8).
    
    num_threads = len(ranges)
    assert not num_shards % num_threads
    num_shards_per_batch = int(num_shards / num_threads)
    
    shards_ranges = np.linspace(ranges[thread_index][0],
                                ranges[thread_index][1],
                                num_shards_per_batch + 1).astype(int)
    num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]
    
    counter = 0
    for s in range(num_shards_per_batch):
        # Generate a sharded version of the file name, e.g. 'train-0002-of-00010'
        shard = thread_index * num_shards_per_batch + s
        output_filename = "%s-%.5d-of-%.5d" % (name, shard, num_shards)
        output_file = os.path.join(FLAGS.output_path, output_filename)
        writer = tf.io.TFRecordWriter(output_file)
        
        shard_counter = 0
        files_in_shard = np.arange(shards_ranges[s], shards_ranges[s + 1], dtype=int)
        for i in files_in_shard:
            filename = filenames[i]
            label = labels[i]
            class_name = classes[label]
            
            image_buffer, height, width = _process_image(filename, coder)
            example = _convert_to_example(filename, image_buffer, label,
                                          class_name, height, width)
            writer.write(example.SerializeToString())
            shard_counter += 1
            counter += 1
            
            if not counter % 1000:
                logging.info("%s [thread %d]: Processed %d of %d images in thread batch." %
                             (datetime.now(), thread_index, counter, num_files_in_thread))
                sys.stdout.flush()
        writer.close()
        logging.info("%s [thread %d]: Wrote %d images to %s" %
                     (datetime.now(), thread_index, shard_counter, output_file))
        sys.stdout.flush()
        shard_counter = 0
    logging.info("%s [thread %d]: Wrote %d images to %d shards." %
                 (datetime.now(), thread_index, counter, num_files_in_thread))
    sys.stdout.flush()
        
            
        
        
def _process_image_files(name, filenames, labels, classes,
                         num_shards):
    assert len(filenames) == len(labels)
    
    # Break all images into batches with a [ranges[i][0], ranges[i][1]].
    spacing = np.linspace(0, len(filenames), FLAGS.NUM_THREADS + 1).astype(np.int)
    ranges = []
    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])
    
    # Launch a thread for each batch.
    print('Launching {} threads for spacing: {}'.format(FLAGS.NUM_THREADS, ranges))
    sys.stdout.flush()
    
    # Create a mechanism for monitoring when all threads are finished.
    coord = tf.train.Coordinator()
    
    # Create a generic Tensorflow-based utility for converting all image codings.
    coder = ImageCoder()
    
    threads = []
    for thread_index in range(len(ranges)):
        args = (coder, thread_index, ranges, name,
                filenames, labels, classes, num_shards)
        t = threading.Thread(target=_process_image_files_batch, args=args)
        t.start()
        threads.append(t)
    
    # Wait for all the threads to terminate.
    coord.join(threads)
    logging.info("{}: Finished writing all {} images in data set.".format(datetime.now(), len(filenames)))
    sys.stdout.flush()
    
def _find_image_files(directory, classes): 
    labels = []
    filenames = []
    matched_class_cnt = 0
    # Label index 0 is for background class.
    label_index = 0
    
    for class_name in tqdm(classes, "classes wise extracting:"):
        class_file_path = directory+"/{}_*_*_*.jpg".format(class_name.zfill(4))
        #print(class_file_path)
        matching_files = tf.io.gfile.glob(class_file_path)

        if matching_files != []:
            labels.extend([int(class_name)] * len(matching_files))
            filenames.extend(matching_files)    
            matched_class_cnt += 1
            #logging.info("Finished finding files with {} matched of {} class".format(len(matching_files), class_name))
            
    #print(len(filenames))
    shuffled_index = list(range(len(filenames)))
    random.seed(12345)
    random.shuffle(shuffled_index)
    
    filenames = [filenames[i] for i in shuffled_index]
    labels = [labels[i] for i in shuffled_index]
    
    logging.info("Finished finding files with {} matched  of {} out of {} class in {} directory".format(len(labels), matched_class_cnt, len(classes), directory))
    
    return filenames, labels

def _process_dataset(name, directory, num_shards, classes):
    filesnames, labels = _find_image_files(directory, classes)
    _process_image_files(name, filesnames, labels, classes, num_shards)

def build_class_lookup(class_text_file):
    classes = [i.strip() for i in tf.io.gfile.GFile(class_text_file, 'r').readlines()]
    return classes
    
def main(_argv):
    assert not FLAGS.train_shards % FLAGS.NUM_THREADS, (
        'Please make the NUM_THREADS commensurate with FLAGS.train_shards')
    assert not FLAGS.val_shards % FLAGS.NUM_THREADS, (
        'Please make the NUM_THREADS commensurate with '
        'FLAGS.val_shards')
    logging.info('Saving result to %s' % FLAGS.output_path)
    
    # Build a class lookup
    classes = build_class_lookup(FLAGS.classes)
    
    _process_dataset('validation', FLAGS.val_dataset, FLAGS.val_shards, classes)
    _process_dataset('train', FLAGS.train_dataset, FLAGS.train_shards, classes)

if __name__ == "__main__":
    app.run(main)