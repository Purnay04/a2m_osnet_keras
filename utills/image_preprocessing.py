import math
import random 

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.python.ops import control_flow_ops

def _smallest_size_at_least(height, width, smallest_side):
    smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

    height = tf.cast(height, tf.float32)
    width = tf.cast(width, tf.float32)
    smallest_side = tf.cast(smallest_side, tf.float32)
    
    scale = tf.cond(tf.greater(height, width),
                    lambda: smallest_side / width,
                    lambda: smallest_side / height)
    new_height = tf.cast(tf.math.rint(height * scale), tf.int32)
    new_width = tf.cast(tf.math.rint(width * scale), tf.int32)
    return new_height, new_width

def _aspect_preserving_resize(image, smallest_side):
    smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)
    
    shape = tf.shape(image)
    height, width = tf.shape(image)[0], tf.shape(image)[1]
    new_height, new_width = _smallest_size_at_least(height, width, smallest_side) 
    image = tf.expand_dims(image, 0)
    resized_image = tf.compat.v1.image.resize_bilinear(image, [new_height, new_width],
                                                       align_corners=False)
    resized_image = tf.squeeze(resized_image)
    resized_image.set_shape([None, None, 3])
    return resized_image


def _crop(image, offset_height, offset_width, crop_height, crop_width):
    original_shape = tf.shape(image)
    
    rank_assertion = tf.Assert(
        tf.equal(tf.rank(image), 3),
        ['Rank of image must be equal to 3.'])
    
    with tf.control_dependencies([rank_assertion]):
        cropped_shape = tf.stack([crop_height, crop_width, original_shape[2]])
        
    size_assertion = tf.Assert(
        tf.logical_and(
            tf.greater_equal(original_shape[0], crop_height),
            tf.greater_equal(original_shape[1], crop_width)),
            ['crop size greater than the image size.'])
    
    offsets = tf.cast(tf.stack([offset_height, offset_width, 0]), tf.int32)
    
    # Use tf.slice instead of crop_to_bounding box as it accepts tensors to
    # define the crop size.
    with tf.control_dependencies([size_assertion]):
        image = tf.slice(image, offsets, cropped_shape)
    return tf.reshape(image, cropped_shape)

def _central_crop(image_list, crop_height, crop_width):
    outputs = []
    for image in image_list:
        image_height = tf.shape(image)[0]
        image_width = tf.shape(image)[1]
        
        offset_height = (image_height - crop_height) / 2
        offset_width = (image_width - crop_width) / 2
        
        outputs.append(_crop(image, offset_height, offset_width, crop_height, crop_width))
        
        return outputs


# need to learn about control_flow
def apply_with_random_selector(x, func, num_cases):
    sel = tf.random.uniform([], maxval=num_cases, dtype=tf.int32)
    # Pass the real x only to one of the func calls
    return control_flow_ops.merge([
        func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case)
        for case in range(num_cases)])[0]
    
def distort_color(image, color_ordering=0, fast_mode=True, scope=None):
    with tf.name_scope('distort_color') as scope:
        if fast_mode:
            if color_ordering == 0:
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            else:
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                
        else:
            if color_ordering == 0:
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            
            elif color_ordering == 1:
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
                
            elif color_ordering == 2:
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                
            elif color_ordering == 3:
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                
            else:
                raise ValueError('color_ordering must be in [0, 3]')
                
        return tf.clip_by_value(image, 0.0, 1.0)
    
def distorted_bounding_box_crop(image,
                                bbox,
                                min_object_covered = 0.1,
                                aspect_ratio_range = (0.75, 1.33),
                                area_range = (0.05, 1.0),
                                max_attempts = 100,
                                scope = None):
    with tf.name_scope('distorted_bounding_box_crop') as scope:
        sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
            tf.shape(image),
            bounding_boxes = bbox,
            min_object_covered = min_object_covered,
            aspect_ratio_range = aspect_ratio_range,
            area_range = area_range,
            max_attempts = max_attempts,
            use_image_if_no_bounding_boxes=True)
        bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box
        
        #
        cropped_image = tf.slice(image,bbox_begin, bbox_size)
        return cropped_image, distort_bbox
    
def resize_and_rescale_image(image, height, width, do_mean_subtraction=True, scope=None):
    with tf.name_scope('resize_image') as scope:
        image = tf.expand_dims(image, 0)
        image = tf.compat.v1.image.resize_bilinear(image, [height, width], align_corners=False)
        
        image = tf.squeeze(image, [0])
        if do_mean_subtraction:
            image = tf.subtract(image, 0.5)
            image = tf.multiply(image, 2.0)
            
    return image

def preprocess_for_train(image,
                         height,
                         width,
                         bbox,
                         max_angle=15.,
                         fast_mode = True,
                         scope = None,
                         add_image_summaries = False):
    with tf.name_scope('distort_image') as scope:
        assert image.dtype == tf.float32
        #
        angle = random.uniform(-max_angle, max_angle) if random.random() < 0.75 else 0.
        
        rotated_image = tfa.image.rotate(image, math.radians(0), interpolation='BILINEAR')
        # random cropping
        distorted_image, distorted_bbox = distorted_bounding_box_crop(rotated_image,
                                                                      bbox,
                                                                      min_object_covered = 0.6,
                                                                      area_range = (0.6, 1.0))
        distorted_image.set_shape([None, None, 3])
        num_resize_cases = 1 if fast_mode else 4
        distorted_image = apply_with_random_selector(
            distorted_image,
            lambda x, method: tf.compat.v1.image.resize(x, [height, width], method),
            num_cases = num_resize_cases)
        
        distorted_image = tf.image.random_flip_left_right(distorted_image)
        
        num_distort_cases = 1 if fast_mode else 4
        distorted_image = apply_with_random_selector(
            distorted_image,
            lambda x, ordering: distort_color(x, ordering, fast_mode),
            num_cases = num_distort_cases)
        
        distorted_image = tf.subtract(distorted_image, 0.5)
        distorted_image = tf.multiply(distorted_image, 2.0)
        return distorted_image
    
def preprocess_for_eval(image,
                        height,
                        width,
                        scope = None,
                        add_image_summaries = False):
    with tf.name_scope('eval_image') as scope:
        assert image.dtype == tf.float32
        
        image = _aspect_preserving_resize(image, max(height, width))
        image = _central_crop([image], height, width)[0]
        image.set_shape([height, width, 3])
        
        image = tf.subtract(image, 0.5)
        image = tf.multiply(image, 2.0)
        return image
    
def preprocess_image(image,
                     height,
                     width,
                     is_training=False):
    if is_training:
        bbox = tf.constant([0.0, 0.0, 1.0, 1.0],
                           dtype = tf.float32,
                           shape = [1, 1, 4])
        return preprocess_for_train(image, height, width, bbox, fast_mode=True)
    else:
        return preprocess_for_eval(image, height, width)

    
    