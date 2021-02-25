from get_model import get_model

from absl import app, logging, flags
from absl.flags import FLAGS
import tensorflow as tf

import os 
import random
import numpy as np
from utills import config_keras_backend, clear_keras_session
flags.DEFINE_string("image_dir","./Market-1501-v15.09.15/training","path to the image file")

def main(_argv):
    #32668
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)
        
    model = get_model()
    model.load_weights('./checkpoints/2021-02-18/1613636421.5762243/osnet-ckpt-050.h5')
    
    if os.path.exists(FLAGS.image_dir):
        print("exists")
        images = [i for i in os.listdir(FLAGS.image_dir) if int(i.split("_")[0]) not in  [0, -1]]
    idx_image = random.randint(0, len(images))
    img_path = os.path.join(FLAGS.image_dir, images[idx_image])
    corr_label = int(images[idx_image].split("_")[0])
    
    with tf.io.gfile.GFile(img_path, 'rb') as f:
        image_data = f.read()
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = np.expand_dims(image, 0)
    
    res = model(image)
    print("Predicted:",np.argmax(res),"Actual:", corr_label)
if __name__ == "__main__":
    app.run(main)