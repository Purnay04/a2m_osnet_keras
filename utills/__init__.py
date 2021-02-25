from .image_preprocessing import *
from .dataset import *

import tensorflow as tf

def config_keras_backend():
    """Config tensorflow backen to use less GPU memory"""
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(session)
    
def clear_keras_session():
    """Clear keras session.
    """
    tf.keras.backend.clear_session()
    