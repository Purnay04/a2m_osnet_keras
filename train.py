import os
import time
import argparse
from datetime import datetime
from utills.dataset import get_dataset

from absl import app, logging, flags
from absl.flags import FLAGS

import tensorflow as tf

from utills import config_keras_backend, clear_keras_session
from models.models import get_optimizer, get_training_model, get_lr_func

#NUM_DATA_WORKERS = 8

NOW = str(datetime.date(datetime.now()))
SAVE_DIR = os.path.join("./checkpoints", NOW, str(time.time()))
LOG_DIR = os.path.join("./logs", NOW)

flags.DEFINE_string('dataset_dir', './Market-1501-v15.09.15/shards', 'Specify the path of tfrecord shards dataset')
flags.DEFINE_integer('num_classes', 1501, 'specify number of classes')
flags.DEFINE_float('dropout_rate', 0.3, 'specify the dropout rate')
flags.DEFINE_string('optimizer', 'adam', 'specify the optimizer')
flags.DEFINE_float('epsilon', 1e-3, 'Specify the epsilon')
flags.DEFINE_bool('label_smoothing', False, 'Specify True if you want label_smoothing')
flags.DEFINE_bool('data_aug', False, 'Specify True if you want DATA_AUGMENTATION')
flags.DEFINE_integer('batch_size', 16, 'specify batch size')
flags.DEFINE_enum('lr_sched', 'linear', ['linear', 'exp'],
                'linear : linear Way'
                'exp : Exponetial Way')
flags.DEFINE_float('initial_lr', 1e-2, 'Specify initial learning rate')
flags.DEFINE_float('final_lr', 1e-5, 'Specify final learning rate')
flags.DEFINE_float('weight_decay', 1e-4, 'Specify weight decay rate')
flags.DEFINE_integer('epochs', 50, 'specify epochs')
flags.DEFINE_string('model', 'osnet', 'specify model')
flags.DEFINE_integer('num_data_workers', 8, 'specify number of workers')
    
def train(model_name, dropout_rate, optim_name, epsilon,
          label_smoothing, batch_size, lr_sched, initial_lr,
          final_lr, weight_decay, epochs, dataset_dir):
    
    """ Prepare data and train the model."""
    batch_size = batch_size
    initial_lr = initial_lr
    final_lr = final_lr
    optimizer = get_optimizer(model_name, optim_name, initial_lr, epsilon)
    weight_decay = weight_decay
    
    # get training and validation data
    ds_train = get_dataset(dataset_dir, 'train', batch_size)
    ds_val = get_dataset(dataset_dir, 'validation', batch_size)
    
    lrate = get_lr_func(epochs, lr_sched, initial_lr, final_lr)
    save_name = model_name if not model_name.endswith('.h5') else \
        os.path.split(model_name)[-1].split('.')[0].split('-')[0]
    model_ckpt = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(SAVE_DIR,save_name) + '-ckpt-{epoch:03d}.h5',
        verbose=1,
        monitor='val_accuracy'
    )    
    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir='{}/{}'.format(LOG_DIR, time.time()))
    
    # build model and do training
    model = get_training_model(
        model_name = model_name,
        dropout_rate = dropout_rate,
        optimizer = optimizer,
        label_smoothing = label_smoothing,
        weight_decay = weight_decay)
    
    model.fit(
        x = ds_train,
        steps_per_epoch = 28841 // batch_size,
        validation_data = ds_val,
        validation_steps = 15905 // batch_size,
        callbacks = [lrate, model_ckpt, tensorboard],
        workers = FLAGS.num_data_workers,
        epochs = epochs)
    
    
def main(_argv):
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    config_keras_backend()
    train(FLAGS.model, FLAGS.dropout_rate, FLAGS.optimizer, FLAGS.epsilon,
          FLAGS.label_smoothing,
          FLAGS.batch_size, FLAGS.lr_sched, FLAGS.initial_lr, FLAGS.final_lr,
          FLAGS.weight_decay, FLAGS.epochs, FLAGS.dataset_dir)
    clear_keras_session()
    
if __name__ == "__main__":
    app.run(main) 

    
    
               