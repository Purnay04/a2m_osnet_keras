import os
import time
import argparse
from utils.dataset import get_dataset

import tensorflow as tf

from config.config import config
from utils import config_keras_backend, clear_keras_session
from models.models import get_optimizer, get_training_model, get_lr_func


DESCRIPTION = """For example:
$ python3 train.py --dataset_dir  /media/toanmh/Workspace/Datasets/ImageNet/ILSVRC2012/tfrecords \
                   --dropout_rate 0.4 \
                   --optimizer    adam \
                   --epsilon      1e-1 \
                   --label_smoothing \
                   --batch_size   16 \
                   --iter_size    1 \
                   --lr_sched     exp \
                   --initial_lr   1e-2 \
                   --final_lr     1e-5 \
                   --weight_decay 2e-4 \
                   --epochs       60 \
                   os_net
"""

SUPPORTED_MODELS = ('"osnet"'
                'or just specify a saved Keras model (.h5) file')


def train(model_name, dropout_rate, optim_name, epsilon,
          label_smoothing, batch_size, lr_sched, initial_lr, final_lr,
          weight_decay, epochs, dataset_dir):
    """Prepare data and train the model."""
    batch_size = batch_size
    initial_lr = initial_lr
    final_lr = final_lr
    optimizer = get_optimizer(model_name, optim_name, initial_lr, epsilon)
    weight_decay = weight_decay

    # get trainign and validation data
    ds_train = get_dataset(dataset_dir, 'train', batch_size)
    ds_val = get_dataset(dataset_dir, 'validation', batch_size)

    
    lrate = get_lr_func(epochs, lr_sched, initial_lr, final_lr)
    save_name = model_name if not model_name.endswith('.h5') else \
                os.path.split(model_name)[-1].split('.')[0].split('-')[0]
    model_ckpt = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(config.SAVE_DIR, save_name) + '-ckpt-{epoch:03d}.h5',
        verbose=1, 
        monitor='val_accuracy'
        )
    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir='{}/{}'.format(config.LOG_DIR, time.time()))

    # build model and do training
    model = get_training_model(
        model_name=model_name,
        dropout_rate=dropout_rate,
        optimizer=optimizer,
        label_smoothing=label_smoothing,
        weight_decay=weight_decay)

    model.fit(
        x=ds_train,
        steps_per_epoch=320291 // batch_size, # number of image in training set is 1M
        validation_data=ds_val,                # number of image in validation set is 50K
        validation_steps=12500 // batch_size,
        callbacks=[lrate, model_ckpt, tensorboard],
        # The following doesn't seem to help in terms of speed.
        # use_multiprocessing=True, 
        workers=config.NUM_DATA_WORKERS,
        epochs=epochs)
    
    #Launching 4 threads for spacings: [[0, 320291], [320291, 640583], [640583, 960875], [960875, 1281167]]
    

def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--dataset_dir', type=str, default=config.DATASET_DIR)
    parser.add_argument('--dropout_rate', type=float, default=0.3)
    parser.add_argument('--optimizer', type=str, default='adam', choices=['sgd', 'adam', 'rmsprop'])
    parser.add_argument('--epsilon', type=float, default=1e-3)
    parser.add_argument('--label_smoothing', action='store_true')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr_sched', type=str, default='linear', choices=['linear', 'exp'])
    parser.add_argument('--initial_lr', type=float, default=1e-2)
    parser.add_argument('--final_lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=50, help='total number of epochs for training')
    parser.add_argument('model', type=str, default='osnet', help=SUPPORTED_MODELS)
    args = parser.parse_args()

    os.makedirs(config.SAVE_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)
    config_keras_backend()
    train(args.model, args.dropout_rate, args.optimizer, args.epsilon,
          args.label_smoothing,
          args.batch_size, args.lr_sched, args.initial_lr, args.final_lr,
          args.weight_decay, args.epochs, args.dataset_dir)
    clear_keras_session()


if __name__ == '__main__':
    main()