from numpy.lib.arraysetops import isin
import tensorflow as tf

from config.config import config
from .osnet import OSNet

IMG_HEIGHT = 256
IMG_WIDTH = 128
IN_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3) # H, W, C
NUM_CLASSES = 1000

def _set_l2(model, weigth_decay):
    """Add L2 regularization into layers with weights
    Reference: https://jricheimer.github.io/keras/2019/02/06/keras-hack-1/
    """
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.DepthwiseConv2D):
            layer.add_loss(
                tf.keras.regularizers.l2(weigth_decay)(layer.depthwise_kernel)
            )
        elif isinstance(layer, tf.keras.layers.Conv2D):
            layer.add_loss(
                tf.keras.regularizers.l2(weigth_decay)(layer.kernel)
            )
        elif isinstance(layer, tf.keras.layers.Dense):
            layer.add_loss(
                tf.keras.regularizers.l2(weigth_decay)(layer.kernel)
            )

def get_lr_func(total_epochs, lr_sched='linear',
                initial_lr=1e-2, final_lr=1e-5):
    """Returns a learning decay function for training.
    2 types of lr_sched are supported: 'linear' or 'exp' (exponential).
    """
    def linear_decay(epoch):
        """Decay LR linearly for each epoch."""
        if total_epochs == 1:
            return initial_lr
        else:
            ratio = max((total_epochs - epoch - 1.) / (total_epochs - 1.), 0.)
            lr = final_lr + (initial_lr - final_lr) * ratio
            print('Epoch %d, lr = %f' % (epoch+1, lr))
            return lr

    def exp_decay(epoch):
        """Decay LR exponentially for each epoch."""
        if total_epochs == 1:
            return initial_lr
        else:
            lr_decay = (final_lr / initial_lr) ** (1. / (total_epochs - 1))
            lr = initial_lr * (lr_decay ** epoch)
            print('Epoch %d, lr = %f' % (epoch+1, lr))
            return lr
    
    if total_epochs < 1:
        raise ValueError('bad total_epochs (%d)' % total_epochs)
    if lr_sched == 'linear':
        return tf.keras.callbacks.LearningRateScheduler(linear_decay)
    elif lr_sched == 'exp':
        return tf.keras.callbacks.LearningRateScheduler(exp_decay)
    else:
        raise ValueError('bad lr_sched')

def get_optimizer(model_name, optim_name, initial_lr, epsilon=1e-2):
    """get_optimizer
    Note:
    1. Learning rate decay is implemented as a callback in model.fit(),
       so I do not specify 'decay' in the optimizers here.
    2. Refer to the following for information about 'epsilon' in Adam:
       https://github.com/tensorflow/tensorflow/blob/v1.14.0/tensorflow/python/keras/optimizer_v2/adam.py#L93
    """
    if optim_name == 'sgd':
        return tf.keras.optimizers.SGD(lr=initial_lr, momentum=0.9)
    elif optim_name == 'adam':
        return tf.keras.optimizers.Adam(lr=initial_lr, epsilon=epsilon)
    elif optim_name == 'rmsprop':
        return tf.keras.optimizers.RMSprop(lr=initial_lr, epsilon=epsilon,
                                           rho=0.9)
    else:
        # implementation of 'AdamW' is removed temporarily
        raise ValueError

def get_training_model(model_name, dropout_rate, optimizer, label_smoothing, weight_decay):
    """Build the model to be trained."""
    # initialize the model from scratch
    model_class = {'osnet': OSNet,}[model_name]

    backbone = model_class(input_shape=IN_SHAPE, include_top=False, weights=None)
    
    # Add a dropout layer before the final dense output
    x = tf.keras.layers.GlobalAveragePooling2D()(backbone.output)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
        
    kernel_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.02)
    bias_initializer = tf.constant_initializer(value=0.0)
    x = tf.keras.layers.Dense(
        NUM_CLASSES, activation='softmax', name='Logits',
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer)(x)
    model = tf.keras.models.Model(inputs=backbone.input, outputs=x)
        
    
    if weight_decay > 0.:
        _set_l2(model, weight_decay)
    
    # make sure all layers are set to be trainable
    for layer in model.layers:
        layer.trainable = True
    
    if tf.__version__ >= '1.14':
        smooth = 0.1 if label_smoothing else 0.
        loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=smooth)
    else:
        tf.logging.warning('"--label_smoothing" not working for '
                           'tensorflow-%s' % tf.__version__)
        loss = 'categorical_crossentropy'
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy'])

    print(model.summary())

    return model
