from models.osnet import OSNet
from models.models import _set_l2, get_optimizer
import tensorflow as tf

def store_summary(model):
    with open("./data/model_summary.txt", 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

def get_model():
    IMG_HEIGHT = 128
    IMG_WIDTH = 64
    IN_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3) # H, W, C
    
    backbone_model = OSNet(
        input_shape=IN_SHAPE,
        include_top = False,
        weights = None)
    
    x = tf.keras.layers.GlobalAveragePooling2D()(backbone_model.output)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    kernel_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.02)
    bias_initializer = tf.constant_initializer(value=0.0)
    
    x = tf.keras.layers.Dense(
        1501, activation='softmax', name='Logits',
        kernel_initializer=kernel_initializer,
        bias_initializer= bias_initializer)(x)
    model = tf.keras.models.Model(inputs=backbone_model.input, outputs=x)
    return model

if __name__ == '__main__':
    model = get_model()
    _set_l2(model, 1e-4)
    
    for layer in model.layers:
        layer.trainable = True
        
    if tf.__version__ >= '1.14':
        smooth = 0
        loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=smooth)
    
    optimizer = get_optimizer('osnet', 'adam', 1e-2, 1e-3)
        
    model.compile(
        optimizer = optimizer,
        loss = loss,
        metrics = ['accuracy'])
    
    store_summary(model)
    print("Done")
    
    
     
