import tensorflow as tf
from tensorflow.keras import Model
def DeepSupervision(criterion, xs, y):
    loss = 0.
    for x in xs:
        loss += criterion(x, y)
    return loss
class CrossEntropyLabelSmooth(Model):
    def __init__(self,
                 num_classes,
                 epsilon = 0.1):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
    def call(self, inputs, targets):
        log_probs = tf.nn.log_softmax(inputs, axis=1)
        targets = tf.tensor_scatter_nd_update(tf.zeros(log_probs.shape), tf.transpose([tf.range(log_probs.shape[0]), targets]), tf.ones(log_probs.shape[0]))
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        
        loss = tf.math.reduce_sum(tf.math.reduce_mean(- targets * log_probs, axis = 1))
        return loss
    
if __name__ == '__main__':
    cel = CrossEntropyLabelSmooth(3)
    inputs = tf.constant([[0.2, 0.6, 0.8]]*3)
    #targets = tf.constant([0, 1, 2])
    targets = [0, 1, 2]
    
    print(inputs, targets)
    print(cel(inputs, targets))
    
