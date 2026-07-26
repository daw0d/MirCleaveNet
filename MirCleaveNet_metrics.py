import tensorflow as tf
from keras import backend as K

def avg_proximity_metric():

    def prox(y_true, y_pred):
        eps = K.epsilon()
        beta = 1e10
        y_pred_T = tf.transpose(y_pred,perm=[0,2,1])
        y_true_T = tf.transpose(y_true,perm=[0,2,1])
        y_pred_argmax = K.cast(K.argmax(y_pred_T,axis=-1),dtype='float32')
        y_true_argmax = K.cast(K.argmax(y_true_T,axis=-1),dtype='float32')
        y_pred_slice = y_pred_argmax[:,1:5]
        y_true_slice = y_true_argmax[:,1:5]
        y_prox = K.abs(y_pred_slice - y_true_slice)
        y_prox_sum = K.sum(y_prox,axis=-1)
        
        return K.mean(y_prox_sum, axis=-1)
    
    return prox
