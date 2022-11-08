import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback
import tensorflow


class callback(Callback):
    def __init__(self, model, data):
        self.model = model
        self.data = data
        # self.x = X_train

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        # inp = self.model.input  # input placeholder
        outputs = self.model.layers[3](self.data)  # get output of N's layer
        # functors = K.function([inp, K.learning_phase()], [outputs])
        # layer_outs = functors([self.x, 1.])
        # print('\r OUTPUT TENSOR : %s' % layer_outs)
        tensorflow.print(outputs)

        return


def K_eval(x):
    try:
        return K.get_value(K.to_dense(x))
    except:
        eval_fn = K.function([], [x])
        return eval_fn([])[0]
