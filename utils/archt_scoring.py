import numpy as np
import tensorflow as tf
import os
os.environ['TF_DETERMINISTIC_OPS'] = '1'

np.random.seed(0)
from tensorflow.keras import backend
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, Embedding
from tensorflow.keras.layers import BatchNormalization, Activation, LSTM, TimeDistributed, Bidirectional
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.python.framework.convert_to_constants import  convert_variables_to_constants_v2_as_graph
from tensorflow.keras.initializers import GlorotNormal, GlorotUniform

def scorefunc_slogdet(K, labels=None):
    s, ld = np.linalg.slogdet(K)
    return ld


def tf_net_kmatrix(model, batch_size, input ):
    kmatrix = np.zeros((batch_size, batch_size))

    # forward(batch_inputs)
    # for each RELU layer in layers of model:
    for layer in model.layers:
        if "relu" in layer._name:
            print (layer._name)
            activations_model = Model(model.inputs, outputs=model.get_layer(layer._name).output)
            print (activations_model.summary())

            # amsgrad = optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=True,
            #                           name='Adam')
            # activations_model.compile(loss='mean_squared_error', optimizer=amsgrad, metrics='mae')

            # activations_1 = activations_model.predict(input)


            # Flatten each relu output

            # Binarize each relu output ( set positive linear values to 1)

            # K = x @ x.t()

            # K2 = (1. - x) @ (1. - x.t())

            # kmatrix = kmatrix + K + K2


    # ############# Original PyTorch ver. #############
    # network.K = np.zeros((args.batch_size, args.batch_size))
    #
    # def counting_forward_hook(module, inp, out):
    #     try:
    #         if not module.visited_backwards:
    #             return
    #         if isinstance(inp, tuple):
    #             inp = inp[0]
            ## Returns a new tensor with the same data as the self tensor but of a different shape
    #         inp = inp.view(inp.size(0), -1)
    #         x = (inp > 0).float()
    #
    #         # @ is an operator for matrix multiplication # https://docs.python.org/3/whatsnew/3.5.html#whatsnew-pep-465
    #         K = x @ x.t()
    #         K2 = (1. - x) @ (1. - x.t())
    #         network.K = network.K + K.cpu().numpy() + K2.cpu().numpy()
    #     except:
    #         pass
    #
    # def counting_backward_hook(module, inp, out):
    #     module.visited_backwards = True
    #
    # for name, module in network.named_modules():
    #     if 'ReLU' in str(type(module)):
    #         # hooks[name] = module.register_forward_hook(counting_hook)
            ## hook is 'counting_forward_hook'
            ## The hook will be called every time after forward() has computed an output. It should have the following signature:
    #         module.register_forward_hook(counting_forward_hook)
    #         module.register_backward_hook(counting_backward_hook)
    # ##########################à



    return kmatrix