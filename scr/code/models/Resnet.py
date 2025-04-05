from tensorflow.keras import backend as k
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, Activation, Dense, Flatten, MaxPooling2D, AveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

from scr.code.models.Blocks import conv_bn_relu, bn_relu, residual_block
from scr.code.utils.FunctionUtils import sharpe_ratio_loss, sharpe_ratio


def build_resnet_model(params):
    conv1_ksize = params['filters_1']
    conv1_nfilter = params['filters']

    kernel_size_1 = params['repetitions_1']
    kernel_size_2 = params['repetitions_3']
    kernel_size_3 = params['repetitions_5']
    kernel_size_4 = params['repetitions_7']

    num_filter_1 = params['filters_2']
    num_filter_2 = params['filters_3']
    num_filter_3 = params['filters_4']
    num_filter_4 = params['filters_5']

    reps_1 = params['repetitions']
    reps_2 = params['repetitions_2']
    reps_3 = params['repetitions_4']
    reps_4 = params['repetitions_6']

    conv2_nfilter = params['filters_6']

    regularized_coff_1 = params['l2']
    regularized_coff_2 = params['l2_1']
    regularized_coff_3 = params['l2_2']
    learning_rate = params['l2_3']
    input_shape = params['input_shape']
    temperature = params['tem_rate']
    ts = input_shape[1]
    tickers = input_shape[0]

    input = Input(shape=input_shape)
    conv1 = conv_bn_relu(filters=conv1_nfilter, kernel_size=(1, conv1_ksize), strides=(1, 1),
                         kernel_regularizer=regularizers.l2(regularized_coff_1))(input)

    pool1 = MaxPooling2D(pool_size=(1, 3), strides=(1, 2), padding="same")(conv1)

    out = residual_block(filters=num_filter_1, repetitions=reps_1, kernel_size=(1, kernel_size_1),
                         strides=(1, 2), is_first_layer=True)(pool1)

    out = residual_block(filters=num_filter_2, repetitions=reps_2,
                         kernel_size=(1, kernel_size_2), strides=(1, 2))(out)

    out = residual_block(filters=num_filter_3, repetitions=reps_3,
                         kernel_size=(1, kernel_size_3), strides=(1, 2))(out)

    out = residual_block(filters=num_filter_4, repetitions=reps_4,
                         kernel_size=(1, kernel_size_4), strides=(1, 2))(out)

    out = bn_relu(out)

    conv2 = conv_bn_relu(filters=conv2_nfilter, kernel_size=(tickers, 1), strides=(1, 1),
                         kernel_regularizer=regularizers.l2(regularized_coff_2), padding='valid')(out)

    out_shape = k.int_shape(conv2)
    out = AveragePooling2D(pool_size=(out_shape[1], out_shape[2]),
                           strides=(1, 1))(conv2)

    out = Flatten()(out)

    out = Dense(tickers, kernel_regularizer=regularizers.l2(regularized_coff_3))(out)
    #out = tf.reduce_mean(out, axis=1)
    out = Activation(lambda x: tf.nn.softmax(x / temperature))(out)

    model = Model([input], [out])
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss=sharpe_ratio_loss, optimizer=optimizer, metrics=[sharpe_ratio])

    return model
