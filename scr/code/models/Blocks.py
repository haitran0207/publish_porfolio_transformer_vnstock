from tensorflow.keras import backend as k
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Activation, BatchNormalization, Add, Conv2D

def bn_relu(input):
    """Helper to build a BN -> relu block
    """
    norm = BatchNormalization()(input)
    return Activation("relu")(norm)

def conv_bn_relu(**conv_params):
    """Helper to build a conv -> BN -> relu block
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "glorot_uniform")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", regularizers.l2(1.e-4))

    def f(input):
        conv = Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(input)
        return bn_relu(conv)

    return f

def bn_relu_conv(**conv_params):
    """Helper to build a BN -> relu -> conv block.
    """

    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "glorot_uniform")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", regularizers.l2(1.e-4))

    def f(input):
        activation = bn_relu(input)
        return Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(activation)

    return f

def short_cut(input, residual):
    """Adds a shortcut between input and residual block and merges them with "sum"
    """

    input_shape = k.int_shape(input)
    residual_shape = k.int_shape(residual)
    stride_width = int(round(input_shape[1] / residual_shape[1]))
    stride_height = int(round(input_shape[2] / residual_shape[2]))
    equal_channels = input_shape[3] == residual_shape[3]

    shortcut = input
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Conv2D(filters=residual_shape[3],
                          kernel_size=(1, 1),
                          strides=(stride_width, stride_height),
                          padding="valid",
                          kernel_regularizer=regularizers.l2(0.001))(input)

    return Add()([shortcut, residual])

def residual_block(filters, repetitions, kernel_size=(3, 3), strides=(2, 2), is_first_layer=False):
    """Builds a residual block with repeating bottleneck blocks.
    """

    def f(input):
        for i in range(repetitions):
            init_strides = (1, 1)
            if i == 0 and not is_first_layer:
                init_strides = strides
            input = basic_block(filters=filters, kernel_size=kernel_size, init_strides=init_strides,
                                is_first_block_of_first_layer=(is_first_layer and i == 0))(input)
        return input

    return f

def basic_block(filters, kernel_size=(3, 3), init_strides=(1, 1), is_first_block_of_first_layer=False):
    def f(input):

        if is_first_block_of_first_layer:
            conv1 = Conv2D(filters=filters, kernel_size=kernel_size,
                           strides=init_strides,
                           padding="same",
                           kernel_initializer="glorot_uniform",
                           kernel_regularizer=regularizers.l2(1e-4))(input)
        else:
            conv1 = bn_relu_conv(filters=filters, kernel_size=kernel_size,
                                 strides=init_strides)(input)

        residual = bn_relu_conv(filters=filters, kernel_size=kernel_size)(conv1)
        return short_cut(input, residual)

    return f