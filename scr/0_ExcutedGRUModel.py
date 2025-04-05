import sys
import os
import tensorflow as tf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
            tf.config.experimental.set_virtual_device_configuration(
                gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=9216)]
            )
    except RuntimeError as e:
        print(e)


from scr.A_GruBatchExcuted import excutedGRUModelInBatch

data_path = ['data/price_data_fa_2011_2023_101.csv',
             'data/price_data_ta_2011_2023_101.csv']
windows = [
    [200,100]
]
optimize_method = 'SRO'

for i in range(len(windows)):
    window_x = windows[i][0]
    window_y = windows[i][1]
    print("excutedGRUModelInBatch" + "_" + str(window_x) + "_" + str(window_y))
    excutedGRUModelInBatch(optimize_method=optimize_method, data_path=data_path, window_x=window_x, window_y=window_y)