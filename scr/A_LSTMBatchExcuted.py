import gc

from scr.code.constants.constant import *
from scr.code.models.RnnModels import build_lstm_model
from scr.code.preData.preprocess_data import *
from scr.code.utils.FunctionUtils import *
from scr.code.utils.ModelFunction import *


def excutedLSTMModelInBatch(optimize_method,data_path, window_x, window_y):

    optimize_method = optimize_method
    data_path = data_path
    window_x = window_x
    window_y = window_y

    # Train Model
    for path in data_path:
        model_name = 'LSTM'

        # Load Model
        try:
            os.listdir(os.path.join(pretrain_model_path, model_name))
        except FileNotFoundError:
            os.mkdir(os.path.join(pretrain_model_path, model_name))

        # Load Config
        hyper_params = load_config_file(model_config_path[model_name])

        # Add Sub Path
        sub_string = path.split('_')[2] + '_' + path.split('_')[3] + '_' + path.split('_')[4] + '_' + str(
            window_x) + '_' + str(window_y)
        model_name = model_name + '/' + sub_string

        # Load Data
        X, y, y_vnindex, tickers, dates, X_return, close = prepair_data_new(path, window_x, window_y, step_roll=step_roll)
        hyper_params[input_shape] = (X.shape[1], X.shape[2], X.shape[3])
        model = build_lstm_model(hyper_params)

        # if a checkpoint exists, restore the latest checkpoint.
        ckpt = tf.train.Checkpoint(transformer=model, optimizer=model.optimizer)
        ckpt_manager = tf.train.CheckpointManager(ckpt, os.path.join(pretrain_model_path, model_name), max_to_keep=5)

        # if a checkpoint exists, restore the latest checkpoint.
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print('Latest checkpoint restored!!')
        else:
            model.compile(loss=sharpe_ratio_loss, optimizer=model.optimizer, metrics=[sharpe_ratio])

        TrainModel_New(optimize_method, model_name, close, tickers, model,X, X_return, dates, window_y, y,None)
