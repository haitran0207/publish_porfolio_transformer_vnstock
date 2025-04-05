from scr.code.models.TransfomerModels import build_transformer_model, Transformer, CustomSchedule
from scr.code.preData.preprocess_data import *
from scr.code.utils.FunctionUtils import *
from scr.code.utils.ModelFunction import *
from tensorflow.keras import backend as k

from scr.code.constants.constant import *


def excutedTransfomerModelInBatch(optimize_method,data_path, window_x, window_y):

    optimize_method = optimize_method
    data_path = data_path
    window_x = window_x
    window_y = window_y

    learning_rate = CustomSchedule(window_x)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    # Train Model
    for path in data_path:
        model_name = 'TRANSFORMER'

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

        # if a checkpoint exists, restore the latest checkpoint.
        X, y, y_vnindex, tickers, dates, X_return, close = prepair_data_new(path, window_x, window_y,step_roll=step_roll)

        # Build Model
        model = Transformer(
            num_layers=hyper_params['num_layers'], d_model=hyper_params['d_model'],
            num_heads=hyper_params['num_heads'], dff=hyper_params['dff'],
            rate=hyper_params['dropout_rate'], max_statements=k.maximum(window_x,X.shape[1]),
            temperature=hyper_params['tem_rate']
        )

        ckpt = tf.train.Checkpoint(transformer=model, optimizer=optimizer)
        ckpt_manager = tf.train.CheckpointManager(ckpt, os.path.join(pretrain_model_path, model_name), max_to_keep=5)
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            model.compile(loss=sharpe_ratio_loss, optimizer=optimizer, metrics=[sharpe_ratio])
            print('Latest checkpoint restored!!')
        else:
            model = build_transformer_model(hyper_params, hyper_params['num_layers'], hyper_params['d_model'], hyper_params['num_heads'], hyper_params['dff'], hyper_params['dropout_rate'], k.maximum(window_x,X.shape[1]), learning_rate = 1e-3, temperature=hyper_params['tem_rate'])

        TrainModel_New(optimize_method, model_name, close, tickers, model,X, X_return, dates, window_y, y,optimizer)