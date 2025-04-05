n_fold = 5
batch_size = 2
epochs = 1
window_x = 200 # 32
window_y = 100 # 8
step_roll = 10

top_n = 20
risk_free_rate = 0.05

input_shape = 'input_shape'
pretrain_model_path = 'pretrain_models'
logs_path = 'logs'
data_path = 'data/price_data_2010_2024_2.csv'

model_config_path = {'ResNet': "config/resnet_hyper_params.json", 'GRU': "config/gru_hyper_params.json",
                     'LSTM': "config/lstm_hyper_params.json", 'AA_GRU': "config/gru_hyper_params.json",
                     'AA_LSTM': "config/lstm_hyper_params.json", 'SA_GRU': "config/gru_hyper_params.json",
                     'SA_LSTM': "config/lstm_hyper_params.json", 'TRANSFORMER': "config/transformer_hyper_params.json"}