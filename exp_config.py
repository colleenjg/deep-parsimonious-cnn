""" parameters of experiments """


def CIFAR10_baseline():
    param = {
        'device': '/gpu:0',
        'data_folder': '../cifar-10-batches-py',  # the path of unzipped CIFAR10 data
        'save_path': '../cifar10_model',  # the path to save your model
        'dataset_name': 'CIFAR10',
        'model_name': 'baseline',
        'merge_valid': False,
        'resume_training': False,
        'bat_size': 100,
        'img_height': 32,
        'img_width': 32,
        'img_channel': 3,
        'disp_iter': 100,
        'save_iter': 10000,
        'max_train_iter': 100000,
        'valid_iter': 1000,
        'base_learn_rate': 1.0e-2,
        'learn_rate_decay_step': 2000,
        'learn_rate_decay_rate': 0.85,
        'label_size': 10,
        'momentum': 0.9,
        'weight_decay': 0.0,
        'init_std_cnn': [1.0e-2, 1.0e-2, 1.0e-2],
        'init_std_mlp': [1.0e-1, 1.0e-1],
        'filter_shape': [[5, 5, 3, 32], [5, 5, 32, 32], [5, 5, 32, 64]],
        'filter_stride': [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
        'pool_func': ['max', 'avg', 'avg'],
        'pool_size': [[1, 3, 3, 1], [1, 3, 3, 1], [1, 3, 3, 1]],
        'pool_stride': [[1, 2, 2, 1], [1, 2, 2, 1], [1, 2, 2, 1]],
        'act_func_cnn': ['relu'] * 3,
        'act_func_mlp': [None] * 2,
        'dims_mlp': [64, 10, 1024],

        'test_model_name': 'baseline_snapshot_0060000.ckpt',
        'test_folder': '../cifar10_model/CIFAR10_baseline_2017-Apr-19-22-33-15'  # the path of your testing model
    }

    return param


def CIFAR100_baseline():
    param = CIFAR10_baseline()
    param['dataset_name'] = 'CIFAR100'
    param['data_folder'] = '../cifar-100-python'  # the path of unzipped CIFAR100 data
    param['label_size'] = 100
    param['init_std_cnn'] = [1.0e-1, 1.0e-1, 1.0e-1]
    param['dims_mlp'] = [64, 100, 1024]
    param['test_model_name'] = 'baseline_snapshot_0010000.ckpt'
    param['test_folder'] = '../cifar10_model/CIFAR100_baseline_2017-Apr-08-19-50-52'

    return param


def CIFAR10_sample_clustering():
    param = CIFAR10_baseline()
    param['eta'] = 0.1
    param['model_name'] = 'parsimonious'
    param['init_std_cnn'] = [1.0e-2, 1.0e-2, 1.0e-2]
    param['num_cluster_cnn'] = [100, 100, 100]
    param['clustering_type_cnn'] = ['sample', 'sample', 'sample']
    param['clustering_shape_cnn'] = [[100, 32768], [100, 8192], [100, 4096]]
    param['clustering_alpha_cnn'] = [1.0e-1, 1.0e-1, 1.0e-1]
    param['num_cluster_mlp'] = [100, 100]
    param['clustering_shape_mlp'] = [[100, 64], [100, 10]]
    param['clustering_alpha_mlp'] = [1.0e-1, 1.0e-1]
    param['clustering_iter'] = 1
    param['test_model_name'] = 'parsimonious_snapshot_0060000.ckpt'
    param['test_folder'] = '../cifar10_model/CIFAR10_sample_clustering_2017-Apr-09-15-34-05'  # the path of your testing model
    param['resume_training'] = False

    if param['resume_training']:
        param['max_train_iter'] = 60000
        param['base_learn_rate'] = 1.0e-3
        param['merge_valid'] = True
        param['resume_step'] = 50000
        param['resume_exp_id'] = ''  # exp id of resume
        param['resume_model_name'] = 'parsimonious_snapshot_0050000.ckpt'

    return param


def CIFAR10_spatial_clustering():
    param = CIFAR10_baseline()
    param['eta'] = 0.1
    param['model_name'] = 'parsimonious'
    param['init_std_cnn'] = [1.0e-2, 1.0e-2, 1.0e-2]
    param['num_cluster_cnn'] = [100, 100, 100]
    param['clustering_type_cnn'] = ['spatial', 'spatial', 'spatial']
    param['clustering_shape_cnn'] = [[102400, 32], [25600, 32], [6400, 64]]
    param['clustering_alpha_cnn'] = [1.0e-1, 1.0e-1, 1.0e-1]
    param['num_cluster_mlp'] = [100, 100]
    param['clustering_shape_mlp'] = [[100, 64], [100, 10]]
    param['clustering_alpha_mlp'] = [1.0e-1, 1.0e-1]
    param['clustering_iter'] = 1
    param['test_model_name'] = 'parsimonious_snapshot_0060000.ckpt'
    param['test_folder'] = '../cifar10_model/CIFAR10_spatial_clustering_2017-Apr-09-15-35-04'
    param['resume_training'] = False

    if param['resume_training']:
        param['max_train_iter'] = 60000
        param['base_learn_rate'] = 1.0e-3
        param['merge_valid'] = True
        param['resume_step'] = 50000
        param['resume_exp_id'] = ''
        param['resume_model_name'] = 'parsimonious_snapshot_0050000.ckpt'

    return param


def CIFAR10_channel_clustering():
    param = CIFAR10_baseline()
    param['eta'] = 0.1
    param['model_name'] = 'parsimonious'
    param['init_std_cnn'] = [1.0e-2, 1.0e-2, 1.0e-2]
    param['num_cluster_cnn'] = [100, 100, 100]
    param['clustering_type_cnn'] = ['channel', 'channel', 'channel']
    param['clustering_shape_cnn'] = [[3200, 1024], [3200, 256], [6400, 64]]
    param['clustering_alpha_cnn'] = [1.0e-1, 1.0e-1, 1.0e-1]
    param['num_cluster_mlp'] = [100, 100]
    param['clustering_shape_mlp'] = [[100, 64], [100, 10]]
    param['clustering_alpha_mlp'] = [1.0e-1, 1.0e-1]
    param['clustering_iter'] = 1
    param['test_model_name'] = 'parsimonious_snapshot_0060000.ckpt'
    param['test_folder'] = '../cifar10_model/CIFAR10_channel_clustering_2017-Apr-09-15-35-52'
    param['resume_training'] = False

    if param['resume_training']:
        param['max_train_iter'] = 60000
        param['base_learn_rate'] = 1.0e-3
        param['merge_valid'] = True
        param['resume_step'] = 50000
        param['resume_exp_id'] = ''
        param['resume_model_name'] = 'parsimonious_snapshot_0050000.ckpt'

    return param


def CIFAR100_sample_clustering():
    param = CIFAR100_baseline()
    param['eta'] = 0.05
    param['model_name'] = 'parsimonious'
    param['num_cluster_cnn'] = [100, 100, 100]
    param['clustering_type_cnn'] = ['sample', 'sample', 'sample']
    param['clustering_shape_cnn'] = [[100, 32768], [100, 8192], [100, 4096]]
    param['clustering_alpha_cnn'] = [1.0e+0, 1.0e+0, 1.0e+0]
    param['num_cluster_mlp'] = [100, None]
    param['clustering_shape_mlp'] = [[100, 64], None]
    param['clustering_alpha_mlp'] = [1.0e+0, None]
    param['clustering_iter'] = 1
    param['test_model_name'] = 'parsimonious_snapshot_0070000.ckpt'
    param['test_folder'] = '../cifar10_model/CIFAR100_sample_clustering_2017-Apr-09-15-37-02'
    param['resume_training'] = False

    if param['resume_training']:
        param['max_train_iter'] = 60000
        param['merge_valid'] = True
        param['base_learn_rate'] = 1.0e-3
        param['resume_step'] = 50000
        param['resume_exp_id'] = ''
        param['resume_model_name'] = 'parsimonious_snapshot_0050000.ckpt'

    return param


def CIFAR100_spatial_clustering():
    param = CIFAR100_baseline()
    param['eta'] = 0.05
    param['model_name'] = 'parsimonious'
    param['num_cluster_cnn'] = [100, 100, 100]
    param['clustering_type_cnn'] = ['spatial', 'spatial', 'spatial']
    param['clustering_shape_cnn'] = [[102400, 32], [25600, 32], [6400, 64]]
    param['clustering_alpha_cnn'] = [1.0e+1, 1.0e+0, 1.0e+0]
    param['num_cluster_mlp'] = [100, None]
    param['clustering_shape_mlp'] = [[100, 64], None]
    param['clustering_alpha_mlp'] = [1.0e+0, None]
    param['clustering_iter'] = 1
    param['test_model_name'] = 'parsimonious_snapshot_0060000.ckpt'
    param['test_folder'] = ''
    param['resume_training'] = False

    if param['resume_training']:
        param['max_train_iter'] = 60000
        param['merge_valid'] = True
        param['base_learn_rate'] = 1.0e-3
        param['resume_step'] = 50000
        param['resume_exp_id'] = ''
        param['resume_model_name'] = 'parsimonious_snapshot_0050000.ckpt'

    return param


def CIFAR100_channel_clustering():
    param = CIFAR100_baseline()
    param['eta'] = 0.05
    param['model_name'] = 'parsimonious'
    param['num_cluster_cnn'] = [100, 100, 100]
    param['clustering_type_cnn'] = ['channel', 'channel', 'channel']
    param['clustering_shape_cnn'] = [[3200, 1024], [3200, 256], [6400, 64]]
    param['clustering_alpha_cnn'] = [1.0e+1, 1.0e+0, 1.0e+0]
    param['num_cluster_mlp'] = [100, None]
    param['clustering_shape_mlp'] = [[100, 64], None]
    param['clustering_alpha_mlp'] = [1.0e+0, None]
    param['clustering_iter'] = 1
    param['test_model_name'] = 'parsimonious_snapshot_0060000.ckpt'
    param['test_folder'] = ''
    param['resume_training'] = False

    if param['resume_training']:
        param['max_train_iter'] = 60000
        param['merge_valid'] = True
        param['base_learn_rate'] = 1.0e-3
        param['resume_step'] = 50000
        param['resume_exp_id'] = ''
        param['resume_model_name'] = 'parsimonious_snapshot_0050000.ckpt'

    return param


def CIFAR10_distilled(lambda_=0.9, temperature=10):
    param = {
        'device': '/gpu:0',
        'data_folder': '../cifar-10-batches-py',  # the path of unzipped CIFAR10 data
        'save_path': '../cifar10_model/CIFAR10_distilled_2017-Apr-10-23-06-49',  # the path to save your model
        'dataset_name': 'CIFAR10',
        'model_name': 'distilled',
        'merge_valid': False,
        'resume_training': False,
        'lambda': lambda_,  # determines the weight of the two objective functions
        'temperature': temperature,
        'bat_size': 100,
        'img_height': 32,
        'img_width': 32,
        'img_channel': 3,
        'disp_iter': 100,
        'save_iter': 10000,
        'max_train_iter': 100000,
        'valid_iter': 1000,
        'base_learn_rate': 1.0e-3,
        'learn_rate_decay_step': 2000,
        'learn_rate_decay_rate': 0.75,
        'label_size': 10,
        'momentum': 0.9,
        'weight_decay': 0.0,
        'init_std_cnn': [1.0e-2, 1.0e-2, 1.0e-2],
        'init_std_mlp': [1.0e-1, 1.0e-1],
        'filter_shape': [[5, 5, 3, 32], [5, 5, 32, 16], [5, 5, 16, 32]],
        'filter_stride': [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
        'pool_func': ['max', 'avg', 'avg'],
        'pool_size': [[1, 3, 3, 1], [1, 3, 3, 1], [1, 3, 3, 1]],
        'pool_stride': [[1, 2, 2, 1], [1, 2, 2, 1], [1, 2, 2, 1]],
        'act_func_cnn': ['relu'] * 3,
        'act_func_mlp': [None] * 2,
        'dims_mlp': [64, 10, 512],
        'test_model_name': 'distilled_snapshot_0020000.ckpt',
        'test_folder': '../cifar10_model/CIFAR10_distilled-baseline_l0-9_t5_2017-Apr-20-11-26-01'
    }

    if param['resume_training']:
        param['max_train_iter'] = 60000
        param['merge_valid'] = True
        param['base_learn_rate'] = 1.0e-3
        param['resume_step'] = 50000
        param['resume_exp_id'] = ''
        param['resume_model_name'] = 'distilled_snapshot_0010000.ckpt'

    return param


def CIFAR10_hybrid_sample(lambda_=0.9, temperature=10):
    param = {
        'device': '/gpu:0',
        'data_folder': '../cifar-10-batches-py', # the path of unzipped CIFAR10 data
        'save_path': '../cifar10_model',  # the path to save your model
        'dataset_name': 'CIFAR10',
        'model_name': 'hybrid_sample',
        'merge_valid': False,
        'resume_training': False,
        'lambda': lambda_, # determines the weight of the two objective functions
        'temperature': temperature,
        'bat_size': 100,
        'img_height': 32,
        'img_width': 32,
        'img_channel': 3,
        'disp_iter': 100,
        'save_iter': 10000,
        'max_train_iter': 100000,
        'valid_iter': 1000,
        'base_learn_rate': 1.0e-3,
        'learn_rate_decay_step': 2000,
        'learn_rate_decay_rate': 0.75,
        'label_size': 10,
        'momentum': 0.9,
        'weight_decay': 0.0,
        'init_std_cnn': [1.0e-2, 1.0e-2, 1.0e-2],
        'init_std_mlp': [1.0e-1, 1.0e-1],
        'filter_shape': [[5, 5, 3, 32], [5, 5, 32, 16], [5, 5, 16, 32]],
        'filter_stride': [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
        'pool_func': ['max', 'avg', 'avg'],
        'pool_size': [[1, 3, 3, 1], [1, 3, 3, 1], [1, 3, 3, 1]],
        'pool_stride': [[1, 2, 2, 1], [1, 2, 2, 1], [1, 2, 2, 1]],
        'act_func_cnn': ['relu'] * 3,
        'act_func_mlp': [None] * 2,
        'dims_mlp': [64, 10, 512],
        'test_model_name': 'hybrid_sample_snapshot_0060000.ckpt',
        'test_folder': '../cifar10_model/CIFAR10_hybrid_sample_l0-9_t5_2017-Apr-20-18-31-36',
        'eta': 0.1,

        # clustering fields
        'num_cluster_cnn': [100, 100, 100],
        'clustering_type_cnn': ['sample', 'sample', 'sample'],
        'clustering_shape_cnn': [[100, 32768], [100, 4096], [100, 2048]],
        'clustering_alpha_cnn': [1.0e-1, 1.0e-1, 1.0e-1],
        'num_cluster_mlp': [100, 100],
        'clustering_shape_mlp': [[100, 64], [100, 10]],
        'clustering_alpha_mlp': [1.0e-1, 1.0e-1],
        'clustering_iter': 1
    }

    return param


def CIFAR10_hybrid_spatial(lambda_=0.9, temperature=10):
    param = {
        'device': '/gpu:0',
        'data_folder': '../cifar-10-batches-py', # the path of unzipped CIFAR10 data
        'save_path': '../cifar10_model',  # the path to save your model
        'dataset_name': 'CIFAR10',
        'model_name': 'hybrid_spatial',
        'merge_valid': False,
        'resume_training': False,
        'lambda': lambda_, # determines the weight of the two objective functions
        'temperature': temperature,
        'bat_size': 100,
        'img_height': 32,
        'img_width': 32,
        'img_channel': 3,
        'disp_iter': 100,
        'save_iter': 10000,
        'max_train_iter': 100000,
        'valid_iter': 1000,
        'base_learn_rate': 1.0e-3,
        'learn_rate_decay_step': 2000,
        'learn_rate_decay_rate': 0.75,
        'label_size': 10,
        'momentum': 0.9,
        'weight_decay': 0.0,
        'init_std_cnn': [1.0e-2, 1.0e-2, 1.0e-2],
        'init_std_mlp': [1.0e-1, 1.0e-1],
        'filter_shape': [[5, 5, 3, 32], [5, 5, 32, 16], [5, 5, 16, 32]],
        'filter_stride': [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
        'pool_func': ['max', 'avg', 'avg'],
        'pool_size': [[1, 3, 3, 1], [1, 3, 3, 1], [1, 3, 3, 1]],
        'pool_stride': [[1, 2, 2, 1], [1, 2, 2, 1], [1, 2, 2, 1]],
        'act_func_cnn': ['relu'] * 3,
        'act_func_mlp': [None] * 2,
        'dims_mlp': [64, 10, 512],
        'test_model_name': 'hybrid_spatial_snapshot_0070000.ckpt',
        'test_folder': '../cifar10_model/CIFAR10_hybrid_spatial_l0-9_t5_2017-Apr-20-15-41-46',
        'eta': 0.1,

        # clustering fields
        'num_cluster_cnn': [100, 100, 100],
        'clustering_type_cnn': ['spatial', 'spatial', 'spatial'],
        'clustering_shape_cnn': [[102400, 32], [25600, 16], [6400, 32]],
        'clustering_alpha_cnn': [1.0e-1, 1.0e-1, 1.0e-1],
        'num_cluster_mlp': [100, 100],
        'clustering_shape_mlp': [[100, 64], [100, 10]],
        'clustering_alpha_mlp': [1.0e-1, 1.0e-1],
        'clustering_iter': 1
    }

    return param
    
