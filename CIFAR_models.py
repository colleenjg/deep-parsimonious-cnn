import tensorflow as tf
import nn_cell_lib as nn


def baseline_model(param):
    """ Build a Alex-net style model """

    ops = {}
    conv_filters = {'filter_shape': param[
        'filter_shape'], 'filter_stride': param['filter_stride']}
    pooling = {'func_name': param['pool_func'], 'pool_size': param[
        'pool_size'], 'pool_stride': param['pool_stride']}

    device = '/cpu:0'
    if 'device' in param.keys():
        device = param['device']

    with tf.device(device):
        input_images = tf.placeholder(
            tf.float32, [None, param['img_height'], param['img_width'],
                         param['img_channel']])
        input_labels = tf.placeholder(tf.int32, [None])

        ops['input_images'] = input_images
        ops['input_labels'] = input_labels

        # build a CNN
        CNN = nn.CNN(
            conv_filters,
            pooling,
            param['act_func_cnn'],
            init_std=param['init_std_cnn'],
            wd=param['weight_decay'],
            scope='CNN')

        # build a MLP
        MLP = nn.MLP(
            param['dims_mlp'],
            param['act_func_mlp'],
            init_std=param['init_std_mlp'],
            wd=param['weight_decay'],
            scope='MLP')

        # prediction model
        feat_map = CNN.run(input_images)
        feat_map_MLP = tf.reshape(feat_map[-1], [-1, param['dims_mlp'][-1]])
        # logits = MLP.run(feat_map_MLP)[-1]
        embedding_mlp = MLP.run(feat_map_MLP)
        logits = embedding_mlp[-1]
        ops['logits'] = logits
        scaled_logits = tf.nn.softmax(logits)
        ops['scaled_logits'] = scaled_logits

        embedding_cnn = feat_map
        ops['embeddings'] = embedding_cnn + embedding_mlp

        # compute cross-entropy loss
        CE_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=input_labels))
        ops['CE_loss'] = CE_loss

        # setting optimization
        global_step = tf.Variable(0.0, trainable=False)
        learn_rate = tf.train.exponential_decay(
            param['base_learn_rate'], global_step,
            param['learn_rate_decay_step'], param['learn_rate_decay_rate'],
            staircase=True)

        # plain optimizer
        ops['train_step'] = tf.train.MomentumOptimizer(
            learning_rate=learn_rate, momentum=param['momentum']).minimize(
            CE_loss, global_step=global_step)

    return ops


def clustering_model(param):
    """ Build a Alex-net style model with clustering """

    ops = {}
    conv_filters = {'filter_shape': param[
        'filter_shape'], 'filter_stride': param['filter_stride']}
    pooling = {'func_name': param['pool_func'], 'pool_size': param[
        'pool_size'], 'pool_stride': param['pool_stride']}

    device = '/cpu:0'
    if 'device' in param.keys():
        device = param['device']

    num_layer_cnn = len(param['num_cluster_cnn'])
    num_layer_mlp = len(param['num_cluster_mlp'])

    with tf.device(device):
        input_images = tf.placeholder(tf.float32, [None, param['img_height'], param[
                                      'img_width'], param['img_channel']])
        input_labels = tf.placeholder(tf.int32, [None])
        input_eta = tf.placeholder(tf.float32, [])

        c_reset_idx_cnn = [tf.placeholder(tf.int32, [None])
                           for _ in xrange(num_layer_cnn)]
        s_reset_idx_cnn = [tf.placeholder(tf.int32, [None])
                           for _ in xrange(num_layer_cnn)]
        c_reset_idx_mlp = [tf.placeholder(tf.int32, [None])
                           for _ in xrange(num_layer_mlp)]
        s_reset_idx_mlp = [tf.placeholder(tf.int32, [None])
                           for _ in xrange(num_layer_mlp)]

        ops['input_images'] = input_images
        ops['input_labels'] = input_labels
        ops['input_eta'] = input_eta
        ops['c_reset_idx_cnn'] = c_reset_idx_cnn
        ops['s_reset_idx_cnn'] = s_reset_idx_cnn
        ops['c_reset_idx_mlp'] = c_reset_idx_mlp
        ops['s_reset_idx_mlp'] = s_reset_idx_mlp

        # build a CNN
        CNN = nn.CNN_cluster(
            conv_filters=conv_filters,
            pooling=pooling,
            clustering_type=param['clustering_type_cnn'],
            clustering_shape=param['clustering_shape_cnn'],
            alpha=param['clustering_alpha_cnn'],
            num_cluster=param['num_cluster_cnn'],
            activation=param['act_func_cnn'],
            wd=param['weight_decay'],
            init_std=param['init_std_cnn'],
            scope='my_CNN')

        # build a MLP
        MLP = nn.MLP_cluster(
            dims=param['dims_mlp'],
            clustering_shape=param['clustering_shape_mlp'],
            alpha=param['clustering_alpha_mlp'],
            num_cluster=param['num_cluster_mlp'],
            activation=param['act_func_mlp'],
            init_std=param['init_std_mlp'],
            scope='my_MLP')

        # prediction ops
        feat_map, embedding_cnn, clustering_ops_cnn, reg_ops_cnn, reset_ops_cnn = CNN.run(
            input_images, input_eta, c_reset_idx_cnn, s_reset_idx_cnn)

        feat_map_mlp = tf.reshape(feat_map[-1], [-1, param['dims_mlp'][-1]])

        logits, embedding_mlp, clustering_ops_mlp, reg_ops_mlp, reset_ops_mlp = MLP.run(
            feat_map_mlp, input_eta, c_reset_idx_mlp, s_reset_idx_mlp)

        logits = logits[-1]
        ops['logits'] = logits
        scaled_logits = tf.nn.softmax(logits)
        ops['scaled_logits'] = scaled_logits
        ops['cluster_label'] = []
        ops['cluster_center'] = []

        for ii, cc in enumerate(CNN.cluster_center):
            if cc is not None:
                ops['cluster_label'] += [CNN.cluster_label[ii]]
                ops['cluster_center'] += [cc]

        for ii, cc in enumerate(MLP.cluster_center):
            if cc is not None:
                ops['cluster_label'] += [MLP.cluster_label[ii]]
                ops['cluster_center'] += [cc]

        ops['embeddings'] = embedding_cnn + embedding_mlp
        ops['clustering_ops'] = clustering_ops_cnn + clustering_ops_mlp
        ops['reg_ops'] = reg_ops_cnn + reg_ops_mlp
        ops['reset_ops'] = reset_ops_cnn + reset_ops_mlp
        reg_term = tf.reduce_sum(tf.stack(ops['reg_ops']))

        # compute cross-entropy loss
        CE_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=input_labels))
        ops['CE_loss'] = CE_loss

        # setting optimization
        global_step = tf.Variable(0.0, trainable=False)
        learn_rate = tf.train.exponential_decay(param['base_learn_rate'], global_step, param[
                                                'learn_rate_decay_step'], param['learn_rate_decay_rate'], staircase=True)

        # plain optimizer
        ops['train_step'] = tf.train.MomentumOptimizer(learning_rate=learn_rate, momentum=param[
                                                       'momentum']).minimize(CE_loss + reg_term, global_step=global_step)

    return ops


def distilled_model(param):
    """ Build a Alex-net style smaller model """

    ops = {}
    conv_filters = {'filter_shape': param[
        'filter_shape'], 'filter_stride': param['filter_stride']}
    pooling = {'func_name': param['pool_func'], 'pool_size': param[
        'pool_size'], 'pool_stride': param['pool_stride']}

    device = '/cpu:0'
    if 'device' in param.keys():
        device = param['device']

    with tf.device(device):
        input_images = tf.placeholder(
            tf.float32, [None, param['img_height'], param['img_width'],
                         param['img_channel']])
        input_labels = tf.placeholder(tf.int32, [None])

        ops['input_images'] = input_images
        ops['input_labels'] = input_labels

        # build a CNN
        CNN = nn.CNN(
            conv_filters,
            pooling,
            param['act_func_cnn'],
            init_std=param['init_std_cnn'],
            wd=param['weight_decay'],
            scope='dist_CNN')

        # build a MLP
        MLP = nn.MLP(
            param['dims_mlp'],
            param['act_func_mlp'],
            init_std=param['init_std_mlp'],
            wd=param['weight_decay'],
            scope='dist_MLP')

        # prediction model
        feat_map = CNN.run(input_images)
        feat_map_MLP = tf.reshape(feat_map[-1], [-1, param['dims_mlp'][-1]])
        embedding_mlp = MLP.run(feat_map_MLP)
        logits = embedding_mlp[-1]
        scaled_logits = tf.nn.softmax(logits)
        ops['scaled_logits'] = scaled_logits

        embedding_cnn = feat_map
        ops['embeddings'] = embedding_cnn + embedding_mlp

        source_model_logits = tf.placeholder(tf.float32, shape=logits.shape)
        ops['source_model_logits'] = source_model_logits

        # compute losses
        temperature = param['temperature']
        q = tf.nn.softmax(tf.scalar_mul(1/temperature, logits))
        p = tf.nn.softmax(tf.scalar_mul(1/temperature, source_model_logits))

        soft_objective = tf.reduce_mean(
            -tf.reduce_sum(q * tf.log(p), reduction_indices=[1]))

        hard_objective = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=input_labels))

        # weighting coefficient for soft and hard targets
        lambda_ = param['lambda']
        soft_objective = tf.scalar_mul(lambda_, soft_objective)
        hard_objective = tf.scalar_mul((1 - lambda_), hard_objective)

        loss = soft_objective + hard_objective

        ops['loss'] = loss

        # setting optimization
        global_step = tf.Variable(0.0, trainable=False)
        learn_rate = tf.train.exponential_decay(
            param['base_learn_rate'], global_step,
            param['learn_rate_decay_step'], param['learn_rate_decay_rate'],
            staircase=True)

        optimizer = tf.train.MomentumOptimizer(
            learning_rate=learn_rate,
            momentum=param['momentum'])

        # gradients from soft targets multiplied by T^2 as recommended
        grads_and_vars_soft = optimizer.compute_gradients(soft_objective)
        grads_and_vars_soft_scaled = []
        for grad, var in grads_and_vars_soft:
            if grad is None:
                grads_and_vars_soft_scaled.append((grad, var))
            else:
                grads_and_vars_soft_scaled.append(
                    (tf.scalar_mul(temperature**2, grad), var))

        grads_and_vars_hard = optimizer.compute_gradients(hard_objective)

        grads_and_vars = list(
            set(grads_and_vars_soft_scaled + grads_and_vars_hard))

        ops['train_step'] = optimizer.apply_gradients(
            grads_and_vars, global_step=global_step)

    return ops


def hybrid_model(param):
    ops = {}
    conv_filters = {'filter_shape': param[
        'filter_shape'], 'filter_stride': param['filter_stride']}
    pooling = {'func_name': param['pool_func'], 'pool_size': param[
        'pool_size'], 'pool_stride': param['pool_stride']}

    device = '/cpu:0'
    if 'device' in param.keys():
        device = param['device']

    num_layer_cnn = len(param['num_cluster_cnn'])
    num_layer_mlp = len(param['num_cluster_mlp'])

    with tf.device(device):
        input_images = tf.placeholder(
            tf.float32, [None, param['img_height'], param['img_width'],
                         param['img_channel']])
        input_labels = tf.placeholder(tf.int32, [None])
        input_eta = tf.placeholder(tf.float32, [])

        c_reset_idx_cnn = [tf.placeholder(tf.int32, [None])
                           for _ in xrange(num_layer_cnn)]
        s_reset_idx_cnn = [tf.placeholder(tf.int32, [None])
                           for _ in xrange(num_layer_cnn)]
        c_reset_idx_mlp = [tf.placeholder(tf.int32, [None])
                           for _ in xrange(num_layer_mlp)]
        s_reset_idx_mlp = [tf.placeholder(tf.int32, [None])
                           for _ in xrange(num_layer_mlp)]

        ops['input_images'] = input_images
        ops['input_labels'] = input_labels
        ops['input_eta'] = input_eta
        ops['c_reset_idx_cnn'] = c_reset_idx_cnn
        ops['s_reset_idx_cnn'] = s_reset_idx_cnn
        ops['c_reset_idx_mlp'] = c_reset_idx_mlp
        ops['s_reset_idx_mlp'] = s_reset_idx_mlp

        # build a CNN
        CNN = nn.CNN_cluster(
            conv_filters=conv_filters,
            pooling=pooling,
            clustering_type=param['clustering_type_cnn'],
            clustering_shape=param['clustering_shape_cnn'],
            alpha=param['clustering_alpha_cnn'],
            num_cluster=param['num_cluster_cnn'],
            activation=param['act_func_cnn'],
            wd=param['weight_decay'],
            init_std=param['init_std_cnn'],
            scope='my_CNN')

        # build a MLP
        MLP = nn.MLP_cluster(
            dims=param['dims_mlp'],
            clustering_shape=param['clustering_shape_mlp'],
            alpha=param['clustering_alpha_mlp'],
            num_cluster=param['num_cluster_mlp'],
            activation=param['act_func_mlp'],
            init_std=param['init_std_mlp'],
            scope='my_MLP')

        # prediction ops
        feat_map, embedding_cnn, clustering_ops_cnn, reg_ops_cnn, reset_ops_cnn = CNN.run(
            input_images, input_eta, c_reset_idx_cnn, s_reset_idx_cnn)

        feat_map_mlp = tf.reshape(feat_map[-1], [-1, param['dims_mlp'][-1]])

        logits, embedding_mlp, clustering_ops_mlp, reg_ops_mlp, reset_ops_mlp = MLP.run(
            feat_map_mlp, input_eta, c_reset_idx_mlp, s_reset_idx_mlp)

        # prediction model
        logits = logits[-1]
        ops['logits'] = logits
        scaled_logits = tf.nn.softmax(logits)
        ops['scaled_logits'] = scaled_logits

        source_model_logits = tf.placeholder(tf.float32, shape=logits.shape)
        ops['source_model_logits'] = source_model_logits

        # compute losses
        temperature = param['temperature']
        q = tf.nn.softmax(tf.scalar_mul(1/temperature, logits))
        p = tf.nn.softmax(tf.scalar_mul(1/temperature, source_model_logits))

        soft_objective = tf.reduce_mean(
            -tf.reduce_sum(q * tf.log(p), reduction_indices=[1]))

        hard_objective = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=input_labels))

        lambda_ = param['lambda']
        soft_objective = tf.scalar_mul(lambda_, soft_objective)
        hard_objective = tf.scalar_mul((1 - lambda_), hard_objective)

        loss = soft_objective + hard_objective

        ops['loss'] = loss

        ops['cluster_label'] = []
        ops['cluster_center'] = []

        for ii, cc in enumerate(CNN.cluster_center):
            if cc is not None:
                ops['cluster_label'] += [CNN.cluster_label[ii]]
                ops['cluster_center'] += [cc]

        for ii, cc in enumerate(MLP.cluster_center):
            if cc is not None:
                ops['cluster_label'] += [MLP.cluster_label[ii]]
                ops['cluster_center'] += [cc]

        ops['embeddings'] = embedding_cnn + embedding_mlp
        ops['clustering_ops'] = clustering_ops_cnn + clustering_ops_mlp
        ops['reg_ops'] = reg_ops_cnn + reg_ops_mlp
        ops['reset_ops'] = reset_ops_cnn + reset_ops_mlp
        reg_term = tf.reduce_sum(tf.stack(ops['reg_ops']))

        # setting optimization
        global_step = tf.Variable(0.0, trainable=False)
        learn_rate = tf.train.exponential_decay(
            param['base_learn_rate'], global_step,
            param['learn_rate_decay_step'], param['learn_rate_decay_rate'],
            staircase=True)

        optimizer = tf.train.MomentumOptimizer(
            learning_rate=learn_rate,
            momentum=param['momentum'])

        # gradients from soft targets multiplied by T^2 as recommended
        grads_and_vars_soft = optimizer.compute_gradients(soft_objective)
        grads_and_vars_soft_scaled = []
        for grad, var in grads_and_vars_soft:
            if grad is None:
                grads_and_vars_soft_scaled.append((grad, var))
            else:
                grads_and_vars_soft_scaled.append(
                    (tf.scalar_mul(temperature ** 2, grad), var))

        grads_and_vars_hard = optimizer.compute_gradients(hard_objective)

        grads_and_vars = list(
            set(grads_and_vars_soft_scaled + grads_and_vars_hard))

        ops['train_step'] = optimizer.apply_gradients(
            grads_and_vars, global_step=global_step)

    return ops
