"""run_distill_model.py

Usage:
  run_distill_model.py <distilled_model_id> <cumbersome_model_id> <lambda> <temperature>
"""
import exp_config as cg
import math
import numpy as np
import os
import tensorflow as tf
import time
import nn_cell_lib as nn
import cPickle as pickle

from CIFAR_input import read_CIFAR10, read_CIFAR100
from CIFAR_models import baseline_model, clustering_model, distilled_model
from docopt import docopt
from mini_batch_iter import MiniBatchIterator


def main():
    args = docopt(__doc__)
    lambda_ = args['<lambda>']
    temperature = args['<temperature>']
    param = getattr(cg, args['<distilled_model_id>'])(
        lambda_=float(lambda_), temperature=float(temperature))

    if param['resume_training']:
        param['exp_id'] = param['resume_exp_id']
    else:
        param['exp_id'] = args['<distilled_model_id>'] + '_l' \
                          + lambda_.replace('.', '-') + '_t' + temperature \
                          + '_' + time.strftime("%Y-%b-%d-%H-%M-%S")

    param['save_folder'] = os.path.join(param['save_path'], param['exp_id'])
    param_cumb = getattr(cg, args['<cumbersome_model_id>'])()

    # read data from file
    param['denom_const'] = 255.0
    if param['dataset_name'] == 'CIFAR10':
        input_data = read_CIFAR10(param['data_folder'])
    else:
        input_data = read_CIFAR100(param['data_folder'])
    print 'Reading data done!'

    if param['dataset_name'] != param_cumb['dataset_name']:
        raise ValueError(
            'Distilled model must use same dataset as source model')

    if param['dataset_name'] not in ['CIFAR10', 'CIFAR100']:
        raise ValueError('Unsupported dataset name!')

    # save parameters
    if not os.path.isdir(param['save_folder']):
        os.mkdir(param['save_folder'])

    with open(os.path.join(param['save_folder'], 'hyper_param.txt'), 'w') as f:
        for key, value in param.iteritems():
            f.write('{}: {}\n'.format(key, value))

    # build cumbersome model
    if param_cumb['model_name'] == 'baseline':
        cumb_model_ops = baseline_model(param_cumb)
    elif param_cumb['model_name'] == 'parsimonious':
        cumb_model_ops = clustering_model(param_cumb)
    else:
        raise ValueError('Unsupported cumbersome model')
    cumb_op_names = ['logits']
    cumb_ops = [cumb_model_ops[i] for i in cumb_op_names]
    cumb_vars = tf.global_variables()
    print 'Rebuilding cumbersome model done!'

    # restore session of cumbersome model
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    saver_cumb = tf.train.Saver(var_list=cumb_vars)
    saver_cumb.restore(sess, os.path.join(
        param_cumb['test_folder'], param_cumb['test_model_name']))
    print 'Restoring cumbersome model done!'

    # build distilled model
    with tf.variable_scope('dist') as dist_var_scope:
        model_ops = distilled_model(param)

    # initiate session for new distilled model
    dist_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='dist')
    sess.run(tf.variables_initializer(dist_vars))

    saver = tf.train.Saver(var_list=dist_vars)

    train_op_names = ['train_step', 'loss']
    val_op_names = ['scaled_logits']
    train_ops = [model_ops[i] for i in train_op_names]
    val_ops = [model_ops[i] for i in val_op_names]
    print 'Building new model done!\n'

    num_train_img = input_data['train_img'].shape[0]
    num_val_img = input_data['test_img'].shape[0]
    max_val_iter = int(math.ceil(num_val_img / param['bat_size']))
    train_iterator = MiniBatchIterator(
        idx_start=0, bat_size=param['bat_size'], num_sample=num_train_img,
        train_phase=True, is_permute=True)
    val_iterator = MiniBatchIterator(
        idx_start=0, bat_size=param['bat_size'], num_sample=num_val_img,
        train_phase=False, is_permute=False)

    train_iter_start = 0

    for train_iter in xrange(train_iter_start, param['max_train_iter']):
        # generate a batch
        idx_train_bat = train_iterator.get_batch()

        bat_imgs = (input_data['train_img'][idx_train_bat, :, :, :].astype(
            np.float32) - input_data['mean_img']) / param['denom_const']
        bat_labels = input_data['train_label'][idx_train_bat].astype(np.int32)

        feed_data = {
            cumb_model_ops['input_images']: bat_imgs,
            cumb_model_ops['input_labels']: bat_labels
        }

        # get logits from cumbersome model
        source_model_logits = sess.run(
            cumb_model_ops['logits'], feed_dict=feed_data)

        feed_data = {
            model_ops['input_images']: bat_imgs,
            model_ops['input_labels']: bat_labels,
            model_ops['source_model_logits']: source_model_logits
        }

        # with tf.variable_scope(dist_var_scope):
        results = sess.run(train_ops, feed_dict=feed_data)

        train_results = {}
        for res, name in zip(results, train_op_names):
            train_results[name] = res

        loss = train_results['loss']

        # display statistic
        if (train_iter + 1) % param['disp_iter'] == 0 or train_iter == 0:
            disp_str = 'Train Step = {:06d} || CE loss = {:e}'.format(
                train_iter + 1, loss)

            print disp_str

        # valid model
        if (train_iter + 1) % param['valid_iter'] == 0 or train_iter == 0:
            num_correct = 0.0

            if param['resume_training'] == True:
                print 'Resume Exp ID = {}'.format(param['exp_id'])
            else:
                print 'Exp ID = {}'.format(param['exp_id'])

            for val_iter in xrange(max_val_iter):
                idx_val_bat = val_iterator.get_batch()

                bat_imgs = (input_data['test_img'][idx_val_bat, :, :, :].astype(
                    np.float32) - input_data['mean_img']) / param['denom_const']
                bat_labels = input_data['test_label'][
                    idx_val_bat].astype(np.int32)

                feed_data[model_ops['input_images']] = bat_imgs
                feed_data[model_ops['input_labels']] = bat_labels

                results = sess.run(val_ops, feed_dict=feed_data)

                val_results = {}
                for res, name in zip(results, val_op_names):
                    val_results[name] = res

                pred_label = np.argmax(val_results['scaled_logits'], axis=1)
                num_correct += np.sum(np.equal(pred_label,
                                               bat_labels).astype(np.float32))

            val_acc = (num_correct / num_val_img)
            print "Val accuracy = {:3f}".format(val_acc * 100)

        # snapshot a model
        if (train_iter + 1) % param['save_iter'] == 0:
            saver.save(sess, os.path.join(param['save_folder'], '{}_snapshot_{:07d}.ckpt'.format(
                param['model_name'], train_iter + 1)))


if __name__ == '__main__':
    main()
