"""record_for_tsne.py

Usage:
  record_for_tsne.py <exp_id>
"""
import os
import time
import math
import random
import numpy as np
import tensorflow as tf
import nn_cell_lib as nn
import exp_config as cg
import pandas as pd

from docopt import docopt
from mini_batch_iter import MiniBatchIterator
from CIFAR_input import read_CIFAR10, read_CIFAR100
from CIFAR_models import baseline_model, clustering_model, distilled_model
from scipy.spatial import distance

EPS = 1.0e-16

def main():
    # get exp parameters
    args = docopt(__doc__)
    param = getattr(cg, args['<exp_id>'])()

    # read data from file
    if param['dataset_name'] == 'CIFAR10':
        input_data = read_CIFAR10(param['data_folder'])

        # print(input_data['train_img'].shape)
        # print(input_data['train_label'].shape)
        # print(input_data['test_img'].shape)
        # print(input_data['test_label'].shape)
        # print(input_data['val_img'].shape)
        # print(input_data['val_label'].shape)

        train_img = input_data['train_img'].reshape((45000,-1), order='C')
        test_img = input_data['test_img'].reshape((10000,-1), order='C')
        val_img = input_data['val_img'].reshape((5000,-1), order='C')
        all_img = np.concatenate((train_img, test_img, val_img), axis=0)
        all_label = np.concatenate((input_data['train_label'], input_data['test_label'], input_data['val_label']), axis=0)
        all_label = all_label.astype(np.int64)
        all_label = all_label.reshape((-1,1))

        # Make a random array for permutation
        vector = np.arange(60000*3072)
        vector_shuffled = np.random.permutation(vector)
        array_shuffled = vector_shuffled.reshape((60000,3072))
        all_img_shuffled = np.empty([60000,3072])

        for x in xrange(60000):
            for y in xrange(3072):
                index = array_shuffled[x][y]
                index_i = np.int(np.trunc(index/3072))
                index_j = np.int(index - index_i*3072)
                all_img_shuffled[index_i][index_j] = all_img[x][y]

        print(all_img)
        print(all_img_shuffled)
        print(all_img_shuffled.shape)

        all_info = np.concatenate((all_label, all_img), axis=1)
        all_info_shuffled = np.concatenate((all_label, all_img_shuffled), axis=1)
        print(all_info.shape)
        print(all_info_shuffled.shape)
        my_df = pd.DataFrame(all_info)
        my_df_shuffled = pd.DataFrame(all_info_shuffled)
        print(my_df.head())
        print(my_df_shuffled.head())
        my_df.to_csv('CIFAR10_images_labels.txt', index=False, header=False)
        my_df_shuffled.to_csv('CIFAR10_images_shuffled_labels.txt', index=False, header=False)
        print('File with images and labels generated')

    # elif param['dataset_name'] == 'CIFAR100':
    #     input_data = read_CIFAR100(param['data_folder'])
    # else:
    #     raise ValueError('Unsupported dataset name!')
    # print 'Reading data done!'
    #
    # # build model
    # test_op_names = ['embeddings']
    #
    # if param['model_name'] == 'baseline':
    #     model_ops = baseline_model(param)
    # elif param['model_name'] == 'parsimonious':
    #     model_ops = clustering_model(param)
    # elif param['model_name'] == 'distilled':
    #     with tf.variable_scope('dist') as dist_var_scope:
    #         model_ops = distilled_model(param)
    # else:
    #     raise ValueError('Unsupported model name!')
    #
    # test_ops = [model_ops[i] for i in test_op_names]
    # print 'Building model done!'
    #
    # # run model
    # input_data['train_img'] = np.concatenate(
    #     [input_data['train_img'], input_data['val_img']], axis=0)
    # input_data['train_label'] = np.concatenate(
    #     [input_data['train_label'], input_data['val_label']])
    #
    # num_train_img = input_data['train_img'].shape[0]
    # max_test_iter = int(math.ceil(num_train_img / param['bat_size']))
    # test_iterator = MiniBatchIterator(
    #     idx_start=0, bat_size=param['bat_size'], num_sample=num_train_img,
    #     train_phase=False, is_permute=False)
    #
    # config = tf.ConfigProto(allow_soft_placement=True)
    # sess = tf.Session(config=config)
    # saver = tf.train.Saver()
    # saver.restore(sess, os.path.join(
    #     param['test_folder'], param['test_model_name']))
    # print 'Graph initialization done!'
    #
    # if param['model_name'] == 'parsimonious':
    #     param['num_layer_cnn'] = len(
    #         [xx for xx in param['num_cluster_cnn'] if xx])
    #     param['num_layer_mlp'] = len(
    #         [xx for xx in param['num_cluster_mlp'] if xx])
    #     num_layer_reg = param['num_layer_cnn'] + param['num_layer_mlp']
    #
    #     cluster_center = sess.run(model_ops['cluster_center'])
    #
    # else:
    #     # num_layer_cnn = len(param['num_cluster_cnn'])
    #     # num_layer_mlp = len(param['num_cluster_mlp'])
    #     # num_layer_reg = num_layer_cnn + num_layer_mlp
    #     num_layer_reg = 5
    #     cluster_center = [None] * num_layer_reg
    #
    # embeddings = [[] for _ in xrange(num_layer_reg)]
    #
    # ## Initialized a fixed size corresponding to batch size of 100
    # labels = np.zeros(100)
    #
    # for test_iter in xrange(max_test_iter):
    #     idx_bat = test_iterator.get_batch()
    #
    #     bat_imgs = (input_data['train_img'][idx_bat, :, :, :].astype(
    #         np.float32) - input_data['mean_img']) / 255.0
    #
    #     # Added to list labels per batch
    #     bat_labels = input_data['train_label'][idx_bat].astype(np.int32)
    #
    #     if test_iter == 0:
    #         labels = bat_labels
    #     else:
    #         labels = np.append(labels, bat_labels)
    #
    #     feed_data = {model_ops['input_images']: bat_imgs}
    #
    #     results = sess.run(test_ops, feed_dict=feed_data)
    #
    #     test_results = {}
    #     for res, name in zip(results, test_op_names):
    #         test_results[name] = res
    #
    #     for ii, ee in enumerate(test_results['embeddings']):
    #         if ii < 3:
    #             continue
    #
    #         embeddings[ii] += [ee]
    #
    # for ii in xrange(num_layer_reg):
    #     if ii < 3:
    #         continue
    #
    #     embeddings[ii] = np.concatenate(embeddings[ii], axis=0)
    #     labels = labels.astype(np.int64)
    #     labels = labels.reshape((-1,1))
    #     embeddings_labelled = np.concatenate((labels, embeddings[ii]), axis = 1)
    #     my_df = pd.DataFrame(embeddings_labelled)
    #     layer = ii+1
    #     my_df.to_csv('Activations_layer_%d_distilled_sample_clustering_model_snap4.txt' % layer, index=False, header=False)
    #     print('File with activations and labels generated')


    # sess.close()

if __name__ == '__main__':
    main()
