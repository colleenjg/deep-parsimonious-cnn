"""record_raw_data.py

This script exports a pandas dataframe containing all the CIFAR10 image data
with labels, in which each image is one dimension. It also generates a shuffled
version in which the data for all images is shuffled, except labels.

It can be run with any of the CIFAR10 models (baseline, distilled, etc.) to
retrieve the same dataset.

Choose directory to save in.

Usage:
  record_raw_data.py <exp_id>
"""
import numpy as np
import exp_config as cg
import pandas as pd

from docopt import docopt
from CIFAR_input import read_CIFAR10

EPS = 1.0e-16


def main():
    # get exp parameters
    args = docopt(__doc__)
    param = getattr(cg, args['<exp_id>'])()

    # read data from file
    if param['dataset_name'] == 'CIFAR10':
        input_data = read_CIFAR10(param['data_folder'])

        # # print the shape of the CIFAR10 image data
        # print(input_data['train_img'].shape)
        # print(input_data['train_label'].shape)
        # print(input_data['test_img'].shape)
        # print(input_data['test_label'].shape)
        # print(input_data['val_img'].shape)
        # print(input_data['val_label'].shape)

        # get nbr of images in each dataset
        train_nbr = input_data['train_img'].shape[0]
        test_nbr = input_data['test_img'].shape[0]
        val_nbr = input_data['val_img'].shape[0]
        total_nbr = train_nbr + test_nbr + val_nbr

        # get total nbr of parameters per image (same for all three sets)
        img_param = (input_data['train_img'].shape[1] *
                     input_data['test_img'].shape[2] *
                     input_data['val_img'].shape[3])

        # reshape 4D (images, x (32), y (32), channels (3)) of CIFAR10 image
        # data  into 2D for all 3 sets (training, test and validation).
        # Order 'C': last axis index changes fastest, so images are kept
        # separate
        train_img = input_data['train_img'].reshape((train_nbr, -1), order='C')
        test_img = input_data['test_img'].reshape((test_nbr, -1), order='C')
        val_img = input_data['val_img'].reshape((val_nbr, -1), order='C')

        # concatenate all data for all 3 sets, and labels for all 3 sets
        all_img = np.concatenate((train_img, test_img, val_img), axis=0)
        all_label = np.concatenate((input_data['train_label'],
                                    input_data['test_label'],
                                    input_data['val_label']), axis=0)

        # convert labels to int64 and to 2D array
        all_label = all_label.astype(np.int64)
        all_label = all_label.reshape((-1, 1))

        # make an array that will serve as an index to completely shuffle
        # all values (except labels) in the CIFAR10 combined dataset
        vector = np.arange(total_nbr * img_param)
        vector_shuffled = np.random.permutation(vector)
        array_shuffled = vector_shuffled.reshape(all_img.shape)
        all_img_shuffled = np.empty(all_img.shape)

        print('Shuffling array generated. ' +
              'Please wait while image data is shuffled (inefficient).')

        # shuffle all values (except labels) in the CIFAR10 combined dataset
        for x in xrange(total_nbr):
            for y in xrange(img_param):
                index = array_shuffled[x][y]
                index_i = np.int(np.trunc(index/img_param))
                index_j = np.int(index - index_i*img_param)
                all_img_shuffled[index_i][index_j] = all_img[x][y]

        # add labels to image data and shuffled image data
        all_info = np.concatenate((all_label, all_img), axis=1)
        all_info_shuffled = np.concatenate((all_label, all_img_shuffled),
                                           axis=1)

        print('Image data shuffled.')

        # make sure everything is in int64 to reduce file size
        all_info = all_info.astype(np.int64)
        all_info_shuffled = all_info_shuffled.astype(np.int64)

        # create dataframes
        my_df = pd.DataFrame(all_info)
        my_df_shuffled = pd.DataFrame(all_info_shuffled)

        # export dataframes to run tsne, for example.
        # file sizes are pretty big (700 MB)
        # choose directory to save in
        my_df.to_csv('../cifar_raw_data_modified/CIFAR10_images_labels.txt',
                     index=False, header=False)
        my_df_shuffled.to_csv('../cifar_raw_data_modified/CIFAR10_images_shuffled_labels.txt',
                              index=False, header=False)
        print('Files with images and labels generated.')


if __name__ == '__main__':
    main()
