from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import os
import numpy as np
from tensorflow.contrib.tensorboard.plugins import projector

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
FLAGS = None

def test():
    words = np.zeros((61188, 61188), dtype=np.uint32)
    with open('/home/ywryoo/20news-bydate/matlab/train.data', 'r') as f:
        cursor = 0
        coocur = []
        for i,line in enumerate(f):
            if(i%10000 == 0):
                print("{}th line".format(i))
            if(i%100000 == 0):
                print(words)
            [article, word, weight] = [int(x) for x in line.split()]
            if cursor < article:
                print("article is chaged to {}".format(article))
                for i, ival in enumerate(coocur):
                    for j, jval in enumerate(coocur):
                        if ival[0] != jval[0]:
                            words[ival[0]][jval[0]] += (ival[1]*jval[1])
                cursor = article
                coocur.clear()
            coocur.append((word,weight))
        for i, ival in enumerate(coocur):
            for j, jval in enumerate(coocur):
                if ival[0] != jval[0]:
                    words[ival[0]][jval[0]] += (ival[1]*jval[1])
    print(words)
    with open('/home/ywryoo/20news-bydate/matlab/words', 'w') as f:
        for (x, y), element in np.ndenumerate(words):
            if x%1000 == 0:
                print("{}th row is processing".format(x))
            if(element != 0):
                f.write("{} {} {}\n".format(x, y, element))

def generate_embeddings():
    # Import data

    with open("/home/ywryoo/20news-bydate/matlab/embedded", "r") as f:
        [_, datadim] = [int(i) for i in f.readline()[:-1].split(' ')]
        data = np.zeros((61188, datadim),dtype=np.float32)
        for _, line in enumerate(f):
            row = line.split()
            data[int(row[0])-1,:]=[float(x) for x in row[1:]]
    
    documents = np.zeros((11269,100),dtype=np.float32)
    with open('/home/ywryoo/20news-bydate/matlab/train.data', 'r') as f:
        cursor = 1
        coocur = []
        weightsum = 0
        for i,line in enumerate(f):
            if(i%10000 == 0):
                print("{}th line".format(i))
            [article, word, weight] = [int(x) for x in line.split()]
            if cursor < article:
                print("article is chaged to {}".format(article))
                if(len(coocur) != 0):
                    vector = np.array([data[i[0]-1]*i[1] for i in coocur])
                    documents[cursor-1,:] = np.sum(vector, axis=0)/weightsum
                cursor = article
                coocur.clear()
                weightsum = 0
            coocur.append((word,weight))
            weightsum += weight
        if(len(coocur) != 0):
            vector = np.array([data[i[0]-1]*i[1] for i in coocur])
            documents[cursor-1,:] = np.sum(vector, axis=0)/weightsum
    
    sess = tf.InteractiveSession()

    # Input set for Embedded TensorBoard visualization
    # Performed with cpu to conserve memory and processing power
    with tf.device("/cpu:0"):
        embedding = tf.Variable(tf.stack(documents, axis=0), trainable=False, name='embedding')

    tf.global_variables_initializer().run()

    saver = tf.train.Saver()
    writer = tf.summary.FileWriter(FLAGS.log_dir + '/projector', sess.graph)

    # Add embedding tensorboard visualization. Need tensorflow version
    # >= 0.12.0RC0
    config = projector.ProjectorConfig()
    embed= config.embeddings.add()
    embed.tensor_name = 'embedding:0'
    embed.metadata_path = os.path.join(FLAGS.log_dir + '/projector/metadata.tsv')

    projector.visualize_embeddings(writer, config)

    saver.save(sess, os.path.join(
        FLAGS.log_dir, 'projector/a_model.ckpt'), global_step=FLAGS.max_steps)

def generate_metadata_file():
    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir,
                                      one_hot=True,
                                      fake_data=FLAGS.fake_data)
    def save_metadata(file):
        with open(file, 'w') as f:
            for i in range(FLAGS.max_steps):
                c = np.nonzero(mnist.test.labels[::1])[1:][0][i]
                f.write('{}\n'.format(c))

    save_metadata(FLAGS.log_dir + '/projector/metadata.tsv')

def main(_):
    if tf.gfile.Exists(FLAGS.log_dir + '/projector'):
        tf.gfile.DeleteRecursively(FLAGS.log_dir + '/projector')
        tf.gfile.MkDir(FLAGS.log_dir + '/projector')
    tf.gfile.MakeDirs(FLAGS.log_dir  + '/projector') # fix the directory to be created
    #generate_metadata_file()
    generate_embeddings()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fake_data', nargs='?', const=True, type=bool,
                        default=False,
                        help='If true, uses fake data for unit testing.')
    parser.add_argument('--max_steps', type=int, default=10000,
                        help='Number of steps to run trainer.')
    parser.add_argument('--data_dir', type=str, default='/home/ywryoo/20ng/data',
                        help='Directory for storing input data')
    parser.add_argument('--log_dir', type=str, default='/home/ywryoo/20ng/logs',
                        help='Summaries log directory')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)


"""
    with open("/home/ywryoo/20news-bydate/matlab/vocabulary.txt", "r") as f:
        vocab = f.read().split('\n')[:-1]

    with open("/home/ywryoo/20news-bydate/matlab/embedded", "r") as f:
        [datalen, datadim] = [int(i) for i in f.readline()[:-1].split(' ')]
        data = np.zeros((datalen, datadim),dtype=np.float32)
        with open(FLAGS.log_dir + '/projector/metadata.tsv', 'w') as ff:
            for i, line in enumerate(f):
                row = line.split()
                ff.write("{}\n".format(vocab[int(row[0])]))
                data[i,:]=[float(x) for x in row[1:]]
"""