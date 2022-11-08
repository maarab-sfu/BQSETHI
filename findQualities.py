from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


## Imports
import numpy as np

import tensorflow as tf
import logging
##logging.getLogger("tensorflow").setLevel(logging.ERROR)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.io as spio
import math
##import pickle
##from sklearn import preprocessing
##from scipy.misc import imresize
##from sklearn.decomposition import PCA
##from sklearn.model_selection import train_test_split
from tensorflow.contrib.learn.python.learn import monitors as monitor_lib
from enum import Enum

tf.logging.set_verbosity(tf.logging.INFO)

def convbnrelu3d(inputs, filters, kernel_size, strides, padding, initializer, regularizer, mode):
    convolution = tf.layers.conv3d(
        inputs = inputs,
        filters = filters,
        kernel_size = kernel_size,
        strides = strides,
        padding = padding,
        kernel_initializer=initializer,
        kernel_regularizer=regularizer,
        activation=None)
##    bn = tf.layers.batch_normalization(convolution, axis=-1, center=True, scale=True, training=(mode == tf.estimator.ModeKeys.TRAIN))
    return tf.nn.relu(convolution)

def convbnrelu2d(inputs, filters, kernel_size, padding, initializer, regularizer, mode):
    convolution = tf.layers.conv2d(
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        padding=padding,
        kernel_initializer=initializer,
        kernel_regularizer=regularizer,
        activation=None)
##    bn = tf.layers.batch_normalization(convolution, axis=-1, center=True, scale=True, training=(mode == tf.estimator.ModeKeys.TRAIN))
    return tf.nn.relu(convolution)


def convsumbnrelu3d(inputs, relu, filters, kernel_size, strides, padding, initializer, regularizer, mode):
    convolution = tf.layers.conv3d(
        inputs = inputs,
        filters = filters,
        kernel_size = kernel_size,
        strides = strides,
        padding = padding,
        kernel_initializer=initializer,
        kernel_regularizer=regularizer,
        activation=None)
    summation = relu + convolution
##    bn = tf.layers.batch_normalization(summation, axis=-1, center=True, scale=True, training=(mode == tf.estimator.ModeKeys.TRAIN))
    return (tf.nn.relu(convolution), summation)

def convsumbnrelu2d(inputs, relu, filters, kernel_size, padding, initializer, regularizer, mode):
    convolution = tf.layers.conv2d(
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        padding=padding,
        kernel_initializer=initializer,
        kernel_regularizer=regularizer,
        activation=None)

    summation = convolution + relu
##    bn = tf.layers.batch_normalization(summation, axis=-1, center=True, scale=True, training=(mode == tf.estimator.ModeKeys.TRAIN))
    return (tf.nn.relu(convolution), summation)

def SSRN_model_fn(features, labels, mode):

    regularizer = tf.contrib.layers.l2_regularizer(scale = 0.0000)
    initializer = tf.contrib.layers.xavier_initializer(uniform = False) #tf.random_normal_initializer(0, 0.1) #glorot_normal_initializer()

    input_layer_spectral = tf.transpose(features["x"], perm= [0, 3, 1, 2])
    input_layer_spectral = tf.reshape(input_layer_spectral, [-1, 200, 7, 7, 1])
    input_layer_spectral = tf.cast(input_layer_spectral, tf.float32)
    conv1 = convbnrelu3d(input_layer_spectral, 24, [7, 1, 1], (2, 1, 1), "valid", initializer, regularizer, mode)
    conv2 = convbnrelu3d(conv1, 24, [7, 1, 1], (1, 1, 1), "same", initializer, regularizer, mode)
    (conv3, summation1) = convsumbnrelu3d(conv2, conv1, 24, [7, 1, 1], (1, 1, 1), "same", initializer, regularizer, mode)
    conv4 = convbnrelu3d(conv3, 24, [7, 1, 1], (1, 1, 1), "same", initializer, regularizer, mode)
    (conv5, summation2) = convsumbnrelu3d(conv4, summation1, 24, [7, 1, 1], (1, 1, 1), "same", initializer, regularizer, mode)
    bn5 = tf.layers.batch_normalization(conv5, axis = -1, center = True, scale = True, training = (mode == tf.estimator.ModeKeys.TRAIN))
    relu5 = tf.nn.relu(bn5)

    conv6 = convbnrelu3d(relu5, 128, [97, 1, 1], (1, 1, 1), "valid", initializer, regularizer, mode)

    ## start of spatial block
    input_layer_spatial = tf.reshape(conv6, [-1, 7, 7, 128])
    conv7 = convbnrelu2d(input_layer_spatial, 24, [3, 3], "valid", initializer, regularizer, mode)
    conv8 = convbnrelu2d(conv7, 24, [3, 3], "same", initializer, regularizer, mode)
    (conv9, summation3) = convsumbnrelu2d(conv8, conv7, 24, [3, 3], "same", initializer, regularizer, mode)
    conv10 = convbnrelu2d(conv9, 24, [3, 3], "same", initializer, regularizer, mode)
    (conv11, summation4) = convsumbnrelu2d(conv10, summation3, 24, [3, 3], "same", initializer, regularizer, mode)

    bn12 = tf.layers.batch_normalization(conv11, axis = -1, center = True, scale = True, training = (mode == tf.estimator.ModeKeys.TRAIN))
    relu12 = tf.nn.relu(bn12)

    pool1 = tf.layers.average_pooling2d(inputs = relu12, pool_size = [5, 5], strides = 1)
    pool1_flat = tf.reshape(pool1, [-1, 24]) #flatten

    dropout = tf.layers.dropout(inputs = pool1_flat, rate = 0.5, training = (mode == tf.estimator.ModeKeys.TRAIN))

    logits = tf.layers.dense(inputs = dropout, units = 16, activation = None, kernel_initializer = initializer)

    ## End of network design

    predictions = {
        ## Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input = logits, axis = 1),
        ## Add `softmax_tensor` to the graph. It is used for PREDICT and by the`logging_hook`.
        "probabilities": tf.nn.softmax(logits, name = "softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode = mode, predictions = predictions)

    ## Calculate Loss (for both TRAIN and EVAL modes)
    onehot_labels = tf.one_hot(indices = tf.cast(labels, tf.int32), depth = 16)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels = onehot_labels, logits = logits, reduction = tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)+ tf.losses.get_regularization_loss()
    #loss = tf.identity(loss, name="loss")


    accuracy_train, update_op1 = tf.metrics.accuracy(labels = labels, predictions = predictions["classes"], name = "my_metric")
    #accuracy_train = tf.identity(accuracy_train, name="accuracy_train")

    ## Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels = labels, predictions = predictions["classes"])
    }

    ## Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:

	    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	    running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope = "my_metric")
	    running_vars_initializer = tf.variables_initializer(var_list = running_vars)
		

	    Accuracy_train = tf.summary.scalar('accuracy_train', update_op1)
	    Loss_train = tf.summary.scalar('loss_train', loss)

	    summary_hook = tf.train.SummarySaverHook(
		    save_steps  = 128,
		    output_dir = "IN_model_ReducedClasses/summary/train",
		    summary_op = [Loss_train, Accuracy_train])

	    logging_hook = tf.train.LoggingTensorHook({"reset":running_vars_initializer, "loss_train": loss,  "accuracy_train":update_op1}, every_n_iter = 128)


	    optimizer = tf.train.RMSPropOptimizer(learning_rate = 0.0003)
	    with tf.control_dependencies(update_ops):
	        train_op = optimizer.minimize(loss = loss, global_step = tf.train.get_global_step())
	    return tf.estimator.EstimatorSpec(mode = mode, loss = loss, train_op = train_op, training_hooks = [summary_hook, logging_hook])


    return tf.estimator.EstimatorSpec(
        mode = mode, loss = loss, eval_metric_ops = eval_metric_ops)



mat = spio.loadmat('data/IN/selectedLabels/X_test_original.mat', squeeze_me = True)                                                                                         
X_test_original = mat['X_test_original']

mat = spio.loadmat('data/IN/selectedLabels/X_test_high.mat', squeeze_me = True)                                                                                         
X_test_high = mat['X_test_high']

mat = spio.loadmat('data/IN/selectedLabels/X_test_medium.mat', squeeze_me = True)                                                                                         
X_test_medium = mat['X_test_medium']

mat = spio.loadmat('data/IN/selectedLabels/X_test_low.mat', squeeze_me = True)                                                                                         
X_test_low = mat['X_test_low']

mat = spio.loadmat('data/IN/selectedLabels/X_test_verylow.mat', squeeze_me = True)                                                                                         
X_test_verylow = mat['X_test_verylow']

mat = spio.loadmat('data/IN/selectedLabels/y_test.mat', squeeze_me = True)                                                                                         
y_test = mat['y_test']

mat = spio.loadmat('data/IN/selectedLabels/indices.mat', squeeze_me = True)                                                                                         
indices = mat['indices']

class NDSparseMatrix:
    def __init__(self):
        self.elements = {}

    def addValue(self, tuple, value):
        self.elements[tuple] = value

    def readValue(self, tuple):
        try:
            value = self.elements[tuple]
        except KeyError:
            # could also be 0.0 if using floats...
            value = 0.0
        return value

loss_obj = NDSparseMatrix()

def findBracket(theList):
    for i in range(len(theList)):
        if(theList[i] == "["):
            print(i)
            return i
    return -1

def testNode(C, quality, isUpperBound = False): # "quality" is an array of 5 numbers
    if (isUpperBound == False):
        newQuality = np.cumsum(quality)
##        print("&&&&&&&&&&&&& This is the actual loss")
    else:
        newQuality = quality
##        print("&&&&&&&&&&&&& This is just an upper bound for loss")

##    print("\n%%%%%%%%%%%%%%%%%%%% The cumulative quality is: ", newQuality)
    
    currentLoss = loss_obj.readValue((1,2,3))
    if(currentLoss != 0):
        return loss_obj.readValue(qualityIndex)
    
    X_test = np.zeros(X_test_original.shape)
##    print("quality shape: ", newQuality[0].shape)
    if(newQuality[0] != 0):
        for l in range(newQuality[0]):                                                                                                
            X_test[:,:,:,indices[l]] = X_test_original[:,:,:,indices[l]]
    if(newQuality[1] != newQuality[0]):
        for l in range(newQuality[1]-newQuality[0]):
            X_test[:,:,:,indices[l+newQuality[0]]] = X_test_high[:,:,:,indices[l+newQuality[0]]]
    if(newQuality[2] != newQuality[1]):
        for l in range(newQuality[2]-newQuality[1]):
            X_test[:,:,:,indices[l+newQuality[1]]] = X_test_medium[:,:,:,indices[l+newQuality[1]]]
    if(newQuality[3] != newQuality[2]):
        for l in range(newQuality[3]-newQuality[2]):
            X_test[:,:,:,indices[l+newQuality[2]]] = X_test_low[:,:,:,indices[l+newQuality[2]]]
    if(newQuality[4] != newQuality[3]):
        for l in range(newQuality[4]-newQuality[3]):
            X_test[:,:,:,indices[l+newQuality[3]]] = X_test_verylow[:,:,:,indices[l+newQuality[3]]]

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x = {"x": X_test},
        y = y_test,
        batch_size = 16,
        num_epochs = 1,
        shuffle = False)
    
    eval_results = C.evaluate(input_fn = eval_input_fn, hooks = None)
    loss_obj.addValue(tuple(newQuality), eval_results['loss'])
    return eval_results["accuracy"]
    
def elimSpaces(myList):
    return [value for value in myList if value != '']
def main():
    
    fname = "IN_ReducedClasses.txt"
    with open(fname) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    qualities = np.zeros(5)
    num = len(content)
    max_data = np.zeros(num)
    acc = np.zeros(num)
    numOfIter = np.zeros(num)
    
    allList = content[0].rsplit(' ')
    print(elimSpaces(allList))

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config = config)
    
    ## Create the Estimator
    SSRN_classifier = tf.estimator.Estimator(model_fn=SSRN_model_fn, model_dir = "IN_model_ReducedClasses/", config = tf.contrib.learn.RunConfig(
            save_checkpoints_steps = 128,
            save_checkpoints_secs = None,
            keep_checkpoint_max = 201,
            save_summary_steps = 500))
    
    for i in range(num):
        myList = content[i].rsplit(' ')
        myList = elimSpaces(myList)
        brackInd = findBracket(myList)
        qualities[0] = int(myList[brackInd+1])
        qualities[1] = int(myList[brackInd+2])
        qualities[2] = int(myList[brackInd+3])
        qualities[3] = int(myList[brackInd+4])
        myString = myList[brackInd+5]
        qualities[4] = int(myString[:-1])
        qualities = qualities.astype(int)
        print(qualities)
        acc[i] = testNode(SSRN_classifier, qualities)
        max_data[i] = myList[0]
    
    x = max_data.argsort()
    max_data = max_data[x]
    acc = acc[x]
##    numOfIter = numOfIter[x]
##
    max_data, x = np.unique(max_data, return_index = True)
    acc = acc[x]
##    numOfIter = numOfIter[x]
##
####    max_data = np.delete(max_data, 17)
####    max_data = np.delete(max_data, 17)
####
####    acc = np.delete(acc, 17)
####    acc = np.delete(acc, 17)
####
####    numOfIter = np.delete(numOfIter, 17)
####    numOfIter = np.delete(numOfIter, 17)
##    
    print(max_data, acc, numOfIter)
####    print(np.where(max_data == 599400))
    spio.savemat('IN_ReducedClasses_maxData.mat', {"max_data_RC" : max_data})
    spio.savemat('IN_ReducedClasses_acc.mat', {"acc_RC" : acc})
##    spio.savemat('UP_numOfIter.mat', {"numOfIter_RC" : numOfIter})


if __name__ == "__main__":
    main()
