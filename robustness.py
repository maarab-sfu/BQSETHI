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
    input_layer_spectral = tf.reshape(input_layer_spectral, [-1, 176, 7, 7, 1])
    input_layer_spectral = tf.cast(input_layer_spectral, tf.float32)
    conv1 = convbnrelu3d(input_layer_spectral, 16, [7, 1, 1], (2, 1, 1), "valid", initializer, regularizer, mode)
    conv2 = convbnrelu3d(conv1, 16, [7, 1, 1], (1, 1, 1), "same", initializer, regularizer, mode)
    (conv3, summation1) = convsumbnrelu3d(conv2, conv1, 16, [7, 1, 1], (1, 1, 1), "same", initializer, regularizer, mode)
    conv4 = convbnrelu3d(conv3, 16, [7, 1, 1], (1, 1, 1), "same", initializer, regularizer, mode)
    (conv5, summation2) = convsumbnrelu3d(conv4, summation1, 16, [7, 1, 1], (1, 1, 1), "same", initializer, regularizer, mode)
    bn5 = tf.layers.batch_normalization(conv5, axis = -1, center = True, scale = True, training = (mode == tf.estimator.ModeKeys.TRAIN))
    relu5 = tf.nn.relu(bn5)

    conv6 = convbnrelu3d(relu5, 128, [85, 1, 1], (1, 1, 1), "valid", initializer, regularizer, mode)

    ## start of spatial block
    input_layer_spatial = tf.reshape(conv6, [-1, 7, 7, 128])
    conv7 = convbnrelu2d(input_layer_spatial, 16, [3, 3], "valid", initializer, regularizer, mode)
    conv8 = convbnrelu2d(conv7, 16, [3, 3], "same", initializer, regularizer, mode)
    (conv9, summation3) = convsumbnrelu2d(conv8, conv7, 16, [3, 3], "same", initializer, regularizer, mode)
    conv10 = convbnrelu2d(conv9, 16, [3, 3], "same", initializer, regularizer, mode)
    (conv11, summation4) = convsumbnrelu2d(conv10, summation3, 16, [3, 3], "same", initializer, regularizer, mode)

    bn12 = tf.layers.batch_normalization(conv11, axis = -1, center = True, scale = True, training = (mode == tf.estimator.ModeKeys.TRAIN))
    relu12 = tf.nn.relu(bn12)

    pool1 = tf.layers.average_pooling2d(inputs = relu12, pool_size = [5, 5], strides = 1)
    pool1_flat = tf.reshape(pool1, [-1, 16]) #flatten

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
		    save_steps  =128,
		    output_dir = "KSC_model/summary/train",
		    summary_op = [Loss_train, Accuracy_train])

	    logging_hook = tf.train.LoggingTensorHook({"reset":running_vars_initializer, "loss_train": loss,  "accuracy_train":update_op1}, every_n_iter = 128)


	    optimizer = tf.train.RMSPropOptimizer(learning_rate = 0.0001)
	    with tf.control_dependencies(update_ops):
	        train_op = optimizer.minimize(loss = loss, global_step = tf.train.get_global_step())
	    return tf.estimator.EstimatorSpec(mode = mode, loss = loss, train_op = train_op, training_hooks = [summary_hook, logging_hook])


    return tf.estimator.EstimatorSpec(
        mode = mode, loss = loss, eval_metric_ops = eval_metric_ops)



mat = spio.loadmat('data/KSC/allLabels/X_test_original.mat', squeeze_me = True)                                                                                         
X_test_original = mat['X_test_original']

mat = spio.loadmat('data/KSC/allLabels/X_test_high.mat', squeeze_me = True)                                                                                         
X_test_high = mat['X_test_high']

mat = spio.loadmat('data/KSC/allLabels/X_test_medium.mat', squeeze_me = True)                                                                                         
X_test_medium = mat['X_test_medium']

mat = spio.loadmat('data/KSC/allLabels/X_test_low.mat', squeeze_me = True)                                                                                         
X_test_low = mat['X_test_low']

mat = spio.loadmat('data/KSC/allLabels/X_test_verylow.mat', squeeze_me = True)                                                                                         
X_test_verylow = mat['X_test_verylow']

mat = spio.loadmat('data/KSC/allLabels/y_test.mat', squeeze_me = True)                                                                                         
y_test = mat['y_test']

mat = spio.loadmat('data/KSC/allLabels/indices.mat', squeeze_me = True)                                                                                         
indices = mat['indices']

def makeTestSet(quality):
    X_test = np.zeros(X_test_original.shape)
    if(quality[0] != 0):
        for l in range(quality[0]):                                                                                                
            X_test[:,:,:,indices[l]] = X_test_original[:,:,:,indices[l]]
    if(quality[1] != quality[0]):
        for l in range(quality[1]-quality[0]):
            X_test[:,:,:,indices[l+quality[0]]] = X_test_high[:,:,:,indices[l+quality[0]]]
    if(quality[2] != quality[1]):
        for l in range(quality[2]-quality[1]):
            X_test[:,:,:,indices[l+quality[1]]] = X_test_medium[:,:,:,indices[l+quality[1]]]
    if(quality[3] != quality[2]):
        for l in range(quality[3]-quality[2]):
            X_test[:,:,:,indices[l+quality[2]]] = X_test_low[:,:,:,indices[l+quality[2]]]
    if(quality[4] != quality[3]):
        for l in range(quality[4]-quality[3]):
            X_test[:,:,:,indices[l+quality[3]]] = X_test_verylow[:,:,:,indices[l+quality[3]]]
    return X_test


def changeTestSet(X_test, qualities, quality, number):
    X_test_modified = np.copy(X_test)
    cumulSum = 0
    for i in range(quality-1):
        cumulSum = cumulSum + qualities[i] 
    X_test_modified[:, :, :, indices[cumulSum + number]] = 0
    return X_test_modified
            
def testNode(C, X_test): # "quality" is an array of 5 numbers
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x = {"x": X_test},
        y = y_test,
        batch_size = 16,
        num_epochs = 1,
        shuffle = False)
    
    eval_results = C.evaluate(input_fn = eval_input_fn, hooks = None)
    return eval_results["accuracy"]
    
def elimSpaces(myList):
    return [value for value in myList if value != '']

def main():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config = config)

    SSRN_classifier = tf.estimator.Estimator(model_fn=SSRN_model_fn, model_dir = "KSC_model/", config = tf.contrib.learn.RunConfig(
            save_checkpoints_steps = 128,
            save_checkpoints_secs = None,
            keep_checkpoint_max = 201,
            save_summary_steps = 500))

    qualities = np.array([1, 15, 19, 41, 99])
    print(type(qualities))
    X_test_total = makeTestSet(qualities)
    print("The Initial TestSet is: ", X_test_total[100, 3, 3, :])
    
    result = np.zeros(12)
    stdresult = np.zeros(12)
    ind2 = 0
    for i in range(12):
        acc = np.zeros(30)
        ind = 0
        fname = "loss\\KSC_loss"+str(i+1)+".txt"
        with open(fname) as f:
            content = f.readlines()
        content = [x.strip() for x in content]
        ## Create the Estimator
        num = len(content)
        print("Number of entries: ", num)
        myList = content[0].rsplit(' ')
        myList = elimSpaces(myList)
        quality = int(myList[0])
        number = int(myList[1])
        X_test = np.copy(X_test_total)
        X_test = changeTestSet(X_test, qualities, quality, number)
        for i in range(num-1):
            myList = content[i+1].rsplit(' ')
            myList = elimSpaces(myList)
            print("The quality and number are: ", myList[0], myList[1])
            if ((int(myList[0]) == quality)and(int(myList[1]) == number)):
                continue
            if ((int(myList[0]) < quality)or((int(myList[0]) == 1)and(int(myList[1]) < number))):
                acc[ind] = testNode(SSRN_classifier, X_test)
                ind = ind + 1
                X_test = np.copy(X_test_total)
            quality = int(myList[0])
            number = int(myList[1])
            X_test = changeTestSet(X_test, qualities, quality, number)
##            print("Modified TestSet is: ", X_test[100, 3, 3, :])
        
        result[ind2] = np.average(acc)
        stdresult[ind2] = np.std(acc)
        ind2 = ind2 + 1
        f.close()
    print(result, stdresult)

    spio.savemat('KSC_loss.mat', {"KSC_loss" : result})
    spio.savemat('KSC_stdloss.mat', {"KSC_stdloss" : stdresult})
##    spio.savemat('UP_numOfIter.mat', {"numOfIter_RC" : numOfIter})


if __name__ == "__main__":
    main()
