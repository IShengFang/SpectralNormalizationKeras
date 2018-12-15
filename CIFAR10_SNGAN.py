import numpy as np
import matplotlib.pyplot as plt
from time import time

from keras.models import Model, Sequential
from keras.optimizers import Adam
import keras.backend as K
from keras.utils.generic_utils import Progbar

from model import *

import sys
import os
import pickle


arg_list = sys.argv
plt.switch_backend('agg') 
#Hyperperemeter
BATCHSIZE=64
LEARNING_RATE = 0.0002
TRAINING_RATIO = 1
BETA_1 = 0.0
BETA_2 = 0.9
EPOCHS = 500
BN_MIMENTUM = 0.9
BN_EPSILON  = 0.00002
LEAK = 0.1
LOSS = 'hinge' #Or
#LOSS = 'wasserstein' #Or
#LOSS = 'binary_crossentropy'

if arg_list[1].lower == "dcgan":
    RESNET = False #for DCGAN
elif arg_list[1].lower == "resnet":
    RESNET = True #for DCGAN
else:
    RESNET = True

if arg_list[2].lower == "SN":
    SN = True 
else:
    SN = False
if arg_list[3].lower == "GP":
    GP = True
    from functools import partial
    LAMDA = 10
else:
    GP = False
    
SAVE_DIR = 'img/{}/generated_img_CIFAR10_{}_{}_{}/'.format(LOSS, arg_list[1], arg_list[2], arg_list[3])
#SAVE_DIR = 'img/generated_img_CIFAR10_{}_fine_b2_09/'.format(arg_list[1])

if not os.path.isdir('img/'+LOSS):
    print('mkdir {}'.format('img/'+LOSS))
    os.mkdir('img/'+LOSS)

if not os.path.isdir(SAVE_DIR):
    print('mkdir {}'.format(SAVE_DIR))
    os.mkdir(SAVE_DIR)

PLOT_MODEL = False
SUMMARY = True
RESIST_GPU_MEM = True
GENERATE_ROW_NUM = 8
GENERATE_BATCHSIZE = GENERATE_ROW_NUM*GENERATE_ROW_NUM

if RESIST_GPU_MEM:
    # for resist GPU memory (Only in TensorFlow Backend)
    if K.backend() == 'tensorflow':
        import tensorflow as tf
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        sess = tf.Session(config=config)
        K.set_session(sess)

# wasserstein_loss
def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true*y_pred)

def hinge_G_loss(y_true, y_pred):
    return -K.mean(y_pred)

def hinge_D_real_loss(y_true, y_pred):
    return K.mean(K.relu(1-y_pred))

def hinge_D_fake_loss(y_true, y_pred):
    return K.mean(K.relu(1+y_pred))

if LOSS == 'wasserstein':
    G_LOSS = wasserstein_loss
    D_real_LOSS = wasserstein_loss
    D_fake_LOSS = wasserstein_loss
elif LOSS == 'binary_crossentropy':
    G_LOSS = LOSS
    D_real_LOSS = LOSS
    D_fake_LOSS = LOSS
elif LOSS == 'hinge':
    G_LOSS = hinge_G_loss
    D_real_LOSS = hinge_D_real_loss
    D_fake_LOSS = hinge_D_fake_loss
#Build Model
generator = BuildGenerator(bn_momentum=BN_MIMENTUM, bn_epsilon=BN_EPSILON, resnet=RESNET, plot=PLOT_MODEL, summary=SUMMARY)
discriminator = BuildDiscriminator(resnet=RESNET,spectral_normalization=SN, plot=PLOT_MODEL, summary=SUMMARY)

#Build Model for Training
if GP:
    from keras.layers.merge import _Merge
    
    def gradient_penalty_loss(y_true, y_pred, averaged_samples, gradient_penalty_weight):
        # first get the gradients:
        #   assuming: - that y_pred has dimensions (batch_size, 1)
        #             - averaged_samples has dimensions (batch_size, nbr_features)
        # gradients afterwards has dimension (batch_size, nbr_features), basically
        # a list of nbr_features-dimensional gradient vectors
        gradients = K.gradients(y_pred, averaged_samples)[0]
        # compute the euclidean norm by squaring ...
        gradients_sqr = K.square(gradients)
        #   ... summing over the rows ...
        gradients_sqr_sum = K.sum(gradients_sqr,
                                  axis=np.arange(1, len(gradients_sqr.shape)))
        #   ... and sqrt
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        # compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
        # return the mean as loss over all the batch samples
        return K.mean(gradient_penalty)
    
    
    class RandomWeightedAverage(_Merge):
        """Takes a randomly-weighted average of two tensors. In geometric terms, this outputs a random point on the line
        between each pair of input points.
        Inheriting from _Merge is a little messy but it was the quickest solution I could think of.
        Improvements appreciated."""

        def _merge_function(self, inputs):
            weights = K.random_uniform((BATCH_SIZE, 1, 1, 1))
            return (weights * inputs[0]) + ((1 - weights) * inputs[1])
    
    Noise_input_for_training_generator = Input(shape=(128,))
    Generated_image                    = generator(Noise_input_for_training_generator)
    Discriminator_output               = discriminator(Generated_image)
    model_for_training_generator       = Model(Noise_input_for_training_generator, Discriminator_output)
    discriminator.trainable = False
    if SUMMARY:
        print("model_for_training_generator")
        model_for_training_generator.summary()

    model_for_training_generator.compile(optimizer=Adam(LEARNING_RATE, beta_1=BETA_1, beta_2=BETA_2), loss=G_LOSS)

    Real_image                             = Input(shape=(32,32,3))
    Noise_input_for_training_discriminator = Input(shape=(128,))
    Fake_image                             = generator(Noise_input_for_training_discriminator)
    Averaged_samples                       = RandomWeightedAverage()([Real_image, Fake_image])
    Discriminator_output_for_real          = discriminator(Real_image)
    Discriminator_output_for_fake          = discriminator(Fake_image)
    Discriminator_output_for_averaged_samples = discriminator(Averaged_samples)

    model_for_training_discriminator       = Model([Real_image,
                                                    Noise_input_for_training_discriminator],
                                                   [Discriminator_output_for_real,
                                                    Discriminator_output_for_fake,
                                                    Discriminator_output_for_averaged_samples])
    generator.trainable = False
    discriminator.trainable = True
    model_for_training_discriminator.compile(optimizer=Adam(LEARNING_RATE, beta_1=BETA_1, beta_2=BETA_2), loss=[D_real_LOSS, D_fake_LOSS])
                      
    if SUMMARY:
        print("model_for_training_discriminator")
        model_for_training_discriminator.summary()
    partial_gp_loss = partial(gradient_penalty_loss,
                          averaged_samples=averaged_samples,
                          gradient_penalty_weight=LAMDA)
    partial_gp_loss.__name__ = 'gradient_penalty'
    
else:
    Noise_input_for_training_generator = Input(shape=(128,))
    Generated_image                    = generator(Noise_input_for_training_generator)
    Discriminator_output               = discriminator(Generated_image)
    model_for_training_generator       = Model(Noise_input_for_training_generator, Discriminator_output)
    discriminator.trainable = False
    if SUMMARY:
        print("model_for_training_generator")
        model_for_training_generator.summary()

    model_for_training_generator.compile(optimizer=Adam(LEARNING_RATE, beta_1=BETA_1, beta_2=BETA_2), loss=G_LOSS)

    Real_image                             = Input(shape=(32,32,3))
    Noise_input_for_training_discriminator = Input(shape=(128,))
    Fake_image                             = generator(Noise_input_for_training_discriminator)
    Discriminator_output_for_real          = discriminator(Real_image)
    Discriminator_output_for_fake          = discriminator(Fake_image)

    model_for_training_discriminator       = Model([Real_image,
                                                    Noise_input_for_training_discriminator],
                                                   [Discriminator_output_for_real,
                                                    Discriminator_output_for_fake])
    generator.trainable = False
    discriminator.trainable = True
    model_for_training_discriminator.compile(optimizer=Adam(LEARNING_RATE, beta_1=BETA_1, beta_2=BETA_2), loss=[D_real_LOSS, D_fake_LOSS])
    if SUMMARY:
        print("model_for_training_discriminator")
        model_for_training_discriminator.summary()
                      
#Load data
from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
X = np.concatenate((x_test,x_train))
## Normalize it 
X = X/255*2-1

# Make Label for traing
if LOSS == 'binary_crossentropy':
    fake_y = np.zeros((BATCHSIZE, 1), dtype=np.float32)
    real_y = np.ones((BATCHSIZE, 1), dtype=np.float32)
else:
    fake_y = np.ones((BATCHSIZE, 1), dtype=np.float32)
    real_y = -fake_y

if GP:
    dummy_y = np.zeros((BATCH_SIZE, 1), dtype=np.float32)
                      
test_noise = np.random.randn(GENERATE_BATCHSIZE, 128)

discriminator_loss = []
generator_loss = []

if GP:
    for epoch in range(EPOCHS):
        np.random.shuffle(X)

        print("epoch {} of {}".format(epoch+1, EPOCHS))
        num_batches = int(X.shape[0] // BATCHSIZE)

        print("number of batches: {}".format(int(X.shape[0] // (BATCHSIZE))))

        progress_bar = Progbar(target=int(X.shape[0] // (BATCHSIZE * TRAINING_RATIO)))
        minibatches_size = BATCHSIZE * TRAINING_RATIO

        start_time = time()
        for index in range(int(X.shape[0] // (BATCHSIZE * TRAINING_RATIO))):
            progress_bar.update(index)
            discriminator_minibatches = X[index * minibatches_size:(index + 1) * minibatches_size]

            for j in range(TRAINING_RATIO):
                image_batch = discriminator_minibatches[j * BATCHSIZE : (j + 1) * BATCHSIZE]
                noise = np.random.randn(BATCHSIZE, 128).astype(np.float32)
                discriminator.trainable = True
                generator.trainable = False
                discriminator_loss.append(model_for_training_discriminator.train_on_batch([image_batch, noise],
                                                                                          [real_y, fake_y, dummy_y]))
            discriminator.trainable = False
            generator.trainable = True
            generator_loss.append(model_for_training_generator.train_on_batch(np.random.randn(BATCHSIZE, 128), real_y))

        print('\nepoch time: {}'.format(time()-start_time))
        #Generate image
        generated_image = generator.predict(test_noise)
        generated_image = (generated_image+1)/2
        for i in range(GENERATE_ROW_NUM):
            new = generated_image[i*GENERATE_ROW_NUM:i*GENERATE_ROW_NUM+GENERATE_ROW_NUM].reshape(32*GENERATE_ROW_NUM,32,3)
            if i!=0:
                old = np.concatenate((old,new),axis=1)
            else:
                old = new
        print('plot generated_image')
        plt.imsave('{}/epoch_{:03}.png'.format(SAVE_DIR, epoch), old)
        
        plt.plot(discriminator_loss)
        plt.plot(generator_loss)
        plt.legend(['discriminator', 'real_D_loss', 'fake_D_loss', 'GP_Loss', 'generator_loss'])
        plt.savefig(SAVE_DIR+'/loss.png')
        plt.clf()

        pickle.dump({'discriminator_loss': discriminator_loss, 
                     'generator_loss': generator_loss}, 
                    open(SAVE_DIR+'/loss-history.pkl', 'wb'))


else:
    for epoch in range(EPOCHS):
        np.random.shuffle(X)

        print("epoch {} of {}".format(epoch+1, EPOCHS))
        num_batches = int(X.shape[0] // BATCHSIZE)

        print("number of batches: {}".format(int(X.shape[0] // (BATCHSIZE))))

        progress_bar = Progbar(target=int(X.shape[0] // (BATCHSIZE * TRAINING_RATIO)))
        minibatches_size = BATCHSIZE * TRAINING_RATIO

        start_time = time()
        for index in range(int(X.shape[0] // (BATCHSIZE * TRAINING_RATIO))):
            progress_bar.update(index)
            discriminator_minibatches = X[index * minibatches_size:(index + 1) * minibatches_size]

            for j in range(TRAINING_RATIO):
                image_batch = discriminator_minibatches[j * BATCHSIZE : (j + 1) * BATCHSIZE]
                noise = np.random.randn(BATCHSIZE, 128).astype(np.float32)
                discriminator.trainable = True
                generator.trainable = False
                discriminator_loss.append(model_for_training_discriminator.train_on_batch([image_batch, noise],
                                                                                          [real_y, fake_y]))
            discriminator.trainable = False
            generator.trainable = True
            generator_loss.append(model_for_training_generator.train_on_batch(np.random.randn(BATCHSIZE, 128), real_y))

        print('\nepoch time: {}'.format(time()-start_time))

        #Generate image
        generated_image = generator.predict(test_noise)
        generated_image = (generated_image+1)/2
        for i in range(GENERATE_ROW_NUM):
            new = generated_image[i*GENERATE_ROW_NUM:i*GENERATE_ROW_NUM+GENERATE_ROW_NUM].reshape(32*GENERATE_ROW_NUM,32,3)
            if i!=0:
                old = np.concatenate((old,new),axis=1)
            else:
                old = new
        print('plot generated_image')
        plt.imsave('{}/epoch_{:03}.png'.format(SAVE_DIR, epoch), old)
        
        plt.plot(discriminator_loss)
        plt.plot(generator_loss)
        plt.legend(['discriminator', 'real_D_loss', 'fake_D_loss', 'generator_loss'])
        plt.savefig(SAVE_DIR+'/loss.png')
        plt.clf()

        pickle.dump({'discriminator_loss': discriminator_loss, 
                     'generator_loss': generator_loss}, 
                    open(SAVE_DIR+'/loss-history.pkl', 'wb'))
