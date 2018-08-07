from keras.layers import Input, Dense, Conv2D, Add, Dot, Conv2DTranspose, Activation, Reshape,BatchNormalization,UpSampling2D,AveragePooling2D, GlobalAveragePooling2D, LeakyReLU, Reshape, Flatten
from keras.models import Model, Sequential
import keras.backend as K
from keras.utils import plot_model
from SpectralNormalizationKeras import DenseSN, ConvSN2D
from keras.layers.pooling import _GlobalPooling2D

class GlobalSumPooling2D(_GlobalPooling2D):
    """Global sum pooling operation for spatial data.
    # Arguments
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, height, width, channels)` while `channels_first`
            corresponds to inputs with shape
            `(batch, channels, height, width)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
    # Input shape
        - If `data_format='channels_last'`:
            4D tensor with shape:
            `(batch_size, rows, cols, channels)`
        - If `data_format='channels_first'`:
            4D tensor with shape:
            `(batch_size, channels, rows, cols)`
    # Output shape
        2D tensor with shape:
        `(batch_size, channels)`
    """

    def call(self, inputs):
        if self.data_format == 'channels_last':
            return K.sum(inputs, axis=[1, 2])
        else:
            return K.sum(inputs, axis=[2, 3])


def ResBlock(input_shape, sampling=None, trainable_sortcut=True, 
             spectral_normalization=False, batch_normalization=True,
             bn_momentum=0.9, bn_epsilon=0.00002,
             channels=256, k_size=3, summary=False,
             plot=False, name=None):
    '''
    ResBlock(input_shape, sampling=None, trainable_sortcut=True, 
             spectral_normalization=False, batch_normalization=True,
             bn_momentum=0.9, bn_epsilon=0.00002,
             channels=256, k_size=3, summary=False,
             plot=False, plot_name='res_block.png')""
             
    Build ResBlock as keras Model
    sampleing = 'up' for upsampling
                'down' for downsampling(AveragePooling)
                None for none
    
    '''
    #input_shape = input_layer.sahpe.as_list()
    
    res_block_input = Input(shape=input_shape)
    
    if batch_normalization:
        res_block_1 = BatchNormalization(momentum=bn_momentum, epsilon=bn_epsilon)(res_block_input)
    else:
        res_block_1 = res_block_input
        
    res_block_1     = Activation('relu')(res_block_1)
    
    if spectral_normalization:
        res_block_1     = ConvSN2D(channels, k_size , strides=1, padding='same',kernel_initializer='glorot_uniform')(res_block_1)
    else:
        res_block_1     = Conv2D(channels, k_size , strides=1, padding='same',kernel_initializer='glorot_uniform')(res_block_1)
    
    if sampling=='up':
        res_block_1     = UpSampling2D()(res_block_1)
    else:
        pass
    
    if batch_normalization:
        res_block_2     = BatchNormalization(momentum=bn_momentum, epsilon=bn_epsilon)(res_block_1)
    else:
        res_block_2     = res_block_1
    res_block_2     = Activation('relu')(res_block_2)
    
    if spectral_normalization:
        res_block_2     = ConvSN2D(channels, k_size , strides=1, padding='same',kernel_initializer='glorot_uniform')(res_block_2)
    else:
        res_block_2     = Conv2D(channels, k_size , strides=1, padding='same',kernel_initializer='glorot_uniform')(res_block_2)
    
    if sampling=='down':
        res_block_2 = AveragePooling2D()(res_block_2)
    else:
        pass
    
    if trainable_sortcut:
        if spectral_normalization:
            short_cut = ConvSN2D(channels, 1 , strides=1, padding='same',kernel_initializer='glorot_uniform')(res_block_input)
        else:
            short_cut = Conv2D(channels, 1 , strides=1, padding='same',kernel_initializer='glorot_uniform')(res_block_input)
    else:
        short_cut = res_block_input
        
    if sampling=='up':
        short_cut       = UpSampling2D()(short_cut)
    elif sampling=='down':
        short_cut       = AveragePooling2D()(short_cut)
    elif sampling=='None':
        pass

    res_block_add   = Add()([short_cut, res_block_2])
    
    res_block = Model(res_block_input, res_block_add, name=name)
    
    if plot:
        plot_model(res_block, name+'.png', show_layer_names=False)
    if summary:
        print(name)
        res_block.summary()
    
    return res_block
    
def BuildGenerator(summary=True, resnet=True, bn_momentum=0.9, bn_epsilon=0.00002, name='Generator', plot=False):
    if resnet:
        model_input = Input(shape=(128,))
        h           = Dense(4*4*256, kernel_initializer='glorot_uniform')(model_input)
        h           = Reshape((4,4,256))(h)
        resblock_1  = ResBlock(input_shape=(4,4,256), sampling='up', bn_epsilon=bn_epsilon, bn_momentum=bn_momentum, name='Generator_resblock_1')
        h           = resblock_1(h)
        resblock_2  = ResBlock(input_shape=(8,8,256), sampling='up', bn_epsilon=bn_epsilon, bn_momentum=bn_momentum, name='Generator_resblock_2')
        h           = resblock_2(h)
        resblock_3  = ResBlock(input_shape=(16,16,256), sampling='up', bn_epsilon=bn_epsilon, bn_momentum=bn_momentum, name='Generator_resblock_3')
        h           = resblock_3(h)
        h           = BatchNormalization(epsilon=bn_epsilon, momentum=bn_momentum)(h)
        h           = Activation('relu')(h)
        model_output= Conv2D(3,   kernel_size=3, strides=1, padding='same', activation='tanh')(h)
        model = Model(model_input, model_output,name=name)
        
    else:
        model = Sequential(name=name)
        model.add(Dense(4*4*512, kernel_initializer='glorot_uniform' , input_dim=128))
        model.add(Reshape((4,4,512)))
        model.add(Conv2DTranspose(256, kernel_size=4, strides=2, padding='same', activation='relu',kernel_initializer='glorot_uniform'))
        model.add(BatchNormalization(epsilon=bn_epsilon, momentum=bn_momentum))
        model.add(Conv2DTranspose(128, kernel_size=4, strides=2, padding='same', activation='relu',kernel_initializer='glorot_uniform'))
        model.add(BatchNormalization(epsilon=bn_epsilon, momentum=bn_momentum))
        model.add(Conv2DTranspose(64,  kernel_size=4, strides=2, padding='same', activation='relu',kernel_initializer='glorot_uniform'))
        model.add(BatchNormalization(epsilon=bn_epsilon, momentum=bn_momentum))
        model.add(Conv2DTranspose(3,   kernel_size=3, strides=1, padding='same', activation='tanh'))
        
    if plot:
        plot_model(model, name+'.png', show_layer_names=True)
    if summary:
        print("Generator")
        model.summary()
    return model

def BuildDiscriminator(summary=True, spectral_normalization=True, batch_normalization=False, bn_momentum=0.9, bn_epsilon=0.00002, resnet=True, name='Discriminator', plot=False):
    if resnet:
        model_input = Input(shape=(32,32,3))
        resblock_1  = ResBlock(input_shape=(32,32,3), channels=128, sampling='down', batch_normalization=True, spectral_normalization=spectral_normalization, name='Discriminator_resblock_Down_1')
        h           = resblock_1(model_input)
        resblock_2  = ResBlock(input_shape=(16,16,128),channels=128, sampling='down', batch_normalization=True, spectral_normalization=spectral_normalization, name='Discriminator_resblock_Down_2')
        h           = resblock_2(h)
        resblock_3  = ResBlock(input_shape=(8,8,128),channels=128 , sampling=None, batch_normalization=True, spectral_normalization=spectral_normalization, trainable_sortcut=False, name='Discriminator_resblock_1' )
        h           = resblock_3(h)
        resblock_4  = ResBlock(input_shape=(8,8,128),channels=128 , sampling=None, batch_normalization=True, spectral_normalization=spectral_normalization, trainable_sortcut=False, name='Discriminator_resblock_2' )
        h           = resblock_4(h)
        h           = Activation('relu')(h)
        h           = GlobalSumPooling2D()(h)
        model_output= DenseSN(1,kernel_initializer='glorot_uniform')(h)

        model = Model(model_input, model_output, name=name)

    else:
        if spectral_normalization:
            model = Sequential(name=name)
            model.add(ConvSN2D(64, kernel_size=3, strides=1,kernel_initializer='glorot_uniform', padding='same', input_shape=(32,32,3) ))
            model.add(LeakyReLU(0.1))
            model.add(ConvSN2D(64, kernel_size=4, strides=2,kernel_initializer='glorot_uniform', padding='same'))
            model.add(LeakyReLU(0.1))
            model.add(ConvSN2D(128, kernel_size=3, strides=1,kernel_initializer='glorot_uniform', padding='same'))
            model.add(LeakyReLU(0.1))
            model.add(ConvSN2D(128, kernel_size=4, strides=2,kernel_initializer='glorot_uniform', padding='same'))
            model.add(LeakyReLU(0.1))
            model.add(ConvSN2D(256, kernel_size=3, strides=1,kernel_initializer='glorot_uniform', padding='same'))
            model.add(LeakyReLU(0.1))
            model.add(ConvSN2D(256, kernel_size=4, strides=2,kernel_initializer='glorot_uniform', padding='same'))
            model.add(LeakyReLU(0.1))
            model.add(ConvSN2D(512, kernel_size=3, strides=1,kernel_initializer='glorot_uniform', padding='same'))
            model.add(LeakyReLU(0.1))
            model.add(GlobalSumPooling2D())
            model.add(DenseSN(1,kernel_initializer='glorot_uniform'))
        else:
            model = Sequential(name=name)
            model.add(Conv2D(64, kernel_size=3, strides=1,kernel_initializer='glorot_uniform', padding='same', input_shape=(32,32,3) ))
            model.add(LeakyReLU(0.1))
            model.add(Conv2D(64, kernel_size=4, strides=2,kernel_initializer='glorot_uniform', padding='same'))
            model.add(LeakyReLU(0.1))
            model.add(Conv2D(128, kernel_size=3, strides=1,kernel_initializer='glorot_uniform', padding='same'))
            model.add(LeakyReLU(0.1))
            model.add(Conv2D(128, kernel_size=4, strides=2,kernel_initializer='glorot_uniform', padding='same'))
            model.add(LeakyReLU(0.1))
            model.add(Conv2D(256, kernel_size=3, strides=1,kernel_initializer='glorot_uniform', padding='same'))
            model.add(LeakyReLU(0.1))
            model.add(Conv2D(256, kernel_size=4, strides=2,kernel_initializer='glorot_uniform', padding='same'))
            model.add(LeakyReLU(0.1))
            model.add(Conv2D(512, kernel_size=3, strides=1,kernel_initializer='glorot_uniform', padding='same'))
            model.add(LeakyReLU(0.1))
            model.add(GlobalSumPooling2D())
            model.add(Dense(1,kernel_initializer='glorot_uniform'))
    if plot:
        plot_model(model, name+'.png', show_layer_names=True)
        
    if summary:
        print('Discriminator')
        print('Spectral Normalization: {}'.format(spectral_normalization))
        model.summary()
    return model

if __name__ == '__main__':
    print('Plot the model visualization')
    from keras.utils import plot_model
    DIR = 'img/model/'
    
    print('DCGAN_Generator')
    model = BuildGenerator(resnet=False)
    plot_model(model, show_shapes=True, to_file=DIR+'DCGAN_Generator.png')
    
    print('ResNet_Generator')
    model = BuildGenerator(resnet=True)
    plot_model(model, show_shapes=True, to_file=DIR+'ResNet_Generator.png')

    
    print('DCGAN_Discriminator')
    model = BuildDiscriminator(resnet=False)
    plot_model(model, show_shapes=True, to_file=DIR+'DCGAN_Discriminator.png')
    
    print('ResNet_Discriminator')
    model = BuildDiscriminator(resnet=True)
    plot_model(model, show_shapes=True, to_file=DIR+'ResNet_Discriminator.png')
    
    
    print('Generator_resblock_1')
    model = ResBlock(input_shape=(4,4,256), sampling='up',  name='Generator_resblock_1')
    plot_model(model, show_shapes=True, to_file=DIR+'Generator_resblock_1.png')

    
    print('Discriminator_resblock_Down_1')
    model = ResBlock(input_shape=(32,32,3), channels=128, sampling='down', spectral_normalization=True, name='Discriminator_resblock_Down_1')
    plot_model(model, show_shapes=True, to_file=DIR+'Discriminator_resblock_Down_1.png')
    
    print('Discriminator_resblock_1')
    model = ResBlock(input_shape=(8,8,128),channels=128 , sampling=None, spectral_normalization=True, name='Discriminator_resblock_1' )
    plot_model(model, show_shapes=True, to_file=DIR+'Discriminator_resblock_1.png')