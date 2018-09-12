
import math
import numpy as np
import cv2
import matplotlib.pylab as plt
import keras
from keras.datasets import mnist
from keras.models import Sequential
# from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.regularizers import l2
from keras.models import Model
'''
这是借用热点图的思想来做小目标检测，训练已经调通，但是目前还不是很理想
1.数据集太少
2.训练过拟合
3.网络模型暂时没有依据
'''

def create_heatmap(num_maps, height, width, all_joints, sigma, stride):
    """
    Creates stacked heatmaps for all joints + background. For 18 joints
    we would get an array height x width x 19.
    Size width and height should be the same as output from the network
    so this heatmap can be used to evaluate a loss function.

    :param num_maps: number of maps. for coco dataset we have 18 joints + 1 background
    :param height: height dimension of the network output
    :param width: width dimension of the network output
    :param all_joints: list of all joints (for coco: 18 items)
    :param sigma: parameter used to calculate a gaussian
    :param stride: parameter used to scale down the coordinates of joints. Those coords
            relate to the original image size
    :return: heat maps (height x width x num_maps)
    """
    heatmap = np.zeros((height, width, num_maps), dtype=np.float64)
    #all_joints eg:[[(x,y),(x1,y1)],[]]
    for joints in all_joints: #取到某个人的关键定
        for plane_idx, joint in enumerate(joints): #取到某个人的某个关键定
            if joint.all():
                _put_heatmap_on_plane(heatmap, plane_idx, joint, sigma, height, width, stride)

    # background
    heatmap[:, :, -1] = np.clip(1.0 - np.amax(heatmap, axis=2), 0.0, 1.0)

    return heatmap

def _put_heatmap_on_plane(heatmap, plane_idx, joint, sigma, height, width, stride):
    start = stride / 2.0 - 0.5

    center_x, center_y = joint

    for g_y in range(height):
        for g_x in range(width):
            x = start + g_x * stride
            y = start + g_y * stride
            d2 = (x-center_x) * (x-center_x) + (y-center_y) * (y-center_y)
            exponent = d2 / 2.0 / sigma / sigma
            if exponent > 4.6052:
                continue

            heatmap[g_y, g_x, plane_idx] += math.exp(-exponent)
            if heatmap[g_y, g_x, plane_idx] > 1.0:
                heatmap[g_y, g_x, plane_idx] = 1.0


def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def resnet_v1(input_shape, depth, num_classes=10):
    """ResNet Version 1 Model builder [a]

    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    # y = AveragePooling2D(pool_size=2)(x)
    outputs=Conv2D(1, (3, 3), padding='same',activation='relu')(x)
    # y = Flatten()(x)
    # outputs = Dense(num_classes,
    #                 activation='softmax',
    #                 kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model

def resnet_v2(input_shape, depth, num_classes=10):
    """ResNet Version 2 Model builder [b]

    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
    bottleneck layer
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filter maps is
    doubled. Within each stage, the layers have the same number filters and the
    same filter map sizes.
    Features maps sizes:
    conv1  : 32x32,  16
    stage 0: 32x32,  64
    stage 1: 16x16, 128
    stage 2:  8x8,  256

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    inputs = Input(shape=input_shape)
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_layer(inputs=inputs,
                     num_filters=num_filters_in,
                     conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2    # downsample

            # bottleneck residual unit
            y = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_in,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])

        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    outputs = AveragePooling2D(pool_size=8)(x)
    # y = BatchNormalization()(x)
    # outputs = Dense(num_classes,
    #                 activation='softmax',
    #                 kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model

def process():
    data=np.loadtxt("/media/han/D/datasets/small_targets/TargetInfo.txt")
    heatmaps=[]
    # for i in range(10):
    for i in range(10):
        tmp=data[10*i:10*(i+1),2:4]
        all_joints=[]
        for box in tmp:
            all_joints.append([box[::-1]])
        # all_joints.append(tmp)
        # all_joints=[[[418,139]],[[231.7,251.8]],[[323.9,107.2]]]
        # all_joints=[[[418,139]]]
        heatmap = create_heatmap(2, 128, 128,
                                    all_joints, 7.0, stride=4)
        cv2.imwrite("hdebug/heatmap_1.png",heatmap[:,:,0]*255)
        heatmaps.append(heatmap[:,:,0]*255)
    
    y_train=np.array(heatmaps)
    x_train=[]
    for i in range(10):
        filename="/media/han/D/datasets/small_targets/"+str(i+1)+".bmp"
        im=cv2.imread(filename,0)
        x_train.append(im)
    
    x_train=np.array(x_train)

    np.save('hdebug/x_train.npy',x_train)
    np.save('hdebug/y_train.npy',y_train)

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')    
    print("save to .npy")


def train_1():
    # process() #生成热图并存储为npy格式

    x_train=np.load('hdebug/x_train.npy')
    y_train=np.load('hdebug/y_train.npy')
    x_train=x_train.reshape(10,512,512,1)
    y_train=y_train.reshape(10,128,128,1)

    input_shape=x_train.shape[1:]
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),padding='same',
                    activation='relu',
                    input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), padding='same',activation='relu'))
    model.add(Conv2D(16, (3, 3), padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(1, (3, 3), padding='same',activation='relu'))

    model.summary()

    model.compile(loss='mean_squared_error',
                    optimizer=keras.optimizers.SGD(),
                    metrics=['accuracy'])
    model.fit(x_train, y_train,
            batch_size=10,
            epochs=20,
            verbose=1)

    y_test=model.predict(x_train[0:10,:,:,:])
    # plt.imshow(y_test.reshape(128,128),cmap='gray')
    # cv2.imshow("test",y_test.reshape(128.128)) 
    print("fit done")      

def train_2():
    # Training parameters
    batch_size = 5  # orig paper trained all networks with batch_size=128
    epochs = 30
    data_augmentation = True
    # num_classes = 2

    # Subtracting pixel mean improves accuracy
    subtract_pixel_mean = True

    n = 3
    # Model version
    # Orig paper: version = 1 (ResNet v1), Improved ResNet: version = 2 (ResNet v2)
    version = 1

    # Computed depth from supplied model parameter n
    if version == 1:
        depth = n * 6 + 2
    elif version == 2:
        depth = n * 9 + 2

    # Model name, depth and version
    model_type = 'ResNet%dv%d' % (depth, version)

    x_train=np.load('hdebug/x_train.npy')
    y_train=np.load('hdebug/y_train.npy')
    x_train=x_train.reshape(10,512,512,1)
    y_train=y_train.reshape(10,128,128,1)

    input_shape=x_train.shape[1:]

    if version == 2:
        model = resnet_v2(input_shape=input_shape, depth=depth)
    else:
        model = resnet_v1(input_shape=input_shape, depth=depth)

    model.compile(loss='mean_squared_error',
                optimizer=Adam(lr=lr_schedule(0)),
                metrics=['accuracy'])
    model.summary()
    savefilepath="hdebug/st_model.h5"
    # checkpoint = ModelCheckpoint(filepath=filepath,
    #                          monitor='val_acc',
    #                          verbose=1,
    #                          save_best_only=True)
    # lr_scheduler = LearningRateScheduler(lr_schedule)
    # lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
    #                            cooldown=0,
    #                            patience=5,
    #                            min_lr=0.5e-6)
    # callbacks = [checkpoint, lr_reducer, lr_scheduler]
    model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,            
            shuffle=True)
            #callbacks=callbacks) 
    
    model.save(savefilepath)
    while True:
        imgpath=input("image path:")
        try:
            im=cv2.imread(imgpath,0)
        except:
            print('Open Error! Try again!')
            continue
        else:
            result=model.predict(im.reshape(1,512,512,1))
            cv2.imwrite("hdebug/y_3.png",result.reshape(128,128))
    # y_test=model.predict(x_train[0:10,:,:,:])

    print("train_2 done")

if __name__ == '__main__':
    # train_1()
    # train_2()
    # ALL_PAF_MASK = np.repeat(
    # np.ones((46, 46, 1), dtype=np.uint8), 38, axis=2)
    # print("pasue")
    