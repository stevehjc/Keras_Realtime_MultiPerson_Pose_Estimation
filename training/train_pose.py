# encoding=utf-8
import math
import os
import re
import sys
import pandas
from functools import partial #用于部分函数应用程序，它“冻结”函数的参数和/或关键字的某些部分，从而产生具有简化签名的新对象。

import keras.backend as K
from keras.applications.vgg19 import VGG19
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, CSVLogger, TensorBoard
from keras.layers.convolutional import Conv2D

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from model.cmu_model import get_training_model
from training.optimizers import MultiSGD
from training.dataset import get_dataflow, batch_dataflow


batch_size = 4 #10
base_lr = 4e-5 # 2e-5 #初始学习率，之后按照lr_policy衰减
momentum = 0.9
weight_decay = 5e-4 #卷积层权重正则化参数
lr_policy =  "step"
gamma = 0.333
stepsize = 68053 #136106 #68053   // after each stepsize iterations update learning rate: lr=lr*gamma
max_iter = 20 # 600000

weights_best_file = "weights.best.h5" #默认加载上一次训练的结果，如果不存在，则利用keras自动下载VGG19预训练模型
training_log = "training.csv" #保存每个epochs训练结束后的loss，包括多个stages
logs_dir = "./logs" #tensorboard logdir

from_vgg = { #VGG19的前10层的layer_name,load权重时按照名称加载
    'conv1_1': 'block1_conv1',
    'conv1_2': 'block1_conv2',
    'conv2_1': 'block2_conv1',
    'conv2_2': 'block2_conv2',
    'conv3_1': 'block3_conv1',
    'conv3_2': 'block3_conv2',
    'conv3_3': 'block3_conv3',
    'conv3_4': 'block3_conv4',
    'conv4_1': 'block4_conv1',
    'conv4_2': 'block4_conv2'
}


def get_last_epoch():
    """
    Retrieves last epoch from log file updated during training.

    :return: epoch number
    """
    data = pandas.read_csv(training_log)
    return max(data['epoch'].values)


def restore_weights(weights_best_file, model):
    """
    Restores weights from the checkpoint file if exists or
    preloads the first layers with VGG19 weights

    :param weights_best_file:
    :return: epoch number to use to continue training. last epoch + 1 or 0
    """
    # load previous weights or vgg19 if this is the first run
    if os.path.exists(weights_best_file):
        print("Loading the best weights...")

        model.load_weights(weights_best_file)

        return get_last_epoch() + 1
    else:
        print("Loading vgg19 weights...")

        vgg_model = VGG19(include_top=False, weights='imagenet') #keras will automatically download VGG19 weights

        for layer in model.layers:
            if layer.name in from_vgg:
                vgg_layer_name = from_vgg[layer.name]
                layer.set_weights(vgg_model.get_layer(vgg_layer_name).get_weights())
                print("Loaded VGG19 layer: " + vgg_layer_name)

        return 0


def get_lr_multipliers(model):
    """
    Setup multipliers for stageN layers (kernel and bias)

    :param model:
    :return: dictionary key: layer name , value: multiplier
    """
    lr_mult = dict()
    for layer in model.layers:

        if isinstance(layer, Conv2D):

            # stage = 1
            if re.match("Mconv\d_stage1.*", layer.name):
                kernel_name = layer.weights[0].name
                bias_name = layer.weights[1].name
                lr_mult[kernel_name] = 1
                lr_mult[bias_name] = 2

            # stage > 1
            elif re.match("Mconv\d_stage.*", layer.name):
                kernel_name = layer.weights[0].name
                bias_name = layer.weights[1].name
                lr_mult[kernel_name] = 4
                lr_mult[bias_name] = 8

            # vgg
            else:
                kernel_name = layer.weights[0].name
                bias_name = layer.weights[1].name
                lr_mult[kernel_name] = 1
                lr_mult[bias_name] = 2

    return lr_mult


def get_loss_funcs():
    """
    Euclidean loss as implemented in caffe
    https://github.com/BVLC/caffe/blob/master/src/caffe/layers/euclidean_loss_layer.cpp
    :return:
    """
    def _eucl_loss(x, y): #keras也有默认的平均平方差损失函数
        return K.sum(K.square(x - y)) / batch_size / 2 #除以batch_size，即每张图片的损失；x，y是batch中所有

    losses = {}
    losses["weight_stage1_L1"] = _eucl_loss
    losses["weight_stage1_L2"] = _eucl_loss
    losses["weight_stage2_L1"] = _eucl_loss
    losses["weight_stage2_L2"] = _eucl_loss
    losses["weight_stage3_L1"] = _eucl_loss
    losses["weight_stage3_L2"] = _eucl_loss
    losses["weight_stage4_L1"] = _eucl_loss
    losses["weight_stage4_L2"] = _eucl_loss
    losses["weight_stage5_L1"] = _eucl_loss
    losses["weight_stage5_L2"] = _eucl_loss
    losses["weight_stage6_L1"] = _eucl_loss
    losses["weight_stage6_L2"] = _eucl_loss

    return losses


def step_decay(epoch, iterations_per_epoch):
    """
    Learning rate schedule - equivalent of caffe lr_policy =  "step"

    :param epoch:
    :param iterations_per_epoch:
    :return:
    """
    initial_lrate = base_lr
    steps = epoch * iterations_per_epoch

    lrate = initial_lrate * math.pow(gamma, math.floor(steps/stepsize))

    return lrate


def gen(df):
    """
    Wrapper around generator. Keras fit_generator requires looping generator.
    :param df: dataflow instance
    """
    while True:
        for i in df.get_data():
            yield i


if __name__ == '__main__':

    # get the model
    model = get_training_model(weight_decay)

    # restore weights
    last_epoch = restore_weights(weights_best_file, model)

    #输出模型结果
    # model.summary()

    # prepare generators
    # curr_dir = os.path.dirname(__file__)
    # annot_path = os.path.join(curr_dir, '../dataset/annotations/person_keypoints_train2017.json') 
    # img_dir = os.path.abspath(os.path.join(curr_dir, '../dataset/train2017/'))
    annot_path="/media/han/E/mWork/datasets/COCO2017/annotations/person_keypoints_val2017.json"
    img_dir='/media/han/E/mWork/datasets/COCO2017/val2017/'

    # get dataflow of samples
    df = get_dataflow(  #数据集读取和处理（mask,augment,heatmap,paf等）多线程
        annot_path=annot_path,
        img_dir=img_dir)
    train_samples = df.size()

    # get generator of batches
    batch_df = batch_dataflow(df, batch_size)
    train_gen = gen(batch_df)

    # setup lr multipliers for conv layers
    lr_multipliers = get_lr_multipliers(model)

    # configure callbacks
    iterations_per_epoch = train_samples // batch_size
    _step_decay = partial(step_decay,
                          iterations_per_epoch=iterations_per_epoch
                          )
    lrate = LearningRateScheduler(_step_decay)
    checkpoint = ModelCheckpoint(weights_best_file, monitor='loss',
                                 verbose=0, save_best_only=False,
                                 save_weights_only=True, mode='min', period=1) #每个epochs都记录
    csv_logger = CSVLogger(training_log, append=True)
    tb = TensorBoard(log_dir=logs_dir, histogram_freq=0, write_graph=True,
                     write_images=False)

    callbacks_list = [lrate, checkpoint, csv_logger, tb]

    # sgd optimizer with lr multipliers
    multisgd = MultiSGD(lr=base_lr, momentum=momentum, decay=0.0,
                        nesterov=False, lr_mult=lr_multipliers)

    # start training
    loss_funcs = get_loss_funcs()

    # multi_gpu
    if False:
        from keras.utils import multi_gpu_model
        model = multi_gpu_model(model, gpus=1) 

    model.compile(loss=loss_funcs, optimizer=multisgd, metrics=["accuracy"])
    model.fit_generator(train_gen,
                        steps_per_epoch=train_samples // batch_size,
                        epochs=max_iter,
                        callbacks=callbacks_list,
                        # validation_data=val_di,
                        # validation_steps=val_samples // batch_size,
                        use_multiprocessing=False,
                        initial_epoch=last_epoch)
