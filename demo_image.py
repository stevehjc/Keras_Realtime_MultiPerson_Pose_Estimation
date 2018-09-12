#encoding=utf-8
import os
import sys
import argparse
import cv2
import math
import time
import numpy as np
import util
from config_reader import config_reader
from scipy.ndimage.filters import gaussian_filter

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from model.cmu_model import get_testing_model
'''
#鼻子 脖子 右肩 右肘 右手腕 左肩 左肘 左手腕 右臀 右膝盖 右脚踝 左臀 左膝盖 左脚踝 右眼 左眼 右耳 左耳
# 1    2   3    4   5    6    7   8     9    10   11    12   13    14   15  16   17  18

heatmap顺序为： 1~18 19

paf(38维度)顺序如下，参考train/dataflow.py中joint_pairs
关键点 ：[2 9] [9 10] [10 11] [2 12] [12 13] [13 14] [2 3] [3 4] [4 5] [3 17] [2 6] [6 7] [7 8]
paf序号：0      2      4       6      8       10      12    14    16    18     20    22    24
关键点 ：[6 18] [2 1] [1 15] [1 16] [15 16] [16 18]
paf序号：26      28    30     32     34      36
'''

# find connection in the specified sequence, center 29 is in the position 15
#不同的limbs,19种，也就是19种连线方式；对应网络设计中的19和38；这个顺序是计算、绘图的顺序，不是paf存储顺序
limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
           [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
           [1, 16], [16, 18], [3, 17], [6, 18]]

# the middle joints heatmap correpondence
#不同limbs对应的paf输出索引；和limbSeq对应，都是19个；应用时mapIdx各元素-19
mapIdx = [[31, 32], [39, 40], [33, 34], [35, 36], [41, 42], [43, 44], [19, 20], [21, 22], \
          [23, 24], [25, 26], [27, 28], [29, 30], [47, 48], [49, 50], [53, 54], [51, 52], \
          [55, 56], [37, 38], [45, 46]]

# visualize
#不同关键点的color,18种;BGR顺序
colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0],
          [0, 255, 0], \
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255],
          [85, 0, 255], \
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]


def process (input_image, params, model_params):

    oriImg = cv2.imread(input_image)  # B,G,R order
    multiplier = [x * model_params['boxsize'] / oriImg.shape[0] for x in params['scale_search']]

    heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 19))
    paf_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 38))

    for m in range(len(multiplier)):
        scale = multiplier[m]

        imageToTest = cv2.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        imageToTest_padded, pad = util.padRightDownCorner(imageToTest, model_params['stride'],
                                                          model_params['padValue'])

        input_img = np.transpose(np.float32(imageToTest_padded[:,:,:,np.newaxis]), (3,0,1,2)) # required shape (1, width, height, channels)
        tic=time.time()
        output_blobs = model.predict(input_img)
        toc=time.time()
        print("------",toc-tic)
        # extract outputs, resize, and remove padding
        heatmap = np.squeeze(output_blobs[1])  # output 1 is heatmaps
        heatmap = cv2.resize(heatmap, (0, 0), fx=model_params['stride'], fy=model_params['stride'],  # 上采样8倍；因为网络中pool导致输出比输入缩小了8倍
                             interpolation=cv2.INTER_CUBIC)
        heatmap = heatmap[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], #图像偏移，从下边和右边割掉一部分行和列,
                  :]
        heatmap = cv2.resize(heatmap, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC) #恢复到输入尺寸大小

        paf = np.squeeze(output_blobs[0])  # output 0 is PAFs
        paf = cv2.resize(paf, (0, 0), fx=model_params['stride'], fy=model_params['stride'],
                         interpolation=cv2.INTER_CUBIC)
        paf = paf[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
        paf = cv2.resize(paf, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)

        heatmap_avg = heatmap_avg + heatmap / len(multiplier)
        paf_avg = paf_avg + paf / len(multiplier)

    all_peaks = []
    peak_counter = 0 #峰值点编号，计数

    #非极大值抑制
    for part in range(18):
        map_ori = heatmap_avg[:, :, part]
        map = gaussian_filter(map_ori, sigma=3) #高斯滤波，去除噪声

        map_left = np.zeros(map.shape)
        map_left[1:, :] = map[:-1, :] #将所有坐标向下移动一行，原作者命名有问题
        map_right = np.zeros(map.shape)
        map_right[:-1, :] = map[1:, :] #将所有坐标向上移动一行
        map_up = np.zeros(map.shape)
        map_up[:, 1:] = map[:, :-1] #将所有坐标向向右移动一列
        map_down = np.zeros(map.shape)
        map_down[:, :-1] = map[:, 1:] #将所有坐标向左移动一列

        #当前像素值必须大于上下左右的像素值
        peaks_binary = np.logical_and.reduce(
            (map >= map_left, map >= map_right, map >= map_up, map >= map_down, map > params['thre1'])) #参数thre1=0.1
        peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))  # note reverse #返回坐标(列，行)
        peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks] #获得peaks的坐标（列,行,score）
        id = range(peak_counter, peak_counter + len(peaks))
        peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))] #给关键点编号

        all_peaks.append(peaks_with_score_and_id)
        peak_counter += len(peaks)

    connection_all = [] #所有19种连接方式，每种连接按照贪婪方法配对
    special_k = [] #特殊情况，没有检测到某一类关键点
    mid_num = 10 

    for k in range(len(mapIdx)):
        score_mid = paf_avg[:, :, [x - 19 for x in mapIdx[k]]] #得到score_imd.shape=[height,width,2]
        candA = all_peaks[limbSeq[k][0] - 1] #列，行
        candB = all_peaks[limbSeq[k][1] - 1]
        nA = len(candA)
        nB = len(candB)
        indexA, indexB = limbSeq[k] #得到两个关键点
        if (nA != 0 and nB != 0):
            connection_candidate = []
            for i in range(nA): #两幅图中，所有的A类关键点和B类关键点尝试连接
                for j in range(nB):
                    vec = np.subtract(candB[j][:2], candA[i][:2]) #尝试连接两个关键点之间的向量
                    norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
                    # failure case when 2 body parts overlaps
                    if norm == 0:
                        continue
                    vec = np.divide(vec, norm)
                    # 将向量分成10等分，得到向量上的10个坐标点
                    startend = list(zip(np.linspace(candA[i][0], candB[j][0], num=mid_num), \
                                   np.linspace(candA[i][1], candB[j][1], num=mid_num)))

                    vec_x = np.array( #获取两个关键点连线上10个中间点的paf值dx ；列
                        [score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0] \
                         for I in range(len(startend))])
                    vec_y = np.array( #获取两个关键点连线上10个中间点的paf值dy ；行
                        [score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1] \
                         for I in range(len(startend))])

                    score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1]) #向量点乘,余弦相似度
                    score_with_dist_prior = sum(score_midpts) / len(score_midpts) + min(
                        0.5 * oriImg.shape[0] / norm - 1, 0)
                    criterion1 = len(np.nonzero(score_midpts > params['thre2'])[0]) > 0.8 * len(
                        score_midpts) #80%的向量乘积大于阈值thre2=0.05
                    criterion2 = score_with_dist_prior > 0
                    if criterion1 and criterion2:
                        connection_candidate.append([i, j, score_with_dist_prior,
                                                     score_with_dist_prior + candA[i][2] + candB[j][2]])
            #对匹配好的关键点对进行筛选，依据score_with_dist_prior从大到小排序
            connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True) 
            connection = np.zeros((0, 5)) #关键点编号1，编号2，分数，i,j
            for c in range(len(connection_candidate)):
                i, j, s = connection_candidate[c][0:3]
                if (i not in connection[:, 3] and j not in connection[:, 4]): #如果已经由匹配了，则不再匹配；这里按照贪婪方法
                    connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                    if (len(connection) >= min(nA, nB)):
                        break

            connection_all.append(connection)
        else: #没有检测到关键点
            special_k.append(k)
            connection_all.append([])

    # last number in each row is the total parts number of that person
    # the second last number in each row is the score of the overall configuration
    subset = -1 * np.ones((0, 20))
    candidate = np.array([item for sublist in all_peaks for item in sublist]) #将检测到的所有关键点展开
    #匈牙利匹配算法
    for k in range(len(mapIdx)):
        if k not in special_k:
            partAs = connection_all[k][:, 0] #获取第k种连接方式的关键点编号
            partBs = connection_all[k][:, 1]
            indexA, indexB = np.array(limbSeq[k]) - 1

            for i in range(len(connection_all[k])):  # = 1:size(temp,1)
                found = 0
                subset_idx = [-1, -1]
                for j in range(len(subset)):  # 1:size(subset,1):
                    if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                        subset_idx[found] = j
                        found += 1

                if found == 1:
                    j = subset_idx[0]
                    if (subset[j][indexB] != partBs[i]):
                        subset[j][indexB] = partBs[i]
                        subset[j][-1] += 1
                        subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                elif found == 2:  # if found 2 and disjoint, merge them
                    j1, j2 = subset_idx
                    membership = ((subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int))[:-2]
                    if len(np.nonzero(membership == 2)[0]) == 0:  # merge
                        subset[j1][:-2] += (subset[j2][:-2] + 1)
                        subset[j1][-2:] += subset[j2][-2:]
                        subset[j1][-2] += connection_all[k][i][2]
                        subset = np.delete(subset, j2, 0)
                    else:  # as like found == 1
                        subset[j1][indexB] = partBs[i]
                        subset[j1][-1] += 1
                        subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]

                # if find no partA in the subset, create a new subset
                elif not found and k < 17:
                    row = -1 * np.ones(20)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    row[-1] = 2
                    row[-2] = sum(candidate[connection_all[k][i, :2].astype(int), 2]) + \
                              connection_all[k][i][2]
                    subset = np.vstack([subset, row])

    # delete some rows of subset which has few parts occur
    deleteIdx = []
    for i in range(len(subset)):
        if subset[i][-1] < 4 or subset[i][-2] / subset[i][-1] < 0.4:
            deleteIdx.append(i)
    subset = np.delete(subset, deleteIdx, axis=0)
    
    canvas = cv2.imread(input_image)  # B,G,R order
    for i in range(18):
        for j in range(len(all_peaks[i])):
            cv2.circle(canvas, all_peaks[i][j][0:2], 4, colors[i], thickness=-1)

    stickwidth = 4

    #绘制关键点之间的连接
    for i in range(17): 
        for n in range(len(subset)): #第n个人的关键点绘图，subset保存了n个人的18关键点连接的另一关键点编号
            index = subset[n][np.array(limbSeq[i]) - 1]
            if -1 in index:
                continue
            cur_canvas = canvas.copy()
            Y = candidate[index.astype(int), 0] #y1 y2
            X = candidate[index.astype(int), 1] #x1,x2
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5 #两个关键点连接向量长度，角度等
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0,
                                       360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0) #标记了关键点的图canvas和肢体连接图cur_canvas按照比例融合

    return canvas


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, default='sample_images/visionteam1.jpg', help='input image') #required=True
    parser.add_argument('--output', type=str, default='result.png', help='output image')
    parser.add_argument('--model', type=str, default='model/keras/model.h5', help='path to the weights file') #model/keras/model.h5

    args = parser.parse_args()
    input_image = args.image
    output = args.output
    keras_weights_file = args.model

    
    print('start processing...')

    # load model

    # authors of original model don't use
    # vgg normalization (subtracting mean) on input images
    model = get_testing_model(GPU=True)

    if False: # if model.h5 is trained on multi GPU
        from keras.utils import training_utils
        model = training_utils.multi_gpu_model(model,gpus=2)
    model.load_weights(keras_weights_file)

    # load config
    params, model_params = config_reader()

    tic = time.time()
    # generate image with body parts
    canvas = process(input_image, params, model_params)

    toc = time.time()
    print ('processing time is %.5f' % (toc - tic))

    cv2.imwrite(output, canvas)

    # cv2.destroyAllWindows()
    # import matplotlib
    # import pylab as plt
    # f,axarr=plt.subplots(1,2)
    # axarr.flat[0].imshow()


