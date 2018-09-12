import math
import numpy as np

from training.dataflow import JointsLoader


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
    #输入的all_joints：每张图上的所有人体关键点，18个点为一个list
    #eg:[[(x,y),(x1,y1),None,...],[....]] None表示当前关键点被遮盖，没有值
    heatmap = np.zeros((height, width, num_maps), dtype=np.float64)
    for joints in all_joints: #取到某个人的关键定
        for plane_idx, joint in enumerate(joints): #取到某个人的某个关键定
            if joint: #如果当前关键点没有被遮挡，不是None
                _put_heatmap_on_plane(heatmap, plane_idx, joint, sigma, height, width, stride)

    # background #增加一层背景
    heatmap[:, :, -1] = np.clip(1.0 - np.amax(heatmap, axis=2), 0.0, 1.0)

    return heatmap


def create_paf(num_maps, height, width, all_joints, threshold, stride):
    """
    Creates stacked paf maps for all connections. One connection requires
    2 maps because of paf vectors along dx and dy axis. For coco we have
    19 connections -> x2 it gives 38 maps

    :param num_maps: number of maps. for coco dataset we have 19 connections
    :param height: height dimension of the network output
    :param width: width dimension of the network output
    :param all_joints: list of all joints (for coco: 18 items)
    :param threshold: parameter determines the "thickness" of a limb within a paf
    :param stride: parameter used to scale down the coordinates of joints. Those coords
            relate to the original image size
    :return: paf maps (height x width x 2*num_maps)
    """
    vectormap = np.zeros((height, width, num_maps*2), dtype=np.float64) #paf
    countmap = np.zeros((height, width, num_maps), dtype=np.uint8) #用来记录某一坐标点是否计算过
    for joints in all_joints:
        for plane_idx, (j_idx1, j_idx2) in enumerate(JointsLoader.joint_pairs):
            center_from = joints[j_idx1] 
            center_to = joints[j_idx2]  

            # skip if no valid pair of keypoints
            if center_from is None or center_to is None:#如果某个关键点缺失，则无法连接
                continue

            x1, y1 = (center_from[0] / stride, center_from[1] / stride) #将坐标缩小到特征图尺度 x:列，y:行
            x2, y2 = (center_to[0] / stride, center_to[1] / stride)

            _put_paf_on_plane(vectormap, countmap, plane_idx, x1, y1, x2, y2,
                              threshold, height, width)
    return vectormap


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


def _put_paf_on_plane(vectormap, countmap, plane_idx, x1, y1, x2, y2,
                     threshold, height, width):
    '''绘制某两个关键点的paf'''
    min_x = max(0, int(round(min(x1, x2) - threshold)))
    max_x = min(width, int(round(max(x1, x2) + threshold)))

    min_y = max(0, int(round(min(y1, y2) - threshold)))
    max_y = min(height, int(round(max(y1, y2) + threshold)))

    vec_x = x2 - x1
    vec_y = y2 - y1

    norm = math.sqrt(vec_x ** 2 + vec_y ** 2)
    if norm < 1e-8:
        return

    vec_x /= norm
    vec_y /= norm

    for y in range(min_y, max_y):
        for x in range(min_x, max_x):
            bec_x = x - x1
            bec_y = y - y1
            dist = abs(bec_x * vec_y - bec_y * vec_x) #向量叉乘

            if dist > threshold: # ||bec||*sin(sita)>1
                continue

            cnt = countmap[y][x][plane_idx] #当前位置是否被访问过，获取访问的次数

            if cnt == 0:
                vectormap[y][x][plane_idx * 2 + 0] = vec_x #保存关键点链接向量的x分量
                vectormap[y][x][plane_idx * 2 + 1] = vec_y #保存关键点链接向量的y分量
            else: #如果当前像素值不为0，也就是重复计算该位置了，则保存平均值（基本不会运行到这里）
                vectormap[y][x][plane_idx*2+0] = (vectormap[y][x][plane_idx*2+0] * cnt + vec_x) / (cnt + 1)
                vectormap[y][x][plane_idx*2+1] = (vectormap[y][x][plane_idx*2+1] * cnt + vec_y) / (cnt + 1)

            countmap[y][x][plane_idx] += 1 #记录该位置计算过了；重复计算的话，记录下次数



if __name__=='__main__':
    # 测试生成paf标签
    all_joints=[[None, (210.37608987149815, 208.76660745267787), (247.07176559600396, 233.86344395118954), 
    (224.3260385921817, 281.4408228986688), (209.2979425620942, 320.85671006135465), (173.68041414699246, 183.66977095416615), 
     None, None, (180.00138081164914, 312.1329442433833), None, None, (125.77001708249438, 283.63005842324026),
     None, None, None, None, (260.14938264181086, 203.10278234938073), None]]
    paf=create_paf(18,46, 46,all_joints, 1, stride=8)
    ind=0
    img=paf[:,:,ind*2+0]
    print(np.max(img),np.min(img))
    img2=paf[:,:,ind*2+1]
    print(np.max(img2),np.min(img2))
    img_n=(img-np.min(img))/(np.max(img)-np.min(img))*255
    img2_n=(img2-np.min(img2))/(np.max(img2)-np.min(img2))*255
    import cv2
    cv2.imwrite("result_imgs/paf_tmpx.png",img_n)
    cv2.imwrite("result_imgs/paf_tmpy.png",img2_n)
    print("pause")