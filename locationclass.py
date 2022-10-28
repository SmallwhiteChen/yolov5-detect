import numpy as np
import cv2
import torch
from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import check_img_size, non_max_suppression
import os
import time
from PIL import Image, ImageDraw, ImageFont
import operator
from concurrent.futures import ThreadPoolExecutor

'''image 2 numpy'''


def img2numpy(img_path):
    image = cv2.imread(img_path)
    image_arr = np.array(image)
    return image_arr


'''save img'''


def save_imgs(image, img_path_name, img_name, imgres=None):
    if not os.path.exists(img_path_name):
        os.makedirs(img_path_name)
    ph = os.path.join(img_path_name, time.strftime("%Y-%m-%d", time.localtime()))
    if not os.path.exists(ph):
        os.makedirs(ph)

    if imgres:
        # 有目标保存已画框的图
        imgres.save(ph + '/r_%s.jpg' % img_name[:-4])
        print("----detected images had saved---- ")
    else:
        # 没有目标返回原图
        cv2.imwrite(ph + '/r_%s.jpg' % img_name[:-4], image)
        print("----images have not goals----")


'''draw images'''
def drawimage(image, res_list):
    fontScale = int(image.shape[1] / 270.0)  # 字体比例
    # dist_ = {"0": "xx", "1": "yy","2": "zz","3": "qq",.......}
    dist_ = {"0": "safe_hat", "1": "no_safe_hat"}

    for item in res_list:
        # print("坐标 + 类别：", item)  # [(124, 77, 144, 98), '1']
        point = item[0]  # 坐标
        first_point = (int(point[0]), int(point[1]))
        last_point = (int(point[2]), int(point[3]))
        if item[1] == '0':
            cv2.rectangle(image, first_point, last_point, (0, 255, 0), 2, lineType=cv2.LINE_AA)  # 在图片上进行绘制框
            cv2.putText(image, dist_[item[1]], first_point, cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5, color=(0, 255, 0),
                        thickness=2)  # 在矩形框上方绘制该框的名称
        else:
            cv2.rectangle(image, first_point, last_point, (0, 0, 255), 2, lineType=cv2.LINE_AA)  # 在图片上进行绘制框
            cv2.putText(image, dist_[item[1]], first_point, cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5, color=(0, 0, 255),
                        thickness=1)  # 在矩形框上方绘制该框的名称
    image_copy = np.asarray(image)
    image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)

    image_copy = Image.fromarray(image_copy)
    # print(image_copy)
    draw = ImageDraw.Draw(image_copy)

    if 2 <= fontScale < 4:
        fontScale = 2.5
    elif fontScale < 2:
        fontScale = 1.5

    font = ImageFont.truetype("./font_type/simhei.ttf", int(fontScale * 10), encoding="utf-8")
    # draw.text((180, image.shape[0] - 150), " %s" % dist_[collection], (0, 255, 0), font=font)
    draw.text((10, image.shape[0] - 75), " %s" % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), (0, 255, 0),
              font=font)
    # return image_copy, collection
    return image_copy


def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2


'''图片还原'''
def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


#
def imgpre_detect(image, model, device):
    img_size = (640, 640)
    conf_thres = 0.15
    iou_thres = 0.4

    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(img_size, s=stride)  # 检查图片尺寸
    model.warmup(imgsz=(1, 3, *imgsz))  # warmup
    # image 预处理
    img = letterbox(image, img_size, stride=stride, auto=pt)[0]  # 保持长宽比进行缩放，剩余部分用灰色填充为正方形
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(img)
    # fp16/32
    im = torch.from_numpy(im).to(device)
    im = im.half() if False else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim

    # Inference/前向推理 predict ---------------------------------------
    pred = model(im, augment=False, visualize=False)
    # NMS去重
    pred = non_max_suppression(pred, conf_thres, iou_thres)
    res_list = []
    for i, det in enumerate(pred):
        print("det =", det)  # 四个坐标值 + obj + cls
        # print("det[:, -1] =",det[:, -1])
        if len(det):
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
            # Rescale boxes from img_size to im0 size（图片还原）
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], image.shape).round()
            cls = ''
            for *box, conf, tensorcls in reversed(det):
                # print("tensorcls =",tensorcls)
                intcls = int(tensorcls)
                if intcls == 0:
                    cls = "0"
                elif intcls == 1:
                    cls = "1"
                # elif intcls == 2:
                #     cls = '2'
                res_list.append([(int(box[0]), int(box[1]), int(box[2]), int(box[3])), cls])
    # print("len(res_list) =",len(res_list))
    if len(res_list) == 0:
        return False, "None"
    print("----locations and cls_index are :", res_list)
    return True, res_list
