import torch
import os
from locationclass import img2numpy, save_imgs, drawimage, imgpre_detect
from models.common import DetectMultiBackend

from concurrent.futures import ThreadPoolExecutor
# executor = ThreadPoolExecutor(2)  # 线程池


def detect(img_path,weights_path):
    # img2numpy
    image = img2numpy(img_path)
    # get image's name
    img_name = img_path.split('/')[-1]

    # copy image
    img_copy = image.copy()

    # load model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = DetectMultiBackend(weights_path, device)

    # predict
    res, result_list = imgpre_detect(image, model, device)

    # draw rectangles
    if res:
        # exist goals --> draw rectangular boxes --> save images
        drawed_images = drawimage(img_copy, result_list)
        save_imgs(img_copy, 'detected_imgs', img_name, imgres=drawed_images)
    else:
        # no exist goals --> save original images
        save_imgs(img_copy, 'detected_imgs', img_name, imgres=None)


if __name__ == '__main__':
    # need detected images path
    img_path = r"D:/ys_projects/safe_hat_yolov5_detect/data/"
    # weigths path
    weights_path = r'./weights/safe_hat.pt'

    for imgs in os.listdir(img_path):
        imgs_path = os.path.join(img_path, imgs)
        print(imgs_path)
        detect(imgs_path,weights_path)
