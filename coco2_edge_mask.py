import time
import numpy as np
import os
import cv2
from skimage import io
from pycocotools.coco import COCO
import pycocotools.mask as mask_util
from tqdm import tqdm
EDGE_THICKNESS = 3

def polygons_to_bitmask(polygons,height,width):
    """
    Args:
        polygons (list[ndarray]): each array has shape (Nx2,)
        height, width (int)

    Returns:
        ndarray: a bool mask of shape (height, width)
    """
    if len(polygons) == 0:
        # COCOAPI does not support empty polygons
        return np.zeros((height, width)).astype(np.bool)
    rles = mask_util.frPyObjects(polygons, height, width)
    rle = mask_util.merge(rles)
    return mask_util.decode(rle)#.astype(np.bool)

def generate_coco_edge_map_from_json(instance_json, edge_root):
    os.makedirs(edge_root, exist_ok=True)

    coco = COCO(instance_json)
    img_ids = np.array(sorted(coco.getImgIds()))
    print(img_ids)
    for img_id in img_ids:
        # file_name = coco.loadImgs(int(img_id))[0]['file_name']
        # print(file_name)
        img_info = coco.loadImgs(int(img_id))[0]
        file_name = img_info['file_name']
        width = img_info['width']
        height = img_info['height']
        image_id = img_id
        ann_ids = coco.getAnnIds(imgIds=image_id, iscrowd=0)
        #file_name = coco.loadImgs(int(image_id))[0]['file_name']
        anno = coco.loadAnns(ann_ids)
        edge_result=np.zeros((height,width),'uint8')
        tmp_result=np.zeros((height,width),'uint8')
        for ann_in in anno:
            ann_array=[ann_in['segmentation'][0]]
            mask_raw=polygons_to_bitmask(ann_array,height,width)
            contours, hierarchy = cv2.findContours(
            mask_raw,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(tmp_result, contours, -1, 1, EDGE_THICKNESS)
            edge_result+=tmp_result
        edge_result[edge_result>0]=1
        io.imsave(os.path.join(edge_root,file_name).replace('.jpg','.tif'),edge_result)
    print("Start writing to {} ...".format(edge_root))
    start = time.time()
    print("Finished. time: {:.2f}s".format(time.time() - start))


if __name__ == "__main__":
    dataset_dir = '/home/py21/changguang_parcel/top1_baseline/out_shp/train/512-512/'
    generate_coco_edge_map_from_json(
        os.path.join(dataset_dir, "annotations/train.json"),
        os.path.join(dataset_dir, "ifly_edge_train_thick3")
    )