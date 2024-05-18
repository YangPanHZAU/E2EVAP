# encoding: utf-8

import shapefile
import os
import sys
import os.path as osp
import mmcv
from itertools import chain
from osgeo import gdal
import numpy as np
from skimage import io as skio
import imgaug as ia
from imgaug.augmentables.polys import Polygon
from sklearn.model_selection import train_test_split
from tqdm import tqdm

__places__ = ['CGDZ_1', 'CGHA_19', 'CGHE_11', 'CGHY_6', 'CGJC_10', 'CGJN_13',
              'CGJN_14', 'CGLY_15', 'CGLY_16', 'CGSG_5','CGSH_1', 'CGTL_11',
              'CGWH_16', 'CGZJ_19', 'CGZJ_20', 'CGZY_22']


def lonlat2imagexy(dataset, lon, lat):
    transform = dataset.GetGeoTransform()
    x_origin = transform[0]
    y_origin = transform[3]
    pixel_width = transform[1]
    pixel_height = transform[5]
    x_pix = (lon - x_origin) / pixel_width
    y_pix = (lat - y_origin) / pixel_height
    return x_pix, y_pix


def Clip_box_lists(img_width, img_height):
    clip_box_lists = []
    w_num = 0
    h, w = [scale, scale]
    while w_num < img_width:
        left_top_width = w_num
        h_num = 0

        if w_num + w < img_width:
            right_sub_width = w_num + w
            while h_num < img_height:
                if h_num + h < img_height:
                    left_top_height = h_num
                    right_sub_height = h_num + h
                    clip_box_list = [left_top_height, right_sub_height, left_top_width, right_sub_width]
                    h_num += stride
                    clip_box_lists.append(clip_box_list)
                else:
                    left_top_height = h_num
                    right_sub_height = img_height
                    clip_box_list = [left_top_height, right_sub_height, left_top_width, right_sub_width]
                    clip_box_lists.append(clip_box_list)
                    break
            w_num += stride
        else:
            right_sub_width = img_width
            while h_num < img_height:
                if h_num + h < img_height:
                    left_top_height = h_num
                    right_sub_height = h_num + h
                    clip_box_list = [left_top_height, right_sub_height, left_top_width, right_sub_width]
                    h_num += stride
                    clip_box_lists.append(clip_box_list)
                else:
                    left_top_height = h_num
                    right_sub_height = img_height
                    clip_box_list = [left_top_height, right_sub_height, left_top_width, right_sub_width]
                    clip_box_lists.append(clip_box_list)
                    break
            break
    return clip_box_lists


def Make_train_test(img_width, img_height, test_scale):
    clip_boxes = {'train': [], 'test': []}
    clip_box_lists = Clip_box_lists(img_width, img_height)

    box_lists_trains, box_lists_tests = train_test_split(clip_box_lists, test_size=test_scale, random_state=128)
    clip_boxes['train'] = box_lists_trains
    clip_boxes['test'] = box_lists_tests
    return clip_boxes


def Preprocess(tif_path, shp_path):
    """
    :param tif_path: 原始图片的宽
    :param shp_path: 原始图片的高
    :return: dict{'train': list[list[y0, y1, x0, x1],...], 'test': list[list[y0, y1, x0, x1],...]}
    """
    # 1 读取栅格数据
    dataset = gdal.Open(tif_path)
    img_width = dataset.RasterXSize  # 栅格矩阵的列数
    img_height = dataset.RasterYSize  # 栅格矩阵的行数
    img_bands = dataset.RasterCount  # 波段数
    img_data_type = gdal.GetDataTypeName(dataset.GetRasterBand(1).DataType)  # 原始数据类型

    # 2 对影像对应的实例分割坐标集进行坐标变换
    polygon_list = []
    reader = shapefile.Reader(shp_path)
    index = 0

    for sr in reader.shapeRecords():
        geom = sr.shape.__geo_interface__
        feature_points = geom["coordinates"][0]
        xy_points_list = []
        for lonlat in feature_points:
            xy = lonlat2imagexy(dataset, float(lonlat[0]), float(lonlat[1]))
            xy_points_list.append(xy)
        polygon_list.append(Polygon(xy_points_list))
        index += 1
    return dataset, img_width, img_height, polygon_list


def data_process():
    """
    生成coco格式的数据集
    """
    image_id = 0
    annotation_id = 0
    train_coco = dict()
    train_coco['images'] = []
    train_coco['type'] = 'instance'
    train_coco['categories'] = []
    train_coco['annotations'] = []

    test_coco = dict()
    test_coco['images'] = []
    test_coco['type'] = 'instance'
    test_coco['categories'] = []
    test_coco['annotations'] = []

    train_json = osp.join(json_path, f"train.json")
    test_json = osp.join(json_path, f"test.json")


    category_item = dict()
    category_item['supercategory'] = str('none')
    category_item['id'] = int(0)
    category_item['name'] = str('CultivatedLand')
    train_coco['categories'].append(category_item)
    test_coco['categories'].append(category_item)

    for place in tqdm(__places__):
        tif_path = osp.join(tif_img_path, f"{place}_offset.tif")
        shp_path = osp.join(shp_label_path, f"{place}_offset.shp")
        train_img = osp.join(train_img_path, f"{place}_offset")
        test_img = osp.join(test_img_path, f"{place}_offset")

        dataset, img_width, img_height, polygon_list = Preprocess(tif_path, shp_path)


        clip_box_lists = Make_train_test(img_width, img_height, train_test_scale)

        train_image_set = set()

        # 2 训练集生成
        for i_clip, clip_box in enumerate(clip_box_lists['train']):
            xmin = clip_box[2]
            ymin = clip_box[0]
            width = (clip_box[3] - clip_box[2])
            height = (clip_box[1] - clip_box[0])

            # 存储图片
            out_img_path = train_img + f"_{i_clip}.jpg"
            img_data_int8 = dataset.ReadAsArray(xmin, ymin, width, height).astype(np.uint8)  # 获取分块数据
            img_data = np.transpose(img_data_int8, (1, 2, 0))[:, :, [2, 1, 0]]
            skio.imsave(out_img_path, img_data)

            polygon_list_shift = list(map(lambda x: x.shift(x=-xmin, y=-ymin), polygon_list))
            psoi = ia.PolygonsOnImage(polygon_list_shift, shape=(height, width))

            psoi_aug = psoi.clip_out_of_image()
            aug_polygon_list = psoi_aug.polygons


            file_name = os.path.basename(out_img_path)
            assert file_name not in train_image_set
            image_item = dict()
            image_item['id'] = int(image_id)
            image_item['file_name'] = str(file_name)
            image_item['height'] = height
            image_item['width'] = width
            train_coco['images'].append(image_item)
            train_image_set.add(file_name)

            # 点位数据
            if len(aug_polygon_list) != 0:
                for aug_polygon in aug_polygon_list:
                    annotation_item = dict()
                    xx_list = aug_polygon.xx.tolist()
                    yy_list = aug_polygon.yy.tolist()
                    seg_list = list(chain.from_iterable(zip(xx_list, yy_list)))
                    x_min = min(xx_list)
                    x_max = max(xx_list)
                    y_min = min(yy_list)
                    y_max = max(yy_list)
                    width = x_max - x_min
                    height = y_max - y_min

                    annotation_item["segmentation"] = [seg_list]
                    annotation_item["area"] = aug_polygon.area
                    annotation_item['ignore'] = 0
                    annotation_item['iscrowd'] = 0
                    annotation_item['image_id'] = int(image_id)
                    annotation_item["bbox"] = [x_min, y_min, width, height]
                    annotation_item['category_id'] = int(0)
                    annotation_item['id'] = int(annotation_id)
                    train_coco['annotations'].append(annotation_item)
                    annotation_id += 1
            image_id += 1

        # 3 验证集生成
        N = len(clip_box_lists['train'])
        test_image_set = set()
        for i_clip, clip_box in enumerate(clip_box_lists['test']):
            xmin = clip_box[2]
            ymin = clip_box[0]
            width = (clip_box[3] - clip_box[2])
            height = (clip_box[1] - clip_box[0])

            # 存储图片
            i_clip = i_clip + N
            out_img_path = test_img + f"_{i_clip}.jpg"
            img_data_int8 = dataset.ReadAsArray(xmin, ymin, width, height).astype(np.uint8)
            img_data = np.transpose(img_data_int8, (1, 2, 0))[:, :, [2, 1, 0]]
            skio.imsave(out_img_path, img_data)

            polygon_list_shift = list(map(lambda x: x.shift(x=-xmin, y=-ymin), polygon_list))
            psoi = ia.PolygonsOnImage(polygon_list_shift, shape=(height, width))

            psoi_aug = psoi.clip_out_of_image()
            aug_polygon_list = psoi_aug.polygons

            file_name = os.path.basename(out_img_path)
            assert file_name not in test_image_set
            image_item = dict()
            image_item['id'] = int(image_id)
            image_item['file_name'] = str(file_name)
            image_item['height'] = height
            image_item['width'] = width
            test_coco['images'].append(image_item)
            test_image_set.add(file_name)

            if len(aug_polygon_list) != 0:
                for aug_polygon in aug_polygon_list:
                    annotation_item = dict()
                    xx_list = aug_polygon.xx.tolist()
                    yy_list = aug_polygon.yy.tolist()
                    seg_list = list(chain.from_iterable(zip(xx_list, yy_list)))
                    x_min = min(xx_list)
                    x_max = max(xx_list)
                    y_min = min(yy_list)
                    y_max = max(yy_list)
                    width = x_max - x_min
                    height = y_max - y_min

                    annotation_item["segmentation"] = [seg_list]
                    annotation_item["area"] = aug_polygon.area
                    annotation_item['ignore'] = 0
                    annotation_item['iscrowd'] = 0
                    annotation_item['image_id'] = int(image_id)
                    annotation_item["bbox"] = [x_min, y_min, width, height]
                    annotation_item['category_id'] = int(0)
                    annotation_item['id'] = int(annotation_id)
                    test_coco['annotations'].append(annotation_item)

                    annotation_id += 1
            image_id += 1
    mmcv.dump(train_coco, train_json)
    mmcv.dump(test_coco, test_json)


if __name__ == '__main__':
    # window size scale，stride
    data_root = '512-512'
    scale, stride = map(int, data_root.split('-'))

    train_test_scale = 1 / 6

    tif_img_path = '/home/py21/changguang_parcel/chusai_train/image/'
    shp_label_path = '/home/py21/changguang_parcel/chusai_train/label/'
    train_img_path = f'out_shp/train/{data_root}/train/'
    test_img_path = f'out_shp/train/{data_root}/test/'
    json_path = f'out_shp/train/{data_root}/annotations/'
    os.makedirs(train_img_path, exist_ok=True)
    os.makedirs(test_img_path, exist_ok=True)
    os.makedirs(json_path, exist_ok=True)
    data_process()
