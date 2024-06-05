import sys
sys.path.append('./')
import math
import numpy as np
from osgeo import gdal
from tqdm import tqdm
import os
import json
def GDAL_read_Img(fileName: str):
    dataset = gdal.Open(fileName)
    if dataset == None:
        print(fileName + "文件无法打开")
    return dataset
def imagexy2geo(dataset, row, col):
    '''
    根据GDAL的六参数模型将影像图上坐标（行列号）转为投影坐标或地理坐标（根据具体数据的坐标系统转换）
    :param dataset: GDAL地理数据
    :param row: 像素的行号
    :param col: 像素的列号
    :return: 行列号(row, col)对应的投影坐标或地理坐标(x, y)
    '''
    trans = dataset.GetGeoTransform()
    px = trans[0] + col * trans[1] + row * trans[2]
    py = trans[3] + col * trans[4] + row * trans[5]
    return px, py
def get_patch_geotrans(dataset,px,py):
    im_geotrans = list(dataset.GetGeoTransform())  # 仿射矩阵
    im_geotrans[0],im_geotrans[3]=px,py
    patch_im_geotrans = tuple(im_geotrans)
    return patch_im_geotrans
def write_geotiff(im_data, im_geotrans, im_proj, path):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    if len(im_data.shape) == 3:
        im_data = np.transpose(im_data, axes=(2, 0, 1))#[h,w,c]->[c,h,w]
        im_bands, im_height, im_width = im_data.shape
    elif len(im_data.shape) == 2:
        im_data = np.array([im_data])
        im_bands, im_height, im_width = im_data.shape
    # 创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, int(im_width), int(im_height), int(im_bands), datatype)
    if (dataset != None):
        dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
        dataset.SetProjection(im_proj)  # 写入投影
    for i in range(im_bands):
        dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset
def create_sliding_window_box(input_size, kernel_size, stride: int):
    """
    :param input_size:
    :param kernel_size:
    :param stride:
    :return:
    """
    ih, iw = input_size
    kh, kw = kernel_size
    assert ih > 0 and iw > 0 and kh > 0 and kw > 0 and stride > 0

    kh = ih if kh > ih else kh
    kw = iw if kw > iw else kw

    num_rows = math.ceil((ih - kh) / stride) if math.ceil((ih - kh) / stride) * stride + kh >= ih else math.ceil(
        (ih - kh) / stride) + 1
    num_cols = math.ceil((iw - kw) / stride) if math.ceil((iw - kw) / stride) * stride + kw >= iw else math.ceil(
        (iw - kw) / stride) + 1

    x, y = np.meshgrid(np.arange(num_cols + 1), np.arange(num_rows + 1))
    xmin = x * stride
    ymin = y * stride

    xmin = xmin.ravel()
    ymin = ymin.ravel()
    xmin_offset = np.where(xmin + kw > iw, iw - xmin - kw, np.zeros_like(xmin))
    ymin_offset = np.where(ymin + kh > ih, ih - ymin - kh, np.zeros_like(ymin))
    boxes = np.stack([xmin + xmin_offset, ymin + ymin_offset,
                      np.minimum(xmin + kw, iw), np.minimum(ymin + kh, ih)], axis=1)

    return boxes
def patch_cropping_img(img_path: str,out_img_dir: str,out_json_path: str,Patch_size,stride,coco_dict):
    dataset = GDAL_read_Img(img_path)
    img_width, img_height = dataset.RasterXSize, dataset.RasterYSize
    im_proj = dataset.GetProjection()
    boxes = create_sliding_window_box([img_height,img_width],kernel_size=Patch_size,stride=stride)
    image_patch_index=0
    poly_index=0
    for clip_box in tqdm(boxes):
        xmin = int(clip_box[0])
        ymin = int(clip_box[1])
        width = int(clip_box[2] - clip_box[0])
        height = int(clip_box[3] - clip_box[1])
        file_name = os.path.basename(img_path)[:-4]
        image_patch_id=file_name+"_Sx_{}_Sy_{}_Ex_{}_Ey_{}".format(clip_box[0],clip_box[1],clip_box[2],clip_box[3])
        img_data_int8 = np.transpose(dataset.ReadAsArray(xmin, ymin, width, height),(1, 2, 0))[:, :, [2, 1, 0]]  # 获取分块数据#[c,h,w]->[h,w,c]
        px,py=imagexy2geo(dataset,ymin, xmin)
        patch_im_geotrans = get_patch_geotrans(dataset, px, py)
        out_img_path = os.path.join(out_img_dir,image_patch_id+'.tif')
        #write_tiff(img_data_int8,out_img_path)
        write_geotiff(img_data_int8,patch_im_geotrans,im_proj,out_img_path)
        coco_dict['images'].append(dict(
            id=image_patch_index,
            file_name=image_patch_id+'.tif',
            height=height,
            width=width))
        coco_dict['annotations'].append(dict(
            image_id=image_patch_index,
            id=poly_index,
            category_id=1,
            bbox=[0, 0, 1, 1],
            area=1.0,
            segmentation=[[0,1,2,1,4,3]],
            iscrowd=0
        ))
        poly_index += 1
        image_patch_index +=1
    with open(out_json_path, 'w') as output_json_file:
        json.dump(coco_dict, output_json_file, indent=4)

if __name__ == "__main__":
    file_path=r'/home/py21/changguang_parcel/chusai_test/image/'
    file_name='CGDZ_8_offset.tif'
    img_path = os.path.join(file_path,file_name)
    out_img_dir =  os.path.join('test_overlap_infer_region',file_name[:-4])
    os.makedirs(out_img_dir,exist_ok=True)
    out_json_path = os.path.join(out_img_dir,'test_region_coco.json')
    Patch_size=[768,768]
    stride=608
    coco_dict=dict(
        images=[],
        annotations=[],
        categories=[{'id':1, 'name': 'cropland_parcel'}]
    )
    patch_cropping_img(img_path,out_img_dir,out_json_path,Patch_size,stride,coco_dict)