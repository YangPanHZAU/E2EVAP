import numpy as np
import json
from pandas import DataFrame
import cv2
from skimage import measure
from pycocotools import mask as mask_util
from shapely.geometry import Polygon
from osgeo import gdal, ogr, osr
import os.path as osp
import torch
import pycocotools.mask as mask_utils
from tqdm import tqdm
def xy2lonlat(points, tif_single_path):
    """
    图像坐标转经纬度坐标
    """
    dataset = gdal.Open(tif_single_path)
    transform = dataset.GetGeoTransform()
    lonlats = np.zeros(points.shape)

    lonlats[:, 0] = transform[
        0] + points[:, 0] * transform[1] + points[:, 1] * transform[2]
    lonlats[:, 1] = transform[
        3] + points[:, 0] * transform[4] + points[:, 1] * transform[5]
    return lonlats
def id_to_filename(path):
    """
    建立剪切图片的 id 与 file_name 的转换字典
    数据集为coco数据集格式，利用 mmdetection 框架进行预测时得到的预测文件只包含图片的id信息
    :param path: json的路径
    :return: dict{id : file_name}
    """
    test_jsons = json.load(open(path))
    images = test_jsons['images']
    Frame = DataFrame(images)
    file_name = np.array(Frame['file_name']).tolist()
    id = np.array(Frame['id']).tolist()
    images_dict = dict(zip(id, file_name))
    return images_dict
def close_contour(contour):
    """
    闭合提取到的边界
    """
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour
def binary_mask_to_polygon(binary_mask):
    """
    提取预测mask的边界
    """
    areas = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(
        binary_mask, pad_width=1, mode='constant', constant_values=0)
    contours, _ = cv2.findContours(
        padded_binary_mask,
        mode=cv2.RETR_EXTERNAL,
        method=cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        areas.append(area)

    assert len(areas) == len(contours)
    if len(areas) > 0:
        max_index = areas.index(max(areas))
        contour = np.squeeze(contours[max_index], axis=1)
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance=0)
        polygons = contour.tolist()
    else:
        polygons = []

    return polygons
class GDAL_shp_Data(object):

    def __init__(self, shp_single_path):
        self.shp_single_path = shp_single_path
        self.shp_file_create()

    def shp_file_create(self):
        gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "YES")
        gdal.SetConfigOption("SHAPE_ENCODING", "GBK")
        ogr.RegisterAll()
        driver = ogr.GetDriverByName("ESRI Shapefile")
        # 打开输出文件及图层
        # 输出模板shp 包含待写入的字段信息
        self.outds = driver.CreateDataSource(self.shp_single_path)
        # 创建空间参考
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        # 创建图层
        self.out_layer = self.outds.CreateLayer("out_polygon", srs,
                                                ogr.wkbPolygon)
        field_name = ogr.FieldDefn("scores", ogr.OFTReal)
        self.out_layer.CreateField(field_name)

    def set_shapefile_data(self, polygons, scores):
        for i in range(len(scores)):
            wkt = polygons[i].wkt  # 创建wkt文本点
            temp_geom = ogr.CreateGeometryFromWkt(wkt)
            feature = ogr.Feature(self.out_layer.GetLayerDefn())  # 创建特征
            feature.SetField("scores", scores[i])
            feature.SetGeometry(temp_geom)
            self.out_layer.CreateFeature(feature)
        self.finish_io()

    def finish_io(self):
        del self.outds
def arrange(score,bbox,xymin,scale,stride):
    """
    运行边界筛选法筛选满足要求的预测结果
    score： tensor(N,1)，预测mask结果的置信度
    bbox: tensor(N,4)，预测mask结果的bounding box
    xymin: tensor(N,2)，小图左上角顶点的坐标
    [score, imageHeight, imageWidth, xmin, ymin, bboxx, bboxy]
    score： tensor(N,1)，预测mask结果的置信度
    imageHeight, imageWidth： tensor(N,1)，小图的高和宽
    xmin, ymin： tensor(N,1)，小图左上角顶点的坐标
    bboxx, bboxy： tensor(N,1)，预测结果的bbox左上角顶点的坐标
    """
    # scale=512
    # stride=416
    imageHeight = torch.tensor((np.ones_like(score)*scale).tolist()).float()  # tensor(N,1)
    imageWidth = torch.tensor((np.ones_like(score)*scale).tolist()).float()  # tensor(N,1)
    score = torch.tensor(score)  # score: tensor(N,1)
    bbox = torch.tensor(bbox)  # bbox: tensor(N,4)
    bboxx = bbox[:, 0]  # bboxx: tensor(N,1)
    bboxy = bbox[:, 1]  # bboxy: tensor(N,1)
    xymin=torch.tensor(xymin)
    xmin = xymin[:,0]#coordinate[:, :, 0]  # tensor(N,1)
    xmin = xmin - min(xmin)
    ymin = xymin[:,1]  # tensor(N,1)

    # coordinate = torch.tensor(np.array(Frame[['coordinate']]).tolist()).float()  # tensor(N,1,2)

    # xmin = coordinate[:, :, 0]  # tensor(N,1)
    # xmin = xmin - min(xmin)
    # ymin = coordinate[:, :, 1]  # tensor(N,1)

    # 3 将数据cat为矩阵形式进行计算，以提高计算速度
    judge_condition = torch.cat(
        [score[:,None], imageHeight[:,None], imageWidth[:,None], xmin[:,None], ymin[:,None], bboxx[:,None], bboxy[:,None]], dim=1)  # tensor(N,6)

    # 4 运用边界筛选法
    keep_mask_0 = (judge_condition[:, 0] > score_thr)  # 对score进行筛选
    keep_inds_0 = torch.nonzero(keep_mask_0, as_tuple=False).reshape(-1)
    judge_condition_1 = judge_condition[keep_mask_0]

    keep_mask_1 = (judge_condition_1[:, 1] == scale) & (judge_condition_1[:, 2] == scale) & \
                  (((judge_condition_1[:, 3] != 0) & (judge_condition_1[:, 4] != 0) &
                    (2 <= judge_condition_1[:, 5]) & (judge_condition_1[:, 5] <= stride) &
                    (2 <= judge_condition_1[:, 6]) & (judge_condition_1[:, 6] <= stride)) |
                   ((judge_condition_1[:, 3] != 0) & (judge_condition_1[:, 4] == 0) &
                    (2 <= judge_condition_1[:, 5]) & (0 <= judge_condition_1[:, 6]) &
                    (judge_condition_1[:, 5] <= stride) & (judge_condition_1[:, 6] <= stride)) |
                   ((judge_condition_1[:, 3] == 0) & (judge_condition_1[:, 4] != 0) &
                    (0 <= judge_condition_1[:, 5]) & (2 <= judge_condition_1[:, 6]) &
                    (judge_condition_1[:, 5] <= stride) & (judge_condition_1[:, 6] <= stride)) |
                   ((judge_condition_1[:, 3] == 0) & (judge_condition_1[:, 4] == 0) &
                    (0 <= judge_condition_1[:, 5]) & (0 <= judge_condition_1[:, 6]) &
                    (judge_condition_1[:, 5] <= stride) & (judge_condition_1[:, 6] <= stride)))
    # # 默认原始place图片下边界剪切得到的小图的高imageHeight小于标准高scale
    # keep_mask_2 = (judge_condition_1[:, 1] < scale) & (judge_condition_1[:, 2] == scale) & \
    #               (((judge_condition_1[:, 3] == 0) &
    #                 (0 <= judge_condition_1[:, 5]) & (2 <= judge_condition_1[:, 6]) &
    #                 (judge_condition_1[:, 5] <= stride)) |
    #                ((judge_condition_1[:, 3] != 0) &
    #                 (2 <= judge_condition_1[:, 5]) & (2 <= judge_condition_1[:, 6]) &
    #                 (judge_condition_1[:, 5] <= stride)))
    # # 默认原始place图片右边界剪切得到的小图的宽imageWidth小于标准宽scale
    # keep_mask_3 = (judge_condition_1[:, 1] == scale) & (judge_condition_1[:, 2] < scale) & \
    #               (((judge_condition_1[:, 4] == 0) &
    #                 (0 <= judge_condition_1[:, 6]) & (2 <= judge_condition_1[:, 5]) &
    #                 (judge_condition_1[:, 6] <= stride)) |
    #                ((judge_condition_1[:, 4] != 0) &
    #                 (2 <= judge_condition_1[:, 6]) & (2 <= judge_condition_1[:, 5]) &
    #                 (judge_condition_1[:, 6] <= stride)))
    # # 则原始place图片剪切得到的右下角位置的最后一张小图的高和宽同时小于标准高和宽scale
    # keep_mask_4 = (judge_condition_1[:, 1] < scale) & (judge_condition_1[:, 2] < scale) & \
    #               ((2 <= judge_condition_1[:, 6]) & (2 <= judge_condition_1[:, 5]))

    # # 5 合并筛选结果，并保存结果到select_place.json
    # keep_mask = keep_mask_1 | keep_mask_2 | keep_mask_3 | keep_mask_4
    keep_mask = keep_mask_1

    keep_inds = torch.nonzero(keep_mask, as_tuple=False).reshape(-1)
    return keep_inds
def coco_poly_to_rle(poly, h, w):
    rle_ = []
    for i in range(len(poly)):
        rles = mask_utils.frPyObjects([poly[i].reshape(-1)], h, w)
        rle = mask_utils.merge(rles)
        rle['counts'] = rle['counts'].decode('utf-8')
        rle_.append(rle)
    return rle_
if __name__ == "__main__":
    segm_json =' test_overlap_infer_region/CGDZ_8_offset_result/result_coco/results.json'
    poi_json_path = 'test_overlap_infer_region/CGDZ_8_offset/test_region_coco.json'
    score_thr=0.3
    Patch=768
    Stride=608
    nms_mode='NMS'
    NMS_iou_thr=0.3
    boundary_select=True
    tif_single_path = r'/home/py21/changguang_parcel/chusai_test/image/CGDZ_8_offset.tif'
    shp_single_path = r'test_overlap_infer_region/CGDZ_8_offset_result/test_region_overlap_{}_{}_score_03_nms03.shp'.format(Patch,Stride)  # .shp文件输出路径
    # 1 读取 mmdetection 的预测文件segm.json
    segm_jsons = json.load(open(segm_json))
    # 2 读取伪标签test.json，得到id to file_name的转换字典
    images_dict = id_to_filename(poi_json_path)
    
    box_all=[]#所有地块对象的包围框在原始大图范围下的坐标[x1,y1,x2,y2]
    label_list=[]#所有地块对象的类别
    poly_list=[]#所有地块对象的多边形坐标
    score_list=[]#所有地块对象的得分
    mask_list=[]#所有地块对象的mask
    xymin_list=[]#每个地块所在的patch处于原始大图的左上角坐标
    sub_box_all=[]#每个地块所在的patch范围内的boxes坐标[x1,y1,x2,y2]
    for segm_json in tqdm(segm_jsons):
        image_id = segm_json['image_id']
        file_name = images_dict[image_id]
        split_sx,split_sy,split_ex,split_ey=file_name.find('_Sx_'),file_name.find('_Sy_'),file_name.find('_Ex_'),file_name.find('_Ey_')
        xmin,ymin=eval(file_name[split_sx+4:split_sy]),eval(file_name[split_sy+4:split_ex])
        rle=segm_json['segmentation']
        polygons = segm_json['seg_coord']
        label = segm_json['category_id']
        score = segm_json['score']
        if len(polygons) > 3:
            if score > score_thr:
                polygon_data_np = np.array(polygons,dtype='object')
                polygon_data_np[polygon_data_np<0]=0
                polygon_data_np[polygon_data_np>Patch]=Patch
                sub_object_xmax,sub_object_ymax=np.max(polygon_data_np,axis=0)
                sub_object_xin,sub_object_ymin=np.min(polygon_data_np,axis=0)
                sub_box_all.append([sub_object_xin,sub_object_ymin,sub_object_xmax,sub_object_ymax])
                polygon_data_np = np.array([xmin,ymin])+polygon_data_np# 每个polygons坐标还原回原位置
                poly_list.append(polygon_data_np)
                score_list.append(score)
                label_list.append(label)
                xymin_list.append([xmin,ymin])
                binary_mask = mask_util.decode(rle)  # segmentation类型转换为binary mask类型
                mask_list.append(binary_mask)
                object_xmax,object_ymax=np.max(polygon_data_np,axis=0)
                object_xin,object_ymin=np.min(polygon_data_np,axis=0)
                box_all.append([object_xin,object_ymin,object_xmax,object_ymax])
                
    if boundary_select:
        keep_inds=arrange(score_list,sub_box_all,xymin_list,Patch,Stride)
        new_poly_list=[poly_list[i] for i in keep_inds]
        new_scores_list=[score_list[i] for i in keep_inds]
        new_box_result=[box_all[i] for i in keep_inds]
    else:
        new_poly_list=poly_list
        new_scores_list=score_list
        new_box_result=box_all

    
    import torch
    
    if nms_mode=='NMS':
        boxes_th, scores_th=torch.tensor(new_box_result),torch.tensor(new_scores_list)
        import torchvision
        keep_inds = torchvision.ops.nms(boxes_th,scores_th,NMS_iou_thr)
        keep_inds = keep_inds.numpy()
        new_poly_list1=[]
        new_scores_list1=[]
        for i in keep_inds:
            new_scores_list1.append(new_scores_list[i])
            polygon_lonlat = Polygon(xy2lonlat(new_poly_list[i], tif_single_path))
            new_poly_list1.append(polygon_lonlat)

    
    shp_data = GDAL_shp_Data(shp_single_path)
    shp_data.set_shapefile_data(new_poly_list1, new_scores_list1)  # 输出为符合要求的.shp文件