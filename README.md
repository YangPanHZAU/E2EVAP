<h2 align="center">E2EVAP: End-to-end vectorization of smallholder agricultural parcel boundaries from high-resolution remote sensing imagery</h2>
<h5 align="right">by <a>Yang Pan</a>,<a>Xingyu Wang</a>,<a href="http://rsidea.whu.edu.cn/">Yanfei Zhong</a>, and  Liangpei Zhang</h5>

![introduction](imgs/Fig.2.jpg)
This is an official implementation of E2EVAP in our ISPRS 2023 paper <a href="https://www.sciencedirect.com/science/article/pii/S0924271623002162">E2EVAP: End-to-end vectorization of smallholder agricultural parcel boundaries from high-resolution remote sensing imagery</a>


---------------------
## Citation
If you use E2EVAP in your research, please cite the following paper:
```
@article{PAN2023246,
title = {E2EVAP: End-to-end vectorization of smallholder agricultural parcel boundaries from high-resolution remote sensing imagery},
journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
volume = {203},
pages = {246-264},
year = {2023},
issn = {0924-2716},
doi = {https://doi.org/10.1016/j.isprsjprs.2023.08.001},
url = {https://www.sciencedirect.com/science/article/pii/S0924271623002162},
author = {Yang Pan and Xinyu Wang and Liangpei Zhang and Yanfei Zhong},
}
```
## Getting Started

Environment reference：<a href="https://github.com/zhang-tao-whu/e2ec/blob/main/INSTALL.md">E2EC</a>

### Prepare iflytek parcel Dataset

- Dataset download
  
All images can be download from the <a href="https://github.com/zhaozhen2333/iFLYTEK2021">top1 solution from iFLYTEK Challenge 2021</a>

- Dataset split
  
for training/valiate dataset, we follow cropping and split strategy from <a href="https://github.com/zhaozhen2333/iFLYTEK2021/blob/main/out_shp/train/pre_for_train.py">top1 solution from iFLYTEK Challenge 2021</a> 

for test dataset, we use same cropping strategy as for training but the images smaller than 512*512 are dropped.

```bash
python scripts/pre_for_train.py
python scripts/pre_for_test.py
```

### Evaluate Model
#### 1. download pretrained weight in this [link](https://drive.google.com/file/d/16IYHK63KKdv8VEOQiaw9uSSSavBfNkjT/view?usp=sharing)

#### 2. test the model
```bash
python test.py dla34_e2evap_ifly_parcel_test --checkpoint /xxxx/ckpt_ifly.pth --eval segm --device 0
```

#### 3. overlapped inference on large size remote sensing imagery
we follow the similar strategy from <a href="https://github.com/zhaozhen2333/iFLYTEK2021/blob/main/out_shp/inference/single_shp_out.py">top1 solution from iFLYTEK Challenge 2021</a> 

step1：clipping the large size imagery
```bash
python overlap_infer/cut_patch.py
```
The parameters patch size and stride can be adjusted according to the extraction result.

step2：infer the cutted images

add the metadata information about the cutted images in <a href="https://github.com/YangPanHZAU/E2EVAP/blob/main/dataset/info.py">dataset/info.py</a> 

infer the cutted images
```bash
python overlap_infer/overlap_infer.py e2evap_ifly_parcel_test_CGDZ_8_768 --checkpoint /xxxx/ckpt_ifly.pth --with_nms True --eval segm --device 0
```

step3：merge the cutted results and converted them into shp format.
```bash
python overlap_infer/merge2shp.py
```
It is necessary to specify the inferred JSON path(**segm_json**), which is different from the original JSON path(**poi_json_path**).
The main parameters for post-processing are:
 **score_thr**, **nms_mode，NMS_iou_thr**
Result path:
**shp_single_path**


### Training your Model

preparing the edge of gt
```bash
python coco2_edge_mask.py
```

training the model

python train_net.py dla34_e2evap_ifly_parcel --device 2

Find the best number of epoches on the validation set to evaluate your model.

### Visualization

visualization of ground truth
```bash
python vis_coco_gt.py
```

visualization of predicted result
```bash
python vis_coco_pred.py
```