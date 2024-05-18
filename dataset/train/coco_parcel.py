import sys
sys.path.append('./')
from dataset.train.base import Dataset
import os
from skimage import io
import numpy as np

class CocoDataset_parcel(Dataset):
    def process_info(self, ann):
        image_id = ann
        ann_ids = self.coco.getAnnIds(imgIds=image_id, iscrowd=0)
        image_path = os.path.join(self.data_root, self.coco.loadImgs(int(image_id))[0]['file_name'])
        ann = self.coco.loadAnns(ann_ids)
        return ann, image_path, image_id

    def read_original_data(self, anno, image_path):
        img = io.imread(image_path)
        instance_polys = [[np.array(poly).reshape(-1, 2) for poly in instance['segmentation']] for instance in anno
                          if not isinstance(instance['segmentation'], dict)]
        cls_ids = [self.json_category_id_to_continuous_id[instance['category_id']] for instance in anno]
        return img, instance_polys, cls_ids

if __name__ == "__main__":
    from configs.resnet50_e2ec_ifly_parcel import config
    from dataset.train import coco, cityscapes, cityscapesCoco, sbd, kitti,coco_parcel
    from dataset.info import DatasetInfo
    cfg=config
    dataset_name = cfg.train.dataset
    info = DatasetInfo.dataset_info[dataset_name]
    dataset = CocoDataset_parcel(info['anno_dir'], info['image_dir'], 'train', cfg)
    #make_data_loader()
    result=dataset[0]
    
    for key in result:
        print(key)
    
    #print(result)
# inp,[3,512,512]
# ct_hm [1,128,128]
# wh 16个[w,h]
# ct_cls
# ct_ind
# img_gt_polys,[128,2]
# can_gt_polys,[128,2]
# keypoints_mask,16个[128,]
# meta {}
#'center':array[256,256]
#'scale':array[512,512]
#'img_id':0
#'ann':0
#'ct_num':16