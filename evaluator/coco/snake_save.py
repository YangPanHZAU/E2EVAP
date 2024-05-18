import os
import json
from ..sbd import utils
import pycocotools.coco as coco
import torch

class Evaluator:
    def __init__(self, result_dir, anno_dir):
        self.results = []
        self.img_ids = []
        self.aps = []

        self.result_dir = result_dir
        os.system('mkdir -p {}'.format(self.result_dir))

        ann_file = anno_dir
        self.coco = coco.COCO(ann_file)

        self.json_category_id_to_contiguous_id = {
            v: i for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }

    def evaluate(self, output, batch,thr):
        detection = output['detection']
        score = detection[:, 2].detach().cpu().numpy()
        label = detection[:, 3].detach().cpu().numpy().astype(int)
        py=output['py'][-1].detach()
        box = torch.cat([torch.min(py, dim=1, keepdim=True)[0], torch.max(py, dim=1, keepdim=True)[0]], dim=1)
        box = box.cpu().numpy()
        py = output['py'][-1].detach().cpu().numpy()

        if len(py) == 0:
            return

        img_id = int(batch['meta']['img_id'][0])
        center = batch['meta']['center'][0].detach().cpu().numpy()
        scale = batch['meta']['scale'][0].detach().cpu().numpy()

        h, w = batch['inp'].size(2), batch['inp'].size(3)
        trans_output_inv = utils.get_affine_transform(center, scale, 0, [w, h], inv=1)
        img = self.coco.loadImgs(img_id)[0]
        ori_h, ori_w = img['height'], img['width']
        py = [utils.affine_transform(py_, trans_output_inv) for py_ in py]
        rles = utils.coco_poly_to_rle(py, ori_h, ori_w)
        coco_dets = []
        for i in range(len(score)):
            if score[i]>thr:
                box_ = utils.affine_transform(box[i].reshape(-1, 2), trans_output_inv).ravel()
                box_[2] -= box_[0]
                box_[3] -= box_[1]
                box_ = list(map(lambda x: float('{:.2f}'.format(x)), box_))
                detection = {
                    'image_id': img_id,
                    'bbox': box_,
                    'category_id': self.contiguous_category_id_to_json_id[label[i]],
                    'segmentation': rles[i],
                    'seg_coord': py[i].tolist(),
                    'score': float('{:.2f}'.format(score[i]))
                }
                coco_dets.append(detection)

        self.results.extend(coco_dets)
        self.img_ids.append(img_id)

    def save(self):
        json.dump(self.results, open(os.path.join(self.result_dir, 'results.json'), 'w'))
        return 0