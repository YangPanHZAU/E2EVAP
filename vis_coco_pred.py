from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import matplotlib
from skimage import io
import os
from tqdm import tqdm
matplotlib.use('Agg')
def make_label_show(img,example_coco,annIds,result_label_show_dir,img_name):
    anns = example_coco.loadAnns(annIds)
    plt.figure(figsize=(8, 8), dpi=100)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.imshow(img);
    plt.axis('off')
    example_coco.showAnns(anns)
    plt.show()
    result_path = result_label_show_dir + '/{}.tif'.format(img_name)
    plt.savefig(result_path, pad_inches=0)
def make_infer_label_show(img,coco_res,annIds,threshold_score,result_infer_label_show_dir,img_name):
    plt.figure(figsize=(8, 8), dpi=100)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.axis('off')
    plt.imshow(img);
    anns_infer = coco_res.loadAnns(annIds)
    select_anns = [ann for ann in anns_infer if ann['score'] > threshold_score]
    coco_res.showAnns(select_anns)
    plt.show()
    result_path = result_infer_label_show_dir + '/{}.jpg'.format(img_name[:-4])
    plt.savefig(result_path, pad_inches=0)
    plt.close()
if __name__ == '__main__':
    predict_json_file  = '/home/py21/e2ec/ckpt/dla34_e2ec_bs8/coco_parcel_ifly_py/infer_epoch_reuslt/results.json'#prediction json file
    test_json_file = r'/home/py21/changguang_parcel/top1_baseline/out_shp/chusai_test/512-512/annotations/chusai_test.json'#gt json file
    imgsPath = r'/home/py21/changguang_parcel/top1_baseline/out_shp/chusai_test/512-512/images'#test image
    result_infer_label_show_dir = r'/home/py21/VPNet/test_vis/iFly/e2ec_dla34/'#save path
    if not os.path.exists(result_infer_label_show_dir):
        os.makedirs(result_infer_label_show_dir)
    threshold_score=0.3
    coco = COCO(test_json_file)
    coco_res = coco.loadRes(predict_json_file)
    image_ids = coco.getImgIds()
    for image_id in tqdm(image_ids[:100]):
        image_dict = coco.loadImgs(image_id)[0]
        img_name=image_dict['file_name']
        image = io.imread(imgsPath + r'/' + img_name)[:, :, 0:3]
        label_annIds = coco.getAnnIds(imgIds=image_id, iscrowd=None)
        #make_label_show(img=image,example_coco=coco,annIds=label_annIds,result_label_show_dir=result_label_show_dir,img_name=img_name)
        infer_annIds = coco_res.getAnnIds(imgIds=image_id, iscrowd=None)
        make_infer_label_show(img=image, coco_res=coco_res, annIds=infer_annIds,threshold_score=threshold_score,
                              result_infer_label_show_dir=result_infer_label_show_dir,img_name=img_name)


