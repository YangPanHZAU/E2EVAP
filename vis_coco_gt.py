from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import matplotlib
from skimage import io
from tqdm import tqdm
matplotlib.use('Agg')
def make_label_show(img,example_coco,annIds,result_label_show_dir,img_name):
    anns = example_coco.loadAnns(annIds)
    plt.figure(figsize=(8, 8), dpi=100)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.axis('off')
    plt.imshow(img)
    example_coco.showAnns(anns)
    plt.show()
    result_path = result_label_show_dir + '/{}.jpg'.format(img_name[:-4])
    plt.savefig(result_path, pad_inches=0)
    plt.close()
def make_infer_label_show(img,coco_res,annIds,threshold_score,result_infer_label_show_dir,img_name):
    plt.figure(figsize=(8, 8), dpi=100)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.axis('off')
    plt.imshow(img)
    anns_infer = coco_res.loadAnns(annIds)
    select_anns = [ann for ann in anns_infer if ann['score'] > threshold_score]
    coco_res.showAnns(select_anns)
    plt.show()
    result_path = result_infer_label_show_dir + '/{}.jpg'.format(img_name[:-4])
    plt.savefig(result_path, pad_inches=0)
    plt.close()
if __name__ == '__main__':
    test_json_file = r'/home/py21/changguang_parcel/top1_baseline/out_shp/chusai_test/512-512/annotations/chusai_test.json'
    imgsPath = r'/home/py21/changguang_parcel/top1_baseline/out_shp/chusai_test/512-512/images'
    result_label_show_dir = r'/home/py21/VPNet/test_vis/iFly/test_gt/'
    threshold_score=0.3
    coco = COCO(test_json_file)
    image_ids = coco.getImgIds()
    for image_id in tqdm(image_ids[:100]):
        image_dict = coco.loadImgs(image_id)[0]
        img_name=image_dict['file_name']
        image = io.imread(imgsPath + r'/' + img_name)[:, :, 0:3]
        label_annIds = coco.getAnnIds(imgIds=image_id, iscrowd=None)
        make_label_show(img=image,example_coco=coco,annIds=label_annIds,result_label_show_dir=result_label_show_dir,img_name=img_name)



