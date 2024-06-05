
class DatasetInfo(object):

    data_dir_iFLy = '/home/py21/changguang_parcel/top1_baseline/out_shp/train/512-512/'
    dataset_info = {
        'coco_train': {
            'name': 'coco',
            'image_dir': 'data/coco/train2017',
            'anno_dir': 'data/coco/annotations/instances_train2017.json',
            'split': 'train'
        },
        'coco_val': {
            'name': 'coco',
            'image_dir': 'data/coco/val2017',
            'anno_dir': 'data/coco/annotations/instances_val2017.json',
            'split': 'val'
        },
        'coco_test': {
            'name': 'coco',
            'image_dir': 'data/coco/test2017',
            'anno_dir': 'data/coco/annotations/image_info_test-dev2017.json',
            'split': 'test'
        },
        'sbd_train': {
            'name': 'sbd',
            'image_dir': 'data/sbd/img',
            'anno_dir': 'data/sbd/annotations/sbd_train_instance.json',
            'split': 'train'
        },
        'sbd_val': {
            'name': 'sbd',
            'image_dir': 'data/sbd/img',
            'anno_dir': 'data/sbd/annotations/sbd_trainval_instance.json',
            'split': 'val'
        },
        'kitti_train': {
            'name': 'kitti',
            'image_dir': 'data/kitti/training/image_2', 
            'anno_dir': 'data/kitti/training/instances_train.json', 
            'split': 'train'
        }, 
        'kitti_val': {
            'name': 'kitti',
            'image_dir': 'data/kitti/testing/image_2', 
            'anno_dir': 'data/kitti/testing/instances_val.json', 
            'split': 'val'
        },
        'cityscapes_train': {
            'name': 'cityscapes',
            'image_dir': 'data/cityscapes/leftImg8bit',
            'anno_dir': ('data/cityscapes/annotations/train', 'data/cityscapes/annotations/train_val'),
            'split': 'train'
        },
        'cityscapes_val': {
            'name': 'cityscapes',
            'image_dir': 'data/cityscapes/leftImg8bit',
            'anno_dir': 'data/cityscapes/annotations/val',
            'split': 'val'
        },
        'cityscapesCoco_train': {
            'name': 'cityscapesCoco',
            'image_dir': 'data/cityscapes/leftImg8bit',
            'anno_dir': ('data/cityscapes/annotations/train', 'data/cityscapes/annotations/train_val'),
            'split': 'train'
        },
        'cityscapesCoco_val': {
            'name': 'cityscapesCoco',
            'image_dir': 'data/cityscapes/leftImg8bit/val',
            'anno_dir': 'data/cityscapes/coco_ann/instance_val.json',
            'split': 'val'
        },
        'cityscapesCoco_gen': {
            'name': 'cityscapesCoco',
            'image_dir': 'data/SYNTHIA/images',
            'anno_dir': 'data/SYNTHIA/annotations/instancesonly_train.json',
            'split': 'val'
        },
        'cityscapes_test': {
            'name': 'cityscapes',
            'image_dir': 'data/cityscapes/leftImg8bit/test', 
            'anno_dir': 'data/cityscapes/annotations/test', 
            'split': 'test'
        },
        'coco_parcel_iFly_Train': {
            'name': 'coco_parcel',
            'image_dir': data_dir_iFLy + 'train',
            'anno_dir': data_dir_iFLy + 'annotations/train.json',
            'split': 'train'
        },
        'coco_parcel_iFly_Val': {
            'name': 'coco_parcel',
            'image_dir': data_dir_iFLy + 'test',
            'anno_dir': data_dir_iFLy + 'annotations/test.json',
            'split': 'val'
        },
        'coco_parcel_iFly_Test': {
            'name': 'coco_parcel',
            'image_dir': '/home/py21/changguang_parcel/top1_baseline/out_shp/chusai_test/512-512/images',
            'anno_dir': '/home/py21/changguang_parcel/top1_baseline/out_shp/chusai_test/512-512/annotations/chusai_test.json',
            'split': 'test'
        },
        'coco_parcel_iFly_Test_overlap': {
            'name': 'coco_parcel',
            'image_dir': '/home/py21/E2EC_parcel/test_overlap_infer_region',
            'anno_dir': '/home/py21/E2EC_parcel/test_overlap_infer_region/test_region_coco.json',
            'split': 'test'
        },
        'coco_parcel_iFly_Test_overlap_CGDZ_8_offset': {
            'name': 'coco_parcel',
            'image_dir': '/home/py21/E2EC_parcel/test_overlap_infer_region/CGDZ_8_offset',
            'anno_dir': '/home/py21/E2EC_parcel/test_overlap_infer_region/CGDZ_8_offset/test_region_coco.json',
            'split': 'test'
        },
        'coco_parcel_iFly_Test_overlap_CGMY_8_offset': {
            'name': 'coco_parcel',
            'image_dir': '/home/py21/E2EC_parcel/test_overlap_infer_region1/CGMY_8_offset',
            'anno_dir': '/home/py21/E2EC_parcel/test_overlap_infer_region1/CGMY_8_offset/test_region_coco.json',
            'split': 'test'
        },
        'coco_parcel_iFly_Test_overlap_CGMY_8_offset_512': {
            'name': 'coco_parcel',
            'image_dir': '/home/py21/E2EC_parcel/test_overlap_infer_region1_512_384/CGMY_8_offset',
            'anno_dir': '/home/py21/E2EC_parcel/test_overlap_infer_region1_512_384/CGMY_8_offset/test_region_coco.json',
            'split': 'test'
        },
    }
