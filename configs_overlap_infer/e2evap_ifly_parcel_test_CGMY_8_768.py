import numpy as np

class commen(object):
    task = 'e2ec'
    points_per_poly = 128
    down_ratio = 4
    result_dir = 'test_overlap_infer_region1/CGMY_8_offset_result/result_coco'


class data(object):
    mean=np.array([123.675, 116.28, 103.53],dtype=np.float32).reshape(1, 1, 3)
    std=np.array([58.395, 57.12, 57.375],dtype=np.float32).reshape(1, 1, 3)
    data_rng = np.random.RandomState(123)
    eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                       dtype=np.float32)
    eig_vec = np.array([
        [-0.58752847, -0.69563484, 0.41340352],
        [-0.5832747, 0.00994535, -0.81221408],
        [-0.56089297, 0.71832671, 0.41158938]
    ], dtype=np.float32)
    down_ratio = commen.down_ratio
    scale = None
    input_w, input_h = (768, 768)
    test_scale = None
    scale_range = None
    points_per_poly = commen.points_per_poly
    train_ms_mode=False

class model(object):
    #dla_layer = 34
    encoder = 'dla34'#'dla34','resnet50'
    in_channel =3
    head_conv = 256
    use_dcn = True
    points_per_poly = commen.points_per_poly
    down_ratio = 4.0
    init_stride = 10.
    coarse_stride = 4.
    evolve_stride = 1.
    evolve_down_ratio = 4.
    backbone_num_layers = 50
    heads = {'ct_hm': 1, 'wh': commen.points_per_poly * 2}
    evolve_iters = 3
    decode_multi_attention_mode = False #True,False
    evolve_poly_multi_attention = False

class test(object):
    test_stage = 'final-dml'  # init, coarse final or final-dml
    test_rescale = None
    ct_score = 0.05
    with_nms = True
    with_post_process = False
    segm_or_bbox = 'segm'
    dataset = 'coco_parcel_iFly_Test_overlap_CGMY_8_offset'

class config(object):
    commen = commen
    data = data
    model = model
    test = test