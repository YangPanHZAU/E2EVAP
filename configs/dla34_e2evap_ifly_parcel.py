import numpy as np

class commen(object):
    task = 'e2ec'
    points_per_poly = 128
    down_ratio = 4
    result_dir = './ckpt/dla34_e2evap_ifly_parcel_ifly_parcel_bs8/coco_parcel_ifly_py/reuslt'
    record_dir = './ckpt/dla34_e2evap_ifly_parcel_ifly_parcel_bs8/coco_parcel_ifly_py/'
    model_dir = './ckpt/dla34_e2evap_ifly_parcel_ifly_parcel_bs8/coco_parcel_ifly_py/ckpt_model'


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
    input_w, input_h = (512, 512)
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

class train(object):
    save_ep = 1
    eval_ep = 1
    optimizer = {'name': 'adam', 'lr': 1e-4,
                 'weight_decay': 5e-4,
                 'milestones': [80, 120, ],
                 'gamma': 0.5}
    batch_size = 8
    num_workers = 4
    epoch = 150
    with_dml = True
    start_epoch = 10
    weight_dict = {'init': 0.1, 'coarse': 0.1, 'evolve': 1.,'init_topo_loss':1.,'coarse_topo_loss':1.,'edge_evolve':1.}
    dataset = 'coco_parcel_iFly_Train'

class test(object):
    test_stage = 'final-dml'  # init, coarse final or final-dml
    test_rescale = None
    ct_score = 0.05
    with_nms = True
    with_post_process = False
    segm_or_bbox = 'segm'
    dataset = 'coco_parcel_iFly_Val'

class config(object):
    commen = commen
    data = data
    model = model
    train = train
    test = test