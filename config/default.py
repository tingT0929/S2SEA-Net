import os
from yacs.config import CfgNode as CN
import time


_C = CN()
_C.seed = 42
_C.network = 'CoviSeq'
_C.path = '/public/home/yuqi/lawliet/new/'
_C.log_path = _C.path + 'logs/'

# ================= dataset ====================
_C.data = CN()
_C.data.root = _C.path + 'data/'
_C.data.enc_steps = 30
_C.data.pred_steps = 7
_C.data.use_month = False
_C.data.agg_travel = True
_C.data.state_num = 51
_C.data.county_num = 3118

# ================= training ====================
_C.train = CN()
_C.train.batch_size = 128
_C.train.lr = 1e-3
_C.train.epochs = 200
_C.train.weight_path = _C.log_path + 'weights/'

# ================= optimizer ====================
_C.optim = CN()
_C.optim.optimizer = 'Adam'
_C.optim.weight_decay = 1e-5
_C.optim.momentum = 0.9

# ================= models ====================
_C.model = CN()
_C.model.latent_dim = 256
_C.model.embed_dim = 64
_C.model.hidden_size = 64
_C.model.dropout = 0.2
_C.model.travel_num = 4 + 6 * (not _C.data.agg_travel)
_C.model.encoder_dim = 2 + _C.data.use_month + _C.model.travel_num
_C.model.decoder_dim = 2
_C.model.temporature = 1


def init_cfg(args):
    cfg = _C.clone()
    if args.cfg:
        cfg.merge_from_file(args.cfg)
    if args.opts:
        cfg.merge_from_list(args.opts)

    if args.seed:
        cfg.seed = args.seed

    cfg.save_path = cfg.log_path + f"{cfg.network}/{cfg.model.encoder_dim}_{cfg.data.pred_steps}/"
    cfg.result_path = cfg.log_path + 'results/'
    time_str = time.strftime('%m-%d-%H-%M')
    log_file = '{}.log'.format(time_str)
    weight_file = f'{cfg.network}{cfg.data.pred_steps}_{time_str}.h5'
    cfg.save_log_path = cfg.save_path + 'log/' + log_file
    cfg.weight_file = cfg.train.weight_path + weight_file
    os.makedirs(cfg.log_path, exist_ok=True)
    os.makedirs(cfg.train.weight_path, exist_ok=True)
    os.makedirs(cfg.save_path, exist_ok=True)
    os.makedirs(cfg.result_path, exist_ok=True)
    os.makedirs(cfg.save_path + 'log/', exist_ok=True)

    return cfg



