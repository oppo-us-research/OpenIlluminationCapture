from yacs.config import CfgNode as CN

_C = CN()

_C.dbg = False
_C.output_dir = ""

_C.cylinder = CN()
_C.cylinder.rgb_path = ""
_C.cylinder.mask_path = ""
_C.cylinder.K_path = ""
_C.cylinder.steps = [10000]
_C.cylinder.lrs = [1e-3]
_C.cylinder.init_Toc = []
_C.cylinder.log_interval = 100
