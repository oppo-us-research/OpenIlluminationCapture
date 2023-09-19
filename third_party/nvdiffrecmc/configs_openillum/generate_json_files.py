import os, json
import os.path as osp
import copy

template = '/home/isabella/Lab/OpenIllumination/nvdiffrecmc/configs_openillum/template.json'
config_folder = '/home/isabella/Lab/OpenIllumination/nvdiffrecmc/configs_openillum'

obj_names_path = '/home/isabella/Lab/OpenIllumination/nvdiffrecmc/configs_openillum/obj_names_small.txt'
obj_name_list = []
with open(obj_names_path, 'r') as f:
    for line in f:
        obj_name_list.append(line.strip())
        
with open(template, 'r') as f:
    cfg = json.load(f)
    
for obj_name in obj_name_list:
    out_path = osp.join(config_folder, obj_name + '.json')
    cfg_obj = copy.deepcopy(cfg)
    
    dataset_dir = osp.join('/home/isabella/Lab/OpenIllumination/data/OpenIllumination', obj_name)
    out_dir = obj_name
    
    cfg_obj['ref_mesh'] = dataset_dir
    cfg_obj['out_dir'] = out_dir
    
    with open(out_path, 'w') as f:
        json.dump(cfg_obj, f)
    