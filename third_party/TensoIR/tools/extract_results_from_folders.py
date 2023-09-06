import os
import os.path as osp
import numpy as np
import glob 
import json

def main():
    illum_res_folder = '/isabella-slow/NeurIPS2023/Logs-supp/MultiIllum/'
    relighting_res_folder = '/isabella-slow/NeurIPS2023/Logs-supp/NovelRelighting_new/'
    
    RELIGHTING = True
    res_folder = relighting_res_folder if RELIGHTING else illum_res_folder
    folder_list = sorted(glob.glob(osp.join(res_folder, 'NovelRelighting*'))) if RELIGHTING else sorted(glob.glob(osp.join(res_folder, '*obj*')))
    
    psnrs = []
    psnr_json = {}
    for cur_idx, folder in enumerate(folder_list):
        if len(folder.split('obj_')) > 1:
            obj_idx = int(folder.split('obj_')[-1].split('_')[0])
        elif len(folder.split('obj-')) > 1:
            obj_idx = int(folder.split('obj-')[-1].split('-')[0])
        psnr = 0
        if RELIGHTING is False:
            metics_file = osp.join(folder, 'imgs_test_all', 'metrics_record.txt')
            if osp.exists(metics_file):
                with open(metics_file, 'r') as f:
                    for line in f:
                        key = 'PSNR_nvs_brdf: '
                        if key in line:
                            psnr = float(line.split(key)[-1])
                            break
            psnrs.append(psnr)
            psnr_json[obj_idx] = psnr
        else:
            metrics_file = osp.join(folder, 'psnr_relighting.json')
            if osp.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    relight_json = json.load(f)
                relight_json.pop('Avg')
                relight_json.pop('009')
                relight_json.pop('011')
                relight_json.pop('013')
                all_valid_psnr = [float(relight_json[key]) for key in relight_json]
                psnr = np.mean(all_valid_psnr)
            else:
                psnr = 0
            psnrs.append(psnr)
            psnr_json[obj_idx] = psnr
                
                
            
    np.savetxt(osp.join(res_folder, 'psnrs.txt'), psnrs, fmt='%.2f')
    json.dump(psnr_json, open(osp.join(res_folder, 'psnrs.json'), 'w'))
    
    psnr_final = []
    for idx in range(63):
        psnr = psnr_json[idx+1]
        psnr_final.append(psnr)
    np.savetxt(osp.join(res_folder, 'psnrs_final.txt'), psnr_final, fmt='%.2f')
    
    

if __name__ == '__main__':
    main()