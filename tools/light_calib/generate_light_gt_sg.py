import os
import numpy as np
import json

LIGHT_PATH  = '/home/isabella/Lab/novel_relighting/light_pos.npy'
LIGHT_OLAT_PATH = '/home/isabella/Lab/novel_relighting/capture_brightness_LGT2.txt'
LIGHT_RANDOM_PATH = '/home/isabella/Lab/novel_relighting/capture_brightness_LGT1.txt'
CAMERA_PATH = '/home/isabella/Lab/novel_relighting/cameras.json'

    
def convert_light_idx():
    light_idx_txt = np.loadtxt(LIGHT_OLAT_PATH)[..., :142]  # [_, OLAT_NUM (142+1)]
    
    light_pos = np.load(LIGHT_PATH)  # [142, 3]
    light_pos_index_convert = np.zeros_like(light_pos)
    
    for frame in range(light_idx_txt.shape[-1]):
        light_idx_list = light_idx_txt[..., frame] > 0
        light_idx = np.where(light_idx_list == True)[0][0]
        light_pos_index_convert[light_idx] = light_pos[frame]
        
        # if frame in list(range(23)):
        #     print(frame)
        

    
    return light_pos_index_convert
        
      
def main():
    # Load cameras 
    with open(CAMERA_PATH) as f:
        cameras = json.load(f)
    for cam_idx in cameras:
        cam = cameras[cam_idx]
        cam_pose = np.array(cam['pose'])
        
    light_pos_index_convert = convert_light_idx()
    
    random_light_list = np.loadtxt(LIGHT_RANDOM_PATH)  # [_, 13]
    
    selected_light_pattern = 10
    
    print()
    

if __name__ == '__main__':
    main()
    