import os
import glob
import os.path as osp
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir",default="",type=str)
parser.add_argument("--output_dir",default="",type=str)
parser.add_argument("--calibration_dir",default="",type=str)

args=parser.parse_args()

def main():
    input_root_dir = osp.expanduser(args.input_dir)
    output_root_dir = osp.expanduser(args.output_dir)
    input_dirs = sorted(glob.glob(osp.join(input_root_dir, '*_obj*')))
    for input_dir in input_dirs:
        output_dir = osp.join(output_root_dir, input_dir.split("/")[-1])
        os.makedirs(output_dir, exist_ok=True)
        os.system(f"cp -r {args.calibration_dir} {output_dir}")
        for i in range(1, 14):
            light_i_dir = glob.glob(osp.join(input_dir, f"*{i:03d}*"))[0]
            os.makedirs(osp.join(output_dir, f"Lights/{i:03d}/raw"), exist_ok=True)
            os.system(f"cp {osp.join(light_i_dir, '*')} {osp.join(output_dir, f'Lights/{i:03d}/raw')}")
        os.makedirs(osp.join(output_dir, "segmentation/images"))
        os.system(f"cp {osp.join(light_i_dir, '*')} {osp.join(output_dir, 'segmentation/images')}")


if __name__ == '__main__':
    main()
