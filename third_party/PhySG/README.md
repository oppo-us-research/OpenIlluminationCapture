# PhySG on OpenIllumination Dataset

Note: Most code are reused from the official Neural-PIL repository, please check their documentation for more information. [PhySG Github Page](https://github.com/Kai-46/PhySG)

## Environment Installation
```
conda env create -f environment.yml
conda activate PhySG
```

## Before Training

The preprocessed data structure used by PhySG is a bit different from ours, it needs a scene normalization.

We write a simple script to convert the our data. Check convert_dataset.py for details.

## Training 

An example usage:
```
python training/exp_runner.py \
--conf confs_sg/default.conf   \
--data_split_dir lightstage_dataset/20230524-17_27_05_obj_26_pumpkin   \
--expname pumpkin  \
--nepoch 8000  \ 
--max_niter 400001 \ 
--gamma 2.2 \
--illumination_idx 013 \
--output_img_size 800 1200 
```


## Evaluation

An example usage:

```
python evaluation/eval.py  \
--conf confs_sg/default.conf \  
--data_split_dir lightstage_dataset/20230524-17_27_05_obj_26_pumpkin \
--expname pumpkin  \
--gamma 2.2 \
--resolution 512 \
--illumination_idx 013 \
--output_img_size 800 1200 
```


