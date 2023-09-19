# TensoIR on OpenIllumination Dataset

Note: Most code are reused from the official TensoIR repository, please check their documentation for more information. [TensoIR Github Page](https://github.com/Haian-Jin/TensoIR)

## Environment Installation
```
conda create -n TensoIR python=3.8
conda activate TensoIR
pip install torch==1.10 torchvision
pip install tqdm scikit-image opencv-python configargparse lpips imageio-ffmpeg kornia lpips tensorboard loguru plyfile
```

## Training under single illumination
```
export PYTHONPATH=.
python train_tensoIR_simple_fix_SG.py --config configs/singleillum.txt  --expname $EXP_NAME$ --datadir $PATH_TO_DATASET_OUTPUT_FOLDER$
```
An example usage:
```
python train_tensoIR_simple_fix_SG.py --config configs/singleillum.txt  --expname 20230524-11_59_05_obj_1_car --datadir lightstage_dataset/20230524-11_59_05_obj_1_car/output
```

## Training under multi illumination
```
export PYTHONPATH=.
python train_tensoIR_simple_fix_SG.py --config configs/multiillum.txt  --expname $EXP_NAME$ --datadir $PATH_TO_DATASET_OUTPUT_FOLDER$
```

## Training under fixed lighting for relighting evaluation
```
export PYTHONPATH=.
python train_tensoIR_simple_fix_SG.py --config configs/fixed_sg.txt  --expname $EXP_NAME$ --datadir $PATH_TO_DATASET_OUTPUT_FOLDER$
```


## Evaluation scripts
### Single illumination
```
export PYTHONPATH=.
python train_tensoIR_simple_fix_SG.py --config configs/singleillum.txt  --expname $EXP_NAME$ --datadir $PATH_TO_DATASET_OUTPUT_FOLDER$ --render_only 1 --render_test 1 --export_mesh 1
```

### Multi-illumination
```
export PYTHONPATH=.
python train_tensoIR_simple_fix_SG.py --config configs/multiillum.txt  --expname $EXP_NAME$ --datadir $PATH_TO_DATASET_OUTPUT_FOLDER$ --render_only 1 --render_test 1 --export_mesh 1
```

### Relighting evaluation under unseen lights
```
python relight_sg_final.py --config configs/fixed_sg.txt --expname $EXP_NAME$ --datadir $PATH_TO_DATASET_OUTPUT_FOLDER$
```
