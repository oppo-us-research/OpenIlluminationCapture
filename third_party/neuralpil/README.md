# Neural-PIL on OpenIllumination Dataset

Note: Most code are reused from the official Neural-PIL repository, please check their documentation for more information. [Neural-PIL Github Page](https://github.com/cgtuebingen/Neural-PIL/tree/main)

## Environment Installation
```
conda env create -f environment.yml
conda activate neuralpil
```

## Training under single illumination
```
python train_neural_pil.py --datadir [DIR_TO_DATASET_FOLDER] --basedir [TRAIN_DIR] --expname [EXPERIMENT_NAME] --gpu [COMMA_SEPARATED_GPU_LIST]
```
An example usage:
```
python train_neural_pil.py \
--config configs/neural_pil/openillum_data_single.txt \
--datadir lightstage_dataset/20230524-13_08_01_obj_2_egg \
--basedir output/ \
--expname egg_single_output \
--gpu 0
```

## Training under multi illumination
```
python train_neural_pil.py --datadir [DIR_TO_DATASET_FOLDER] --basedir [TRAIN_DIR] --expname [EXPERIMENT_NAME] --gpu [COMMA_SEPARATED_GPU_LIST]
```

An example usage:
```
python train_neural_pil.py \
--config configs/neural_pil/openillum_data_multi.txt \
--datadir lightstage_dataset/20230524-13_08_01_obj_2_egg \
--basedir output/ \
--expname egg_multi_output \
--gpu 0
```


## Evaluation

```
python train_neural_pil.py --render_only --config /[TRAIN_DIR]/[EXPERIMENT_NAME]/args.txt
```
An example usage:

```
python train_neural_pil.py --render_only --config /data_fast/neuralpil_output/egg_multi_output/args.txt
```


