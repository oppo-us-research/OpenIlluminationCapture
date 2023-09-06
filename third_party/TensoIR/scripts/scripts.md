<!--
 * @Author: Haian-Jin 3190106083@zju.edu.cn
 * @Date: 2022-08-03 20:38:18
 * @LastEditors: Haian-Jin 3190106083@zju.edu.cn
 * @LastEditTime: 2022-08-10 23:51:27
 * @FilePath: /TensoRFactor/scripts/scripts.md
 * @Description:
-->
* export mesh from tensorf / tensorfactor
    python scripts/export_mesh.py --ckpt ${ckpt_path}  --model_name TensorVMSplit
* run visualize.py
    export PYTHONPATH=.
    python scripts/visualize.py --ckpt "/home/haian/logs_info/tensorfactor/log_tensorfactor/hotdog_ds_1-20220806-222421/hotdog_ds_1_930000.th"  --model_name TensorVMSplit --config configs/hotdog_tensorfactor_jha.txt --ckpt_visibility "/home/haian/logs_info/tensorfactor/log_visibility/hotdog_ds_1-20220809-162255/hotdog_ds_1.th"  --batch_size 4096 --geo_buffer_path ./temp_result