#!/bin/bash
cd "$(dirname "$0")"
set -e

cd ../

resolution=224
budget_model_size=1e6
budget_flops=400e6
max_layers=14
population_size=512
evolution_max_iter=480000

save_dir=dafdb_log/Zen_NAS_rafdb_flops400_idw
mkdir -p ${save_dir}

echo "SuperConvK3BNRELU(3,8,2,1)SuperResIDWE6K3(8,32,2,8,1)SuperResIDWE6K3(32,48,2,32,1)\
SuperResIDWE6K3(48,96,2,48,1)SuperResIDWE6K3(96,128,2,96,1)\
SuperConvK1BNRELU(128,2048,1,1)" > ${save_dir}/init_plainnet.txt

python evolution_search.py --gpu 2 \
  --zero_shot_score Zen \
  --search_space SearchSpace/search_space_IDW_fixfc.py \
  --budget_flops ${budget_flops} \
  --budget_model_size ${budget_model_size}\
  --max_layers ${max_layers} \
  --batch_size 16 \
  --input_image_size 224 \
  --plainnet_struct_txt ${save_dir}/init_plainnet.txt \
  --num_classes 7 \
  --evolution_max_iter ${evolution_max_iter} \
  --population_size ${population_size} \
  --save_dir ${save_dir}


python analyze_model.py \
  --input_image_size 224 \
  --num_classes 7 \
  --arch Masternet.py:MasterNet \
  --plainnet_struct_txt ${save_dir}/best_structure.txt

python train_image_classification.py --dataset rafdb --gpu 2 --num_classes 7 \
  --dist_mode single --workers_per_gpu 6 \
  --input_image_size 224 --epochs 1440 --warmup 5 \
  --optimizer sgd --bn_momentum 0.01 --wd 5e-4 --nesterov --weight_init custom \
  --label_smoothing --random_erase --mixup --auto_augment \
  --lr_per_256 0.1 --target_lr_per_256 0.0 --lr_mode cosine \
  --arch Masternet.py:MasterNet \
  --plainnet_struct_txt ${save_dir}/best_structure.txt \
  --batch_size_per_gpu 32 \
  --save_dir ${save_dir}/rafdb_1440epochs

