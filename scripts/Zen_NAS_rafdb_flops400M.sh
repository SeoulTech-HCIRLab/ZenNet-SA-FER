#!/bin/bash
cd "$(dirname "$0")"
set -e

cd ../

resolution=224
budget_flops=400e6
max_layers=10
population_size=512
evolution_max_iter=480000

save_dir=dafdb_log/Zen_NAS_rafdb_params2M
mkdir -p ${save_dir}

echo "SuperConvK3BNRELU(3,8,1,1)SuperResK3K3(8,16,1,8,1)SuperResK3K3(16,32,2,16,1)SuperResK3K3(32,64,2,32,1)SuperResK3K3(64,64,2,32,1)SuperConvK1BNRELU(64,128,1,1)" \
> ${save_dir}/init_plainnet.txt

#python evolution_search.py --gpu 2 \
 # --zero_shot_score Zen \
 # --search_space SearchSpace/search_space_XXBL.py \
 # --budget_flops ${budget_flops} \
 # --max_layers ${max_layers} \
 # --batch_size 16 \
 # --input_image_size 224 \
 # --plainnet_struct_txt ${save_dir}/init_plainnet_test.txt \
 # --num_classes 10 \
 # --evolution_max_iter ${evolution_max_iter} \
 # --population_size ${population_size} \
 # --save_dir ${save_dir}


python analyze_model.py \
  --input_image_size 224 \
  --num_classes 10 \
  --arch Masternet.py:MasterNet \
  --plainnet_struct_txt ${save_dir}/best_structure.txt

python train_image_classification.py --dataset rafdb --gpu 2 --num_classes 7 \
  --dist_mode single --workers_per_gpu 6 \
  --input_image_size 224 --epochs 1000 --warmup 5 \
  --optimizer sgd --bn_momentum 0.01 --wd 5e-4 --nesterov --weight_init custom \
  --label_smoothing --random_erase --mixup --auto_augment \
  --lr_per_256 0.1 --target_lr_per_256 0.0 --lr_mode cosine \
  --arch Masternet.py:MasterNet \
  --plainnet_struct_txt ${save_dir}/best_structure.txt \
  --batch_size_per_gpu 32 \
  --save_dir ${save_dir}/rafdb_1000epochs

