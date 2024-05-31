#!/bin/bash
cd "$(dirname "$0")"
set -e

cd ../

resolution=224
budget_flops=400e6
max_layers=10
population_size=512
evolution_max_iter=480000

save_dir=SearchLog/ZenNet-FER-R
mkdir -p ${save_dir}

echo "SuperConvK3BNRELU(3,8,1,1)SuperResK3K3(8,16,1,8,1)SuperResK3K3(16,32,2,16,1)SuperResK3K3(32,64,2,32,1)SuperResK3K3(64,64,2,32,1)SuperConvK1BNRELU(64,128,1,1)" \
> ${save_dir}/init_plainnet.txt

python evolution_search.py --gpu 2 \
  --zero_shot_score Zen \
  --search_space SearchSpace/search_space_XXBL.py \
  --budget_flops ${budget_flops} \
  --max_layers ${max_layers} \
  --batch_size 16 \
  --input_image_size 224 \
  --plainnet_struct_txt ${save_dir}/init_plainnet_test.txt \
  --num_classes 10 \
  --evolution_max_iter ${evolution_max_iter} \
  --population_size ${population_size} \
  --save_dir ${save_dir}


python analyze_model.py \
  --input_image_size 224 \
  --num_classes 10 \
  --arch Masternet.py:MasterNet \
  --plainnet_struct_txt ${save_dir}/best_structure.txt

