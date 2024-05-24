save_dir=teacher_training
python ts_train_image_classification.py --dataset rafdb --num_classes 7 \
  --dist_mode single --workers_per_gpu 6 \
  --input_image_size 224 --epochs 1440 --warmup 5 --auto_resume \
  --optimizer sgd --bn_momentum 0.01 --wd 4e-5 --nesterov --weight_init custom --shuffle_train_data \
  --label_smoothing --random_erase --mixup --auto_augment --auto_augment_type CIFA \
  --lr_per_256 0.1 --target_lr_per_256 0.0 --lr_mode cosine \
  --arch Masternet.py:MasterNet \
  --plainnet_struct_txt ZenNet/zen_Mobile_flops400M_rafdb.txt\
  --teacher_arch geffnet_efficientnet_b3 \
  --teacher_load_parameters_from rafdb_log/ts_b3/best-params_rank0.pth \
  --teacher_input_image_size 224 \
  --teacher_feature_weight 0.1 \
  --teacher_logit_weight 0.1 \
  --ts_proj_no_relu \
  --ts_proj_no_bn \
  --use_se \
  --target_downsample_ratio 16 \
  --batch_size_per_gpu 128 --save_dir ${save_dir}/rafdb_ts480_flops400_mobile