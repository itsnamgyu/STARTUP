#!/bin/bash

# bash script to evaluate different representations. 
# finetune.py learns a linear classifier on the features extracted from the support set 
# compile_result.py computes the averages and the 96 confidence intervals from the results generated from finetune.py
# and evaluate on the query set
export CUDA_VISIBLE_DEVICES=0

##############################################################################################
# Evaluate Representations trained on miniImageNet
##############################################################################################

# Before running the commands, please take care of the TODO appropriately
for source in "miniImageNet" "tieredImageNet" "ImageNet"
source="miniImageNet"
for target in "cars" "cub" "places" "plantae"
do
    python finetune.py --image_size 224 --n_way 5 --n_shot 1 5 --n_episode 600 --n_query 15 --seed 1 --freeze_backbone --save_dir results/STARTUP_${source} --source_dataset $source --target_dataset $target --subset_split datasets/split_seed_1/$target\_labeled_80.csv --model resnet10 --embedding_load_path ../student_STARTUP/${source}_source/$target\_unlabeled_20/checkpoint_best.pkl --embedding_load_path_version 1
    python compile_result.py --result_file results/STARTUP_${source}/$source\_$target\_5way.csv
done


source="tieredImageNet"
for target in "ChestX" "ISIC" "EuroSAT" "CropDisease" "cars" "cub" "places" "plantae"
do
    python finetune.py --image_size 224 --n_way 5 --n_shot 1 5 --n_episode 600 --n_query 15 --seed 1 --freeze_backbone --save_dir results/STARTUP_${source} --source_dataset $source --target_dataset $target --subset_split datasets/split_seed_1/$target\_labeled_80.csv --model resnet18 --embedding_load_path ../student_STARTUP/${source}_source/$target\_unlabeled_20/checkpoint_best.pkl --embedding_load_path_version 1
    python compile_result.py --result_file results/STARTUP_${source}/$source\_$target\_5way.csv
done


source="ImageNet"
for target in "cars" "cub" "places" "plantae"
do
    python finetune.py --image_size 224 --n_way 5 --n_shot 1 5 --n_episode 600 --n_query 15 --seed 1 --freeze_backbone --save_dir results/STARTUP_${source} --source_dataset $source --target_dataset $target --subset_split datasets/split_seed_1/$target\_labeled_80.csv --model resnet18 --embedding_load_path ../student_STARTUP/${source}_source/$target\_unlabeled_20/checkpoint_best.pkl --embedding_load_path_version 1
    python compile_result.py --result_file results/STARTUP_${source}/$source\_$target\_5way.csv
done
