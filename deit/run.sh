conda activate cv

python3 -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --model cope_2d_v2_deit_small_patch16_LS_q --data-path ../data/tiny-imagenet-200/ --output_dir ../save/[new]2d-cope+ape+q/ --data-set TINY --batch-size 128 --epochs 400 --input-size 64 --wandb-project Rope-2D-tiny --wandb-name "[new]2d cope+ape+q"

python3 -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --model cope_2d_v2_deit_small_patch16_LS --data-path ../data/tiny-imagenet-200/ --output_dir ../save/[new]2d-cope+ape+k/ --data-set TINY --batch-size 128 --epochs 400 --input-size 64 --wandb-project Rope-2D-tiny --wandb-name "[new]2d cope+ape+k"

python3 -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --model cope_2d_v2_deit_small_patch16_LS_v --data-path ../data/tiny-imagenet-200/ --output_dir ../save/[new]2d-cope+ape+v/ --data-set TINY --batch-size 128 --epochs 400 --input-size 64 --wandb-project Rope-2D-tiny --wandb-name "[new]2d cope+ape+v"

python3 -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --model cope_2d_v2_deit_small_patch16_LS_qk --data-path ../data/tiny-imagenet-200/ --output_dir ../save/[new]2d-cope+ape+qk/ --data-set TINY --batch-size 128 --epochs 400 --input-size 64 --wandb-project Rope-2D-tiny --wandb-name "[new]2d cope+ape+qk"

python3 -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --model cope_2d_v2_deit_small_patch16_LS_qv --data-path ../data/tiny-imagenet-200/ --output_dir ../save/[new]2d-cope+ape+qv/ --data-set TINY --batch-size 128 --epochs 400 --input-size 64 --wandb-project Rope-2D-tiny --wandb-name "[new]2d cope+ape+qv"

python3 -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --model cope_2d_v2_deit_small_patch16_LS_kv --data-path ../data/tiny-imagenet-200/ --output_dir ../save/[new]2d-cope+ape+kv/ --data-set TINY --batch-size 128 --epochs 400 --input-size 64 --wandb-project Rope-2D-tiny --wandb-name "[new]2d cope+ape+kv"

python3 -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --model cope_2d_v2_deit_small_patch16_LS_qkv --data-path ../data/tiny-imagenet-200/ --output_dir ../save/[new]2d-cope+ape+qkv/ --data-set TINY --batch-size 128 --epochs 400 --input-size 64 --wandb-project Rope-2D-tiny --wandb-name "[new]2d cope+ape+qkv"

