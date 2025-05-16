conda activate cv

# python3 -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --model cope_2d_v2_deit_small_patch4_LS --data-path ../data/tiny-imagenet-200/ --output_dir ../save/[p4]2d-cope+ape+patch4+tiny/ --data-set TINY --batch-size 128 --epochs 400 --input-size 64 --wandb-project Rope-2D-tiny --wandb-name "[p4]2d cope+ape"

# python3 -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --model cope_2d_v2_deit_small_patch4_LS --data-path ../data/ --output_dir ../save/[p4]2d-cope+ape+patch4+cifar100/ --data-set CIFAR --batch-size 128 --epochs 400 --input-size 32 --clip-grad 1.0 --wandb-project Rope-2D-cifar100 --wandb-name "[p4]2d cope+ape"


# python3 -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --model rope_axial_ape_deit_small_patch4_LS --data-path ../data/ --output_dir ../save/[p4]2d-rope+ape/ --data-set CIFAR10 --batch-size 128 --epochs 400 --input-size 32 --wandb-project Rope-2D-cifar10 --wandb-name "[p4]2d rope+ape"

# python3 -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --model rope_mixed_ape_deit_small_patch4_LS --data-path ../data/tiny-imagenet-200/ --output_dir ../save/[p4]2d-rope+ape+tiny/ --data-set TINY --batch-size 128 --epochs 400 --input-size 64 --wandb-project Rope-2D-tiny --wandb-name "[p4]2d rope+ape"

# python3 -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --model rope_axial_ape_deit_small_patch4_LS --data-path ../data/ --output_dir ../save/[p4]2d-rope+ape+cifar100/ --data-set CIFAR --batch-size 128 --epochs 400 --input-size 32 --wandb-project Rope-2D-cifar100 --wandb-name "[p4]2d rope+ape"



python3 -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --model cope_2d_v2_sep_keys_deit_small_patch4_LS_q --data-path ../data/ --output_dir ../save/[q]2d-cope+ape+sk/ --data-set CIFAR10 --batch-size 128 --epochs 400 --input-size 32 --wandb-project Rope-2D-cifar10 --wandb-name "[p4]2d cope(sk+q)"

# python3 -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --model cope_2d_v2_sep_keys_deit_small_patch4_LS --data-path ../data/tiny-imagenet-200/ --output_dir ../save/[p4]2d-cope+ape+sk+tiny/ --data-set TINY --batch-size 128 --epochs 400 --input-size 64 --wandb-project Rope-2D-tiny --wandb-name "[p4]2d cope(sk)+ape"

python3 -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --model cope_2d_v2_sep_keys_deit_small_patch4_LS_q --data-path ../data/ --output_dir ../save/[q]2d-cope+ape+sk+cifar100/ --data-set CIFAR --batch-size 128 --epochs 400 --input-size 32 --wandb-project Rope-2D-cifar100 --wandb-name "[p4]2d cope(sk+q)"



