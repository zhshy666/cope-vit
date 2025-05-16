conda activate cv

python3 -m main.py --model cope_2d_v2_deit_small_patch4_LS --data-path ../data/ --output_dir ../save/temp/ --data-set CIFAR10 --batch-size 128 --epochs 400 --input-size 32 --wandb-project Rope-2D-cifar10 --wandb-name "eval" --eval_checkpoint "../save/[p4]2d-cope+ape+patch4/best_checkpoint.pth" --eval --device 'cpu'


python3 -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --model cope_2d_v2_deit_small_patch4_LS --data-path ../data/ --output_dir ../save/[p4]2d-cope+ape+patch4/ --data-set CIFAR10 --batch-size 128 --epochs 400 --input-size 32 --wandb-project Rope-2D-cifar10 --wandb-name "[p4]2d cope+ape"

python3 -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --model cope_2d_v2_deit_small_patch4_LS --data-path ../data/tiny-imagenet-200/ --output_dir ../save/[p4]2d-cope+ape+patch4+tiny/ --data-set TINY --batch-size 128 --epochs 400 --input-size 64 --wandb-project Rope-2D-tiny --wandb-name "[p4]2d cope+ape"

python3 -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --model cope_2d_v2_deit_small_patch4_LS --data-path ../data/ --output_dir ../save/[p4]2d-cope+ape+patch4+cifar100/ --data-set CIFAR --batch-size 128 --epochs 400 --input-size 32 --wandb-project Rope-2D-cifar100 --wandb-name "[p4]2d cope+ape"

python3 -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --model cope_2d_v2_deit_small_patch16_LS --data-path ../data/ --output_dir ../save/[]2d-cope+ape+224/ --data-set CIFAR10 --batch-size 128 --epochs 400 --input-size 224 --wandb-project Rope-2D-cifar10 --wandb-name "[]2d cope+ape+224"

python3 -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --model cope_2d_v2_deit_small_patch16_LS --data-path ../data/tiny-imagenet-200/ --output_dir ../save/[]2d-cope+ape+tiny+224/ --data-set TINY --batch-size 128 --epochs 400 --input-size 224 --wandb-project Rope-2D-tiny --wandb-name "[]2d cope+ape+224"

python3 -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --model cope_2d_v2_deit_small_patch16_LS --data-path ../data/ --output_dir ../save/[]2d-cope+ape+cifar100+224/ --data-set CIFAR --batch-size 128 --epochs 400 --input-size 224 --wandb-project Rope-2D-cifar100 --wandb-name "[]2d cope+ape+224"

python3 -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --model cope_2d_v2_deit_small_patch4_LS --data-path ../data/ --output_dir ../save/[p4]2d-cope+ape+patch4+224/ --data-set CIFAR10 --batch-size 128 --epochs 400 --input-size 224 --wandb-project Rope-2D-cifar10 --wandb-name "[p4]2d cope+ape+224"

python3 -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --model cope_2d_v2_deit_small_patch4_LS --data-path ../data/tiny-imagenet-200/ --output_dir ../save/[p4]2d-cope+ape+patch4+tiny+224/ --data-set TINY --batch-size 128 --epochs 400 --input-size 224 --wandb-project Rope-2D-tiny --wandb-name "[p4]2d cope+ape+224"

python3 -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --model cope_2d_v2_deit_small_patch4_LS --data-path ../data/ --output_dir ../save/[p4]2d-cope+ape+patch4+cifar100+224/ --data-set CIFAR --batch-size 128 --epochs 400 --input-size 224 --wandb-project Rope-2D-cifar100 --wandb-name "[p4]2d cope+ape+224"


