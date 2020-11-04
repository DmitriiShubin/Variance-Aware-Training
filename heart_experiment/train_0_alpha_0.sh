python main.py --start_fold 0 --n_epochs 100 --batch_size 32 --lr 1e-4 --gpu 0,1,2,4 --alpha 0.01 --model fpn

python main.py --start_fold 0 --n_epochs 100 --batch_size 32 --lr 1e-4 --gpu 0,1,2,4 --alpha 0.1 --model adv_fpn
python main.py --start_fold 0 --n_epochs 100 --batch_size 32 --lr 1e-4 --gpu 0,1,2,4 --alpha 0.05 --model adv_fpn
python main.py --start_fold 0 --n_epochs 100 --batch_size 32 --lr 1e-4 --gpu 0,1,2,4 --alpha 0.01 --model adv_fpn
python main.py --start_fold 0 --n_epochs 100 --batch_size 32 --lr 1e-4 --gpu 0,1,2,4 --alpha 0.005 --model adv_fpn
python main.py --start_fold 0 --n_epochs 100 --batch_size 32 --lr 1e-4 --gpu 0,1,2,4 --alpha 0.001 --model adv_fpn
