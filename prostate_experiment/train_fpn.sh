python main.py --start_fold 0 --n_epochs 100 --batch_size 15 --lr 1e-3 --gpu 0 --model fpn &

python main.py --start_fold 0 --n_epochs 100 --batch_size 15 --lr 1e-3 --gpu 1 --alpha 1e-4 --adv_threshold 0.5 --model adv_fpn &
python main.py --start_fold 0 --n_epochs 100 --batch_size 15 --lr 1e-3 --gpu 2 --alpha 1e-4 --adv_threshold 0.6 --model adv_fpn &
python main.py --start_fold 0 --n_epochs 100 --batch_size 15 --lr 1e-3 --gpu 3 --alpha 1e-4 --adv_threshold 0.74 --model adv_fpn &

python main.py --start_fold 0 --n_epochs 100 --batch_size 15 --lr 1e-3 --gpu 4 --alpha 1e-3 --adv_threshold 0.5 --model adv_fpn &
python main.py --start_fold 0 --n_epochs 100 --batch_size 15 --lr 1e-3 --gpu 5 --alpha 1e-3 --adv_threshold 0.6 --model adv_fpn &
python main.py --start_fold 0 --n_epochs 100 --batch_size 15 --lr 1e-3 --gpu 6 --alpha 1e-3 --adv_threshold 0.74 --model adv_fpn &



