python main.py --start_fold 0 --n_epochs 100 --batch_size 15 --lr 1e-3 --gpu 1 --model linknet

python main.py --start_fold 0 --n_epochs 100 --batch_size 15 --lr 1e-3 --gpu 1 --alpha 1e-4 --adv_threshold 0.5 --model adv_linknet
python main.py --start_fold 0 --n_epochs 100 --batch_size 15 --lr 1e-3 --gpu 1 --alpha 1e-4 --adv_threshold 0.6 --model adv_linknet
python main.py --start_fold 0 --n_epochs 100 --batch_size 15 --lr 1e-3 --gpu 1 --alpha 1e-4 --adv_threshold 0.74 --model adv_linknet

python main.py --start_fold 0 --n_epochs 100 --batch_size 15 --lr 1e-3 --gpu 1 --alpha 1e-3 --adv_threshold 0.5 --model adv_linknet
python main.py --start_fold 0 --n_epochs 100 --batch_size 15 --lr 1e-3 --gpu 1 --alpha 1e-3 --adv_threshold 0.6 --model adv_linknet
python main.py --start_fold 0 --n_epochs 100 --batch_size 15 --lr 1e-3 --gpu 1 --alpha 1e-3 --adv_threshold 0.74 --model adv_linknet



