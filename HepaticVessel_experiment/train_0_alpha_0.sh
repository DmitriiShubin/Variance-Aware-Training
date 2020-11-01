python main.py --start_fold 0 --n_epochs 100 --batch_size 28 --lr 1e-4 --gpu 0,1,2,4 --model unet
python main.py --start_fold 0 --n_epochs 100 --batch_size 28 --lr 1e-4 --gpu 0,1,2,3,4,5,6 --alpha 0.01 --model adv_unet
python main.py --start_fold 0 --n_epochs 100 --batch_size 28 --lr 1e-4 --gpu 0,1,2,3,4,5,6 --alpha 0.05 --model adv_unet
python main.py --start_fold 0 --n_epochs 100 --batch_size 28 --lr 1e-4 --gpu 0,1,2,3,4,5,6 --alpha 0.005 --model adv_unet

