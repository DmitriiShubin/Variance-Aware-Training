for FOLD in 0
do
  python main.py --start_fold $FOLD --n_epochs 2 --batch_size 32 --lr 1e-4 --gpu 0,1,2 --model unet
  python main.py --start_fold $FOLD --n_epochs 2 --batch_size 32 --lr 1e-4 --gpu 0,1,2 --alpha 0.001 --model adv_unet
  python main.py --start_fold $FOLD --n_epochs 2 --batch_size 32 --lr 1e-4 --gpu 0,1,2 --model fpn
  python main.py --start_fold $FOLD --n_epochs 2 --batch_size 32 --lr 1e-4 --gpu 0,1,2,3 --alpha 0.2 --model adv_fpn
done
