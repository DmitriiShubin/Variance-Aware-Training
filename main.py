#baseline models

#pre-training encoders

#pre-trained models
from experiments.segmentation.pre_trained_patch.run_experiment import run as run_pre_trained_patch

#adversarial models
from experiments.segmentation.adversarial_network_train_val_early.run_experiment import (
    run as run_adversarial_network_train_val_early,
)
from experiments.segmentation.adversarial_network_train_val_late.run_experiment import (
    run as run_adversarial_network_train_val_late
)






def main():

    # ACDC

    # pre-training
    # Self-supervised contrastive
    #run_contrastive_pre_train(experiment='./experiments/contrastive_loss_encoder/config_ACDC.yml',gpu='7')

    # run_contrastive_pre_train(experiment='./experiments/contrastive_loss_encoder/config_ACDC_1.yml', gpu='7')
    # run_contrastive_pre_train(experiment='./experiments/contrastive_loss_encoder/config_ACDC_2.yml', gpu='7')
    # run_contrastive_pre_train(experiment='./experiments/contrastive_loss_encoder/config_ACDC_3.yml', gpu='7')

    # Self-supervised rotation
    #run_rotation_pre_train(experiment='./experiments/rotation_encoder/config_ACDC.yml',gpu='7')

    # Self-supervised patch
    #run_patch_pre_train(experiment='./experiments/patch_encoder/config_ACDC.yml',gpu='6')

    # baseline, without pre-train
    #run_baseline(experiment='./experiments/baseline/config_ACDC_2.yml')
    #run_baseline(experiment='./experiments/baseline/config_ACDC_4.yml')
    #run_baseline(experiment='./experiments/baseline/config_ACDC_8.yml')
    #run_baseline(experiment='./experiments/baseline/config_ACDC_UB.yml')

    # pre-trained contrastive
    #run_pre_trained_contrastive(experiment='./experiments/pre_trained_contrastive/config_ACDC_2.yml')
    #run_pre_trained_contrastive(experiment='./experiments/pre_trained_contrastive/config_ACDC_4.yml')
    #run_pre_trained_contrastive(experiment='./experiments/pre_trained_contrastive/config_ACDC_8.yml')

    # # pre-trained rotation
    #run_pre_trained_rotation(experiment='./experiments/pre_trained_rotation/config_ACDC_2.yml')
    # run_pre_trained_rotation(experiment='./experiments/pre_trained_rotation/config_ACDC_4.yml')
    # run_pre_trained_rotation(experiment='./experiments/pre_trained_rotation/config_ACDC_8.yml')

    # run_pre_trained_patch(experiment='./experiments/pre_trained_patch/config_ACDC_2.yml')
    # run_pre_trained_patch(experiment='./experiments/pre_trained_patch/config_ACDC_4.yml')
    # run_pre_trained_patch(experiment='./experiments/pre_trained_patch/config_ACDC_8.yml')


    # Single-stage self-supervised early flat
    run_adversarial_network_train_val_early(experiment='./experiments/segmentation/adversarial_network_train_val_early/config_ACDC_2.yml')
    # run_adversarial_network_train_val_early(experiment='./experiments/segmentation/adversarial_network_train_val_early/config_ACDC_4.yml')
    # run_adversarial_network_train_val_early(experiment='./experiments/segmentation/adversarial_network_train_val_early/config_ACDC_8.yml')



    # Single-stage self-supervised late flat
    # run_adversarial_network_train_val_late(experiment='./experiments/adversarial_network_train_val_late/config_ACDC_2.yml')
    # run_adversarial_network_train_val_late(experiment='./experiments/adversarial_network_train_val_late/config_ACDC_4.yml')
    # run_adversarial_network_train_val_late(experiment='./experiments/adversarial_network_train_val_late/config_ACDC_8.yml')

    ###########################################################################
    # BRaTs

    # baseline, without pre-train
    # run_baseline(experiment='./experiments/baseline/config_brats_2.yml')
    # run_baseline(experiment='./experiments/baseline/config_brats_4.yml')
    # run_baseline(experiment='./experiments/baseline/config_brats_8.yml')
    # run_baseline(experiment='./experiments/baseline/config_brats_UB.yml')

    return None


if __name__ == "__main__":
    main()
