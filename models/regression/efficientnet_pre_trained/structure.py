import torch
import yaml

from models.classification.encoder_contrastive_classification.structure import EfficientNet as effnet_contrastive
from models.classification.encoder_patch_classification.structure import EfficientNet as effnet_patch
from models.classification.encoder_rotation_classification.structure import EfficientNet as effnet_rotation
from efficientnet_pytorch import EfficientNet as effnet


class EfficientNet(effnet):
    def __init__(self, blocks_args=None, global_params=None):

        super(EfficientNet, self).__init__(blocks_args=blocks_args, global_params=global_params)

    def forward(self, inputs):
        """EfficientNet's forward function.
           Calls extract_features to extract features, applies final linear layer, and returns logits.
        Args:
            inputs (tensor): Input tensor.
        Returns:
            Output of this model after processing.
        """
        # Convolution layers
        x = self.encoder.extract_features(inputs)

        # Pooling and final linear layer
        x = self._avg_pooling(x)
        if self._global_params.include_top:
            x = x.flatten(start_dim=1)
            x = self._dropout(x)
            x = torch.softmax(self._fc(x), dim=1)
        return x

    def freeze_layers(self):

        for param in self.parameters():
            print(param)
            param.requires_grad = False

        return True

    def load_self_supervised_model(
        self, type_pretrain: str, pre_trained_model: str, pre_trained_model_ssl: str, device
    ):

        hparams = yaml.load(open(pre_trained_model_ssl + '_hparams.yml'), Loader=yaml.FullLoader)

        if type_pretrain == 'contrastive':
            self.encoder = effnet_contrastive.from_pretrained(pre_trained_model).to(device)
            self.encoder.build_projection_network(hparams['model']['emb_dim'], device=device)
        elif type_pretrain == 'patch':
            self.encoder = effnet_patch.from_pretrained(pre_trained_model).to(device)
            self.encoder.build_projection_network(device=device)
        elif type_pretrain == 'rotation':
            self.encoder = effnet_rotation.from_pretrained(pre_trained_model, num_classes=4).to(device)

        self.encoder.load_state_dict(torch.load(pre_trained_model_ssl + '.pt'))

        return True
