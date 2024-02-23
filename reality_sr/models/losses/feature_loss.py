# Copyright Lornatang. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import torch
from torch import Tensor, nn
from torch.nn import functional as F_torch
from torchvision import models, transforms
from torchvision.models.feature_extraction import create_feature_extractor

__all__ = [
    "FeatureLoss",
]


class FeatureLoss(nn.Module):
    """Constructs a content loss function based on the VGG19 network.
    Using high-level feature mapping layers from the latter layers will focus more on the texture content of the image.

    Paper reference list:
        -`Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network <https://arxiv.org/pdf/1609.04802.pdf>` paper.
        -`ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks                    <https://arxiv.org/pdf/1809.00219.pdf>` paper.
        -`Perceptual Extreme Super Resolution Network with Receptive Field Block               <https://arxiv.org/pdf/2005.12597.pdf>` paper.

     """

    def __init__(
            self,
            feature_model_extractor_nodes=None,
            feature_model_normalize_mean: list = None,
            feature_model_normalize_std: list = None,
    ) -> None:
        super(FeatureLoss, self).__init__()
        # Get the name of the specified feature extraction node
        if feature_model_extractor_nodes is None:
            feature_model_extractor_nodes = ["features.2", "features.7", "features.16", "features.25", "features.34"]
        if feature_model_normalize_mean is None:
            feature_model_normalize_mean = [0.485, 0.456, 0.406]
        if feature_model_normalize_std is None:
            feature_model_normalize_std = [0.229, 0.224, 0.225]
        self.feature_model_extractor_nodes = feature_model_extractor_nodes
        # Load the VGG19 model trained on the ImageNet dataset.
        model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        # Extract the thirty-sixth layer output in the VGG19 model as the content loss.
        self.feature_extractor = create_feature_extractor(model, feature_model_extractor_nodes)

        # set to validation mode
        self.feature_extractor.eval()

        # The preprocessing method of the input data.
        # This is the VGG model preprocessing method of the ImageNet dataset.
        self.normalize = transforms.Normalize(feature_model_normalize_mean, feature_model_normalize_std)

        # Freeze model parameters.
        for model_parameters in self.feature_extractor.parameters():
            model_parameters.requires_grad = False

    def forward(self, sr_tensor: Tensor, gt_tensor: Tensor) -> Tensor:
        assert sr_tensor.size() == gt_tensor.size(), "Two tensor must have the same size"

        device = sr_tensor.device

        loss_list = []
        # Standardized operations
        sr_tensor = self.normalize(sr_tensor)
        gt_tensor = self.normalize(gt_tensor)

        # VGG19 conv3_4 feature extraction
        sr_feature = self.feature_extractor(sr_tensor)
        gt_feature = self.feature_extractor(gt_tensor)

        for i in range(len(self.feature_model_extractor_nodes)):
            loss_list.append(F_torch.l1_loss(sr_feature[self.feature_model_extractor_nodes[i]], gt_feature[self.feature_model_extractor_nodes[i]]))

        loss_list = torch.Tensor([loss_list]).to(device=device)

        return loss_list
