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

from reality_sr.models.vgg_feature_extractor import VGGFeatureExtractor

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
            arch_name: str,
            layer_name_list: list,
            normalize: bool,
    ) -> None:
        super(FeatureLoss, self).__init__()
        self.vgg_feature_extractor = VGGFeatureExtractor(
            arch_name=arch_name,
            layer_name_list=layer_name_list,
            normalize=normalize)

    def forward(self, inputs: Tensor, target: Tensor) -> Tensor:
        assert inputs.size() == target.size(), "Two tensor must have the same size"

        device = inputs.device

        inputs_features = self.vgg_feature_extractor(inputs)
        target_features = self.vgg_feature_extractor(target.detach())

        loss_list = []
        for k in inputs_features.keys():
            loss_list.append(F_torch.l1_loss(inputs_features[k], target_features[k]))

        return torch.Tensor([loss_list]).to(device=device)
