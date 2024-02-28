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
from collections import OrderedDict

import torch
from torch import Tensor, nn
from torchvision import models

__all__ = [
    "VGGFeatureExtractor",
]

vgg_layer_names = {
    "vgg11": [
        "conv1_1", "relu1_1", "pool1",
        "conv2_1", "relu2_1", "pool2",
        "conv3_1", "relu3_1", "conv3_2", "relu3_2", "pool3",
        "conv4_1", "relu4_1", "conv4_2", "relu4_2", "pool4",
        "conv5_1", "relu5_1", "conv5_2", "relu5_2", "pool5"
    ],
    "vgg13": [
        "conv1_1", "relu1_1", "conv1_2", "relu1_2", "pool1",
        "conv2_1", "relu2_1", "conv2_2", "relu2_2", "pool2",
        "conv3_1", "relu3_1", "conv3_2", "relu3_2", "pool3",
        "conv4_1", "relu4_1", "conv4_2", "relu4_2", "pool4",
        "conv5_1", "relu5_1", "conv5_2", "relu5_2", "pool5"
    ],
    "vgg16": [
        "conv1_1", "relu1_1", "conv1_2", "relu1_2", "pool1",
        "conv2_1", "relu2_1", "conv2_2", "relu2_2", "pool2",
        "conv3_1", "relu3_1", "conv3_2", "relu3_2", "conv3_3", "relu3_3", "pool3",
        "conv4_1", "relu4_1", "conv4_2", "relu4_2", "conv4_3", "relu4_3", "pool4",
        "conv5_1", "relu5_1", "conv5_2", "relu5_2", "conv5_3", "relu5_3", "pool5"
    ],
    "vgg19": [
        "conv1_1", "relu1_1", "conv1_2", "relu1_2", "pool1",
        "conv2_1", "relu2_1", "conv2_2", "relu2_2", "pool2",
        "conv3_1", "relu3_1", "conv3_2", "relu3_2", "conv3_3", "relu3_3", "conv3_4", "relu3_4", "pool3",
        "conv4_1", "relu4_1", "conv4_2", "relu4_2", "conv4_3", "relu4_3", "conv4_4", "relu4_4", "pool4",
        "conv5_1", "relu5_1", "conv5_2", "relu5_2", "conv5_3", "relu5_3", "conv5_4", "relu5_4", "pool5"
    ]
}


class VGGFeatureExtractor(nn.Module):
    def __init__(
            self,
            arch_name: str = "vgg19",
            layer_name_list: list = None,
            normalize: bool = True,
    ) -> None:
        super(VGGFeatureExtractor, self).__init__()
        assert arch_name in vgg_layer_names.keys(), f"VGG model only support {vgg_layer_names.keys()}"

        if layer_name_list is None:
            layer_name_list = ["conv1_2", "conv2_2", "conv3_4", "conv4_4", "conv5_4"]
        self.layer_name_list = layer_name_list
        self.normalize = normalize

        self.layer_names = vgg_layer_names[arch_name]
        if arch_name == "vgg11":
            vgg_model = models.vgg11(weights=models.VGG11_Weights.IMAGENET1K_V1)
        elif arch_name == "vgg13":
            vgg_model = models.vgg13(weights=models.VGG13_Weights.IMAGENET1K_V1)
        elif arch_name == "vgg16":
            vgg_model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        else:
            vgg_model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)

        # find the last layer index
        max_layer_index = 0
        for layer_name in layer_name_list:
            layer_index = self.layer_names.index(layer_name)
            if layer_index > max_layer_index:
                max_layer_index = layer_index

        vgg_model_features = vgg_model.features[:max_layer_index + 1]
        new_vgg_model_features = OrderedDict()
        for k, v in zip(self.layer_names, vgg_model_features):
            new_vgg_model_features[k] = v

        # build the feature extraction model
        self.vgg_model_features = nn.Sequential(new_vgg_model_features)
        # set to validation mode
        self.vgg_model_features.eval()
        # Freeze model parameters.
        for model_parameters in self.vgg_model_features.parameters():
            model_parameters.requires_grad = False

        if normalize:
            self.register_buffer("mean", torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("std", torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: Tensor) -> dict:
        if self.normalize:
            x = (x - self.mean) / self.std

        outputs = {}
        for key, layer in self.vgg_model_features._modules.items():
            x = layer(x)
            if key in self.layer_name_list:
                outputs[key] = x.clone()

        return outputs
