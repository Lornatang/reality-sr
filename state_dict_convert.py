# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
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
from pathlib import Path

import torch
from omegaconf import OmegaConf

from yolov6.models.yolov6 import YOLOv6
from reality_sr.models import rrdbnet_x4
from reality_sr.utils.checkpoint import strip_optimizer

OLD_MODEL_PATH = "RealESRGAN_x4plus_anime_6B.pth"
NEW_MODEL_PATH = "realesrgan_x4_6b-anime_degradation.pkl"

model = rrdbnet_x4(num_rrdb=6)
model.eval()
new_state_dict = model.state_dict()
old_state_dict = torch.load(OLD_MODEL_PATH, map_location=torch.device("cpu"))["params_ema"]

new_list = []
old_list = []

for k, v in new_state_dict.items():
    new_list.append(k)

for k, v in old_state_dict.items():
    old_list.append(k)

print(new_list)
print(old_list)
print(len(new_list))
print(len(old_list))

for i in range(len(new_list)):
    new_state_dict[new_list[i]] = old_state_dict[old_list[i]]
model.load_state_dict(new_state_dict)
for p in model.parameters():
    p.requires_grad = False

torch.save({"model": model, "ema": None, "optimizer": None, "scheduler": None, "updates": None}, NEW_MODEL_PATH)


