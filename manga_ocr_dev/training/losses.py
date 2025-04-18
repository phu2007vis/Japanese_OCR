# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

import torch
import torch.nn as nn
from manga_ocr_dev.env import LABEL_SMOOTHING_FACTOR

global PRINTED
PRINTED= False
def fixed_cross_entropy(
	source: torch.Tensor,
	target: torch.Tensor,
	num_items_in_batch: Optional[int] = None,
	ignore_index: int = -100,
	label_smoothing = None,
	**kwargs,
) -> torch.Tensor:
	global PRINTED
	if label_smoothing is None:
		assert LABEL_SMOOTHING_FACTOR >=0 and LABEL_SMOOTHING_FACTOR<1 ,f"LABEL_SMOOTHING_FACTOR must in range(0,1) but take {LABEL_SMOOTHING_FACTOR} value"
		label_smoothing = LABEL_SMOOTHING_FACTOR 
	if not PRINTED:
		print(f"Trainning with label smoothing factor: {label_smoothing}")
		PRINTED = True
  
	reduction = "sum" if num_items_in_batch is not None else "mean"
	loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction,label_smoothing = label_smoothing)
	if reduction == "sum":
		loss = loss / num_items_in_batch
	return loss


def ForCausalLMLoss(
	logits,
	labels,
	vocab_size: int,
	num_items_in_batch: Optional[int] = None,
	ignore_index: int = -100,
	shift_labels: Optional[torch.Tensor] = None,
	label_smoothing = None,
	**kwargs,
) -> torch.Tensor:
	# Upcast to float if we need to compute the loss to avoid potential precision issues
	logits = logits.float()

	if shift_labels is None:
		# Shift so that tokens < n predict n
		labels = nn.functional.pad(labels, (0, 1), value=ignore_index)
		shift_labels = labels[..., 1:].contiguous()

	# Flatten the tokens
	logits = logits.view(-1, vocab_size)
	shift_labels = shift_labels.view(-1)
	# Enable model parallelism
	shift_labels = shift_labels.to(logits.device)
 
	loss = fixed_cross_entropy(	logits, 
								shift_labels,
								num_items_in_batch,
								ignore_index,
								label_smoothing=label_smoothing ,
								**kwargs
								)
	return loss
