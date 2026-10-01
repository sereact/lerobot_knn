"""Pre/post-processing pipelines of the KNN policy.

The policy consumes raw observations (images in ``[0, 1]``, un-normalised state) and returns raw
actions copied from the demonstrations, so the pipelines only batch and move tensors; any feature
standardisation happens inside the policy with statistics stored alongside its memory.
"""

from __future__ import annotations

from typing import Any

import torch

from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    RenameObservationsProcessorStep,
)
from lerobot.processor.factory import make_policy_processor_pipelines

from .configuration_knn import KNNConfig


def make_knn_pre_post_processors(
    config: KNNConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    device = config.device or "cpu"
    return make_policy_processor_pipelines(
        input_steps=[
            RenameObservationsProcessorStep(rename_map={}),
            AddBatchDimensionProcessorStep(),
            DeviceProcessorStep(device=device),
        ],
        output_steps=[DeviceProcessorStep(device="cpu")],
    )
