from __future__ import annotations

import numpy as np
import os
import torch
from typing import TYPE_CHECKING

import onnxruntime as ort

if TYPE_CHECKING:
    from typing import Callable


class ParkourOnnxPolicy:
    def __init__(self, model_dir: str, get_subobs_func: Callable, depth_shape: tuple, proprio_slice: slice):
        available_providers = ort.get_available_providers()
        ort_providers = ["CPUExecutionProvider"]
        if torch.cuda.is_available() and "CUDAExecutionProvider" in available_providers:
            ort_providers.insert(0, "CUDAExecutionProvider")
        integrated_path = os.path.join(model_dir, "factorized_encoder_actor.onnx")
        self.integrated_actor = (
            ort.InferenceSession(integrated_path, providers=ort_providers)
            if os.path.exists(integrated_path)
            else None
        )
        if self.integrated_actor is not None:
            self.encoder = None
            self.actor = None
            self.actor_gate = self.integrated_actor
            self.get_subobs_func = get_subobs_func
            self.depth_shape = depth_shape
            self.proprio_slice = proprio_slice
            self.actor_input_name = self.integrated_actor.get_inputs()[0].name
            self.last_actor_input = None
            self.last_gate_weights = None
            return

        self.encoder = ort.InferenceSession(os.path.join(model_dir, "0-depth_encoder.onnx"), providers=ort_providers)
        self.actor = ort.InferenceSession(os.path.join(model_dir, "actor.onnx"), providers=ort_providers)
        gate_path = os.path.join(model_dir, "actor_moe_gate.onnx")
        self.actor_gate = (
            ort.InferenceSession(gate_path, providers=ort_providers) if os.path.exists(gate_path) else None
        )
        self.get_subobs_func = get_subobs_func
        self.depth_shape = depth_shape
        self.proprio_slice = proprio_slice
        self.actor_input_name = self.actor.get_inputs()[0].name
        self.last_actor_input = None
        self.last_gate_weights = None

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        if self.integrated_actor is not None:
            actor_input = obs.cpu().numpy()
            actor_output, self.last_gate_weights = self.integrated_actor.run(
                None, {self.actor_input_name: actor_input}
            )
            self.last_actor_input = actor_input
            return torch.from_numpy(actor_output).to(obs.device)

        depth_image_input = self.get_subobs_func(obs)
        depth_image_input = depth_image_input.cpu().numpy()
        depth_image_input = depth_image_input.reshape((-1, *self.depth_shape))
        depth_image_output = self.encoder.run(None, {self.encoder.get_inputs()[0].name: depth_image_input})[0]
        actor_input = np.concatenate(
            [
                obs.cpu().numpy()[:, self.proprio_slice],
                depth_image_output,
            ],
            axis=1,
        )
        actor_output = self.actor.run(None, {self.actor_input_name: actor_input})[0]
        self.last_actor_input = actor_input
        if self.actor_gate is not None:
            self.last_gate_weights = self.actor_gate.run(
                None, {self.actor_gate.get_inputs()[0].name: actor_input}
            )[0]
        else:
            self.last_gate_weights = None
        return torch.from_numpy(actor_output).to(obs.device)


def load_parkour_onnx_model(
    model_dir: str, get_subobs_func: Callable, depth_shape: tuple, proprio_slice: slice
) -> Callable:
    """Load the ONNX model as policy, but only for parkour task setting."""
    return ParkourOnnxPolicy(model_dir, get_subobs_func, depth_shape, proprio_slice)
