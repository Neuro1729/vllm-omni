# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Generate a video with SkyReels V2 T2V via vLLM-Omni."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.platforms import current_omni_platform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a video with SkyReels V2 T2V.")
    parser.add_argument(
        "--model",
        default="Skywork/SkyReels-V2-T2V-14B-540P-Diffusers",
        help="Diffusers SkyReels V2 T2V model ID or local path.",
    )
    parser.add_argument(
        "--prompt",
        default=(
            "A serene lake surrounded by towering mountains, with a flock of birds "
            "gracefully gliding across the water surface."
        ),
        help="Text prompt.",
    )
    parser.add_argument("--negative-prompt", default="", help="Negative prompt.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--guidance-scale", type=float, default=6.0, help="CFG scale.")
    parser.add_argument("--height", type=int, default=544, help="Video height (540P default).")
    parser.add_argument("--width", type=int, default=960, help="Video width (540P default).")
    parser.add_argument("--num-frames", type=int, default=97, help="Number of frames.")
    parser.add_argument("--num-inference-steps", type=int, default=50, help="Sampling steps.")
    parser.add_argument(
        "--flow-shift",
        type=float,
        default=8.0,
        help="Scheduler flow_shift (8.0 for SkyReels T2V).",
    )
    parser.add_argument("--output", type=str, default="skyreels_v2_output.mp4", help="Output mp4 path.")
    parser.add_argument("--fps", type=int, default=24, help="Output FPS.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = current_omni_platform.device_type
    generator = torch.Generator(device=device).manual_seed(args.seed)

    omni = Omni(
        model=args.model,
        flow_shift=args.flow_shift,
        boundary_ratio=0.0,
    )

    sampling_params = OmniDiffusionSamplingParams(
        height=args.height,
        width=args.width,
        generator=generator,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        num_frames=args.num_frames,
    )

    frames = omni.generate(
        {
            "prompt": args.prompt,
            "negative_prompt": args.negative_prompt,
        },
        sampling_params,
    )

    if isinstance(frames, list):
        frames = frames[0] if frames else None
    if isinstance(frames, OmniRequestOutput):
        video = frames.images
        if isinstance(video, list) and len(video) == 1:
            video = video[0]
    else:
        video = frames

    if isinstance(video, dict) and "video" in video:
        video = video["video"]
    if isinstance(video, (list, tuple)) and len(video) == 1:
        video = video[0]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from diffusers.utils import export_to_video
    except ImportError as exc:
        raise ImportError("diffusers is required for export_to_video.") from exc

    if isinstance(video, torch.Tensor):
        video_tensor = video.detach().cpu()
        if video_tensor.dim() == 5:
            if video_tensor.shape[1] in (3, 4):
                video_tensor = video_tensor[0].permute(1, 2, 3, 0)
            else:
                video_tensor = video_tensor[0]
        if video_tensor.is_floating_point():
            video_tensor = video_tensor.clamp(-1, 1) * 0.5 + 0.5
        video_array = video_tensor.float().numpy()
    else:
        video_array = video
        if hasattr(video_array, "shape") and getattr(video_array, "ndim", 0) == 5:
            video_array = video_array[0]

    if isinstance(video_array, np.ndarray) and video_array.ndim == 4:
        video_array = list(video_array)

    export_to_video(video_array, str(output_path), fps=args.fps)
    print(f"Saved generated video to {output_path}")
    omni.close()


if __name__ == "__main__":
    main()
