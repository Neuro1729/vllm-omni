#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Benchmark API frontend scaling with synthetic video chat requests."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import httpx
import numpy as np


@dataclass(slots=True)
class RequestResult:
    status: int | None
    elapsed_s: float
    error: str | None


def _make_video_data_uri(
    path: Path,
    *,
    seed: int,
    width: int,
    height: int,
    frames: int,
    fps: float,
) -> str:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create benchmark video at {path}")

    background = np.zeros((height, width, 3), dtype=np.uint8)
    background[:, :, 0] = (seed * 17) % 256
    background[:, :, 1] = (seed * 31) % 256
    background[:, :, 2] = (seed * 47) % 256
    radius = max(8, min(width, height) // 8)
    try:
        for frame_id in range(frames):
            frame = background.copy()
            x = int((frame_id * 37 + seed * 101) % width)
            y = int((frame_id * 23 + seed * 67) % height)
            cv2.circle(frame, (x, y), radius, (255, (seed * 13) % 256, 40), -1)
            cv2.rectangle(
                frame,
                (seed % max(1, width // 4), seed % max(1, height // 4)),
                (min(width - 1, width // 2), min(height - 1, height // 2)),
                (40, 220, (seed * 7) % 256),
                max(1, min(width, height) // 180),
            )
            writer.write(frame)
    finally:
        writer.release()

    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:video/mp4;base64,{encoded}"


def _build_videos(args: argparse.Namespace, count: int) -> list[str]:
    videos: list[str] = []
    with tempfile.TemporaryDirectory(prefix="vllm_omni_frontend_bench_") as directory:
        root = Path(directory)
        for index in range(count):
            videos.append(
                _make_video_data_uri(
                    root / f"video-{index}.mp4",
                    seed=args.seed + index,
                    width=args.width,
                    height=args.height,
                    frames=args.frames,
                    fps=args.fps,
                )
            )
    return videos


def _request_payload(args: argparse.Namespace, video: str, index: int) -> dict[str, object]:
    return {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video_url",
                        "video_url": {"url": video, "num_frames": args.frames},
                    },
                    {
                        "type": "text",
                        "text": f"Frontend benchmark request {args.seed + index}: describe the video briefly.",
                    },
                ],
            }
        ],
        "modalities": ["text"],
        "max_tokens": args.max_tokens,
        "temperature": 0,
    }


async def _request(
    client: httpx.AsyncClient,
    args: argparse.Namespace,
    video: str,
    index: int,
) -> RequestResult:
    started = time.perf_counter()
    try:
        response = await client.post(
            args.url,
            json=_request_payload(args, video, index),
        )
    except httpx.HTTPError as exc:
        return RequestResult(
            status=None,
            elapsed_s=time.perf_counter() - started,
            error=repr(exc),
        )

    error = None
    if response.status_code != 200:
        try:
            body = response.json()
            error = str(body.get("error", body) if isinstance(body, dict) else body)
        except ValueError:
            error = response.text[:500]
    return RequestResult(
        status=response.status_code,
        elapsed_s=time.perf_counter() - started,
        error=error,
    )


async def _run(args: argparse.Namespace) -> dict[str, object]:
    request_count = 1 + args.rounds * sum(args.concurrency)
    videos = await asyncio.to_thread(_build_videos, args, request_count)
    max_concurrency = max(args.concurrency)
    timeout = httpx.Timeout(args.request_timeout, connect=args.connect_timeout)
    limits = httpx.Limits(
        max_connections=max_concurrency,
        max_keepalive_connections=args.max_keepalive_connections,
    )

    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        warmup = await _request(client, args, videos[0], 0)
        results: list[dict[str, object]] = []
        next_index = 1
        for concurrency in args.concurrency:
            rounds: list[dict[str, object]] = []
            for round_id in range(args.rounds):
                indices = range(next_index, next_index + concurrency)
                next_index += concurrency
                started = time.perf_counter()
                responses = await asyncio.gather(*(_request(client, args, videos[index], index) for index in indices))
                wall_s = time.perf_counter() - started
                latencies = [response.elapsed_s for response in responses]
                rounds.append(
                    {
                        "round": round_id,
                        "wall_s": wall_s,
                        "completed": sum(response.status == 200 for response in responses),
                        "latency_avg_s": sum(latencies) / len(latencies),
                        "latency_p50_s": float(np.percentile(latencies, 50)),
                        "latency_p95_s": float(np.percentile(latencies, 95)),
                        "latency_max_s": max(latencies),
                        "errors": [response.error for response in responses if response.status != 200],
                    }
                )

            completed = sum(int(round_result["completed"]) for round_result in rounds)
            total_wall_s = sum(float(round_result["wall_s"]) for round_result in rounds)
            results.append(
                {
                    "concurrency": concurrency,
                    "requests": concurrency * args.rounds,
                    "completed": completed,
                    "wall_s": total_wall_s,
                    "throughput_req_s": completed / total_wall_s,
                    "rounds": rounds,
                }
            )

    return {
        "url": args.url,
        "model": args.model,
        "workload": {
            "width": args.width,
            "height": args.height,
            "frames": args.frames,
            "fps": args.fps,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
        },
        "warmup": asdict(warmup),
        "results": results,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8091")
    parser.add_argument("--model", required=True)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8, 64])
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--connect-timeout", type=float, default=10.0)
    parser.add_argument(
        "--max-keepalive-connections",
        type=int,
        default=0,
        help=(
            "Connections retained between rounds. The default disables reuse "
            "so server keep-alive expiry cannot surface as a stale-connection ReadError."
        ),
    )
    args = parser.parse_args()
    if args.rounds < 1 or any(concurrency < 1 for concurrency in args.concurrency):
        parser.error("--rounds and every --concurrency value must be positive")
    if min(args.width, args.height, args.frames, args.fps, args.max_tokens) <= 0:
        parser.error("video dimensions, frames, fps, and max tokens must be positive")
    if args.max_keepalive_connections < 0:
        parser.error("--max-keepalive-connections must be non-negative")
    args.url = f"{args.base_url.rstrip('/')}/v1/chat/completions"
    return args


def main() -> None:
    args = _parse_args()
    print(json.dumps(asyncio.run(_run(args)), ensure_ascii=False))


if __name__ == "__main__":
    main()
