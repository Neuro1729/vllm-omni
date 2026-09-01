# API frontend scaling benchmark

`benchmark_multimodal_frontend.py` measures request throughput and latency
while varying client concurrency against an already-running vLLM-Omni server.
It generates a deterministic, unique 1280×720 MP4 input per request so video
decode and multimodal preprocessing remain part of the measured frontend work.

Run the same command once with a single API frontend and once with multiple
frontends. Keep the model, hardware, server options, seed, and benchmark
arguments unchanged between runs.

```bash
MODEL=Qwen/Qwen3-Omni-30B-A3B-Instruct

CUDA_VISIBLE_DEVICES=0,1 vllm serve "$MODEL" --omni \
    --api-server-count 1 --port 8091

python benchmarks/frontend/benchmark_multimodal_frontend.py \
    --model "$MODEL" \
    --concurrency 1 4 8 64 \
    --rounds 2 \
    --seed 1000
```

After stopping the first server, repeat with `--api-server-count 2` and run the
identical benchmark command. The tool performs one warmup request, reports
each measured round separately, and emits a single JSON object containing
completed requests, wall time, throughput, and latency statistics.

The default workload requires OpenCV (`cv2`), NumPy, and HTTPX in the benchmark
environment. Use a local checkpoint path for `MODEL` when network access is
not available. Cross-round HTTP keep-alive reuse is disabled by default so a
server keep-alive timeout cannot turn a stale pooled connection into a client
`ReadError`; use `--max-keepalive-connections` to test persistent connections
explicitly.
