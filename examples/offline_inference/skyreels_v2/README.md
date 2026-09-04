# SkyReels V2 T2V (vLLM-Omni)

Text-to-video example for Diffusers SkyReels V2 checkpoints.

## Models

- `Skywork/SkyReels-V2-T2V-14B-540P-Diffusers` (default in script: 544x960, 97 frames)
- `Skywork/SkyReels-V2-T2V-14B-720P-Diffusers` (use `--height 720 --width 1280`)

## Run

```bash
python examples/offline_inference/skyreels_v2/text_to_video.py \
  --model Skywork/SkyReels-V2-T2V-14B-540P-Diffusers \
  --prompt "A cat and a dog baking a cake together in a kitchen."
```

Defaults match Diffusers docs: `flow_shift=8.0`, `guidance_scale=6.0`, `num_inference_steps=50`.

You can also use the shared helper:

```bash
python examples/offline_inference/text_to_video/text_to_video.py \
  --model Skywork/SkyReels-V2-T2V-14B-540P-Diffusers
```
