"""Time the two model stages of the play loop, without opening a game window.

    python scripts/bench_inference.py                 # auto-selects cuda / mps / cpu
    python scripts/bench_inference.py --device cpu
    python scripts/bench_inference.py --steps 50

Reports the per-frame cost of the denoiser (next low-res frame) and of the upsampler,
which together set the frame rate `src/play.py` can sustain.
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch
from huggingface_hub import snapshot_download
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import OmegaConf

from agent import Agent
from envs import WorldModelEnv

OmegaConf.register_new_resolver("eval", eval)


def default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def build_env(device: torch.device) -> WorldModelEnv:
    with initialize(version_base="1.3", config_path="../config"):
        cfg = compose(config_name="trainer")
    cfg.agent = OmegaConf.load(ROOT / "config/agent/racing.yaml")
    cfg.env = OmegaConf.load(ROOT / "config/env/racing.yaml")

    agent = Agent(instantiate(cfg.agent, num_actions=cfg.env.num_actions)).to(device).eval()
    agent.load(Path(snapshot_download(repo_id="Enigma-AI/multiverse")) / "agent.pt")

    sl = cfg.agent.denoiser.inner_model.num_steps_conditioning
    if agent.upsampler is not None:
        sl = max(sl, cfg.agent.upsampler.inner_model.num_steps_conditioning)

    return WorldModelEnv(
        agent.denoiser,
        agent.upsampler,
        agent.rew_end_model,
        ROOT / "game/spawn",
        1,
        sl,
        instantiate(cfg.world_model_env, num_batches_to_preload=1),
        return_denoising_trajectory=False,
    )


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None, help="cuda / mps / cpu (default: best available)")
    parser.add_argument("--steps", type=int, default=20, help="Timed frames.")
    parser.add_argument("--warmup", type=int, default=5, help="Untimed frames before measuring.")
    parser.add_argument("--fps", type=int, default=30, help="Frame rate to compare against.")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else default_device()
    print(f"torch {torch.__version__} on {device}")

    env = build_env(device)
    env.reset()
    act = torch.zeros(1, 66, dtype=torch.long, device=device)
    act[0, 0] = 1

    t_denoiser, t_upsampler = [], []
    for i in range(args.warmup + args.steps):
        synchronize(device)
        t0 = time.perf_counter()
        next_obs, _ = env.predict_next_obs()
        synchronize(device)
        t1 = time.perf_counter()
        if env.sampler_upsampling is not None:
            env.upsample_next_obs(next_obs)
            synchronize(device)
        t2 = time.perf_counter()
        env.step(act)
        if i >= args.warmup:
            t_denoiser.append((t1 - t0) * 1e3)
            t_upsampler.append((t2 - t1) * 1e3)

    denoiser = sum(t_denoiser) / len(t_denoiser)
    upsampler = sum(t_upsampler) / len(t_upsampler)
    frame = denoiser + upsampler

    print(f"denoiser  (48x64)    {denoiser:8.2f} ms/frame")
    print(f"upsampler (350x530)  {upsampler:8.2f} ms/frame")
    print(f"total                {frame:8.2f} ms/frame -> {1000 / frame:6.2f} fps "
          f"({args.fps} fps needs {1000 / args.fps:.1f} ms)")


if __name__ == "__main__":
    main()
