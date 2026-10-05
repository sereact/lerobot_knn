"""Leave-one-episode-out check of the shoe-box kNN on dataset frames: absolute vs relative (anchored) retrieval.

For sampled frames of every episode, retrieve with that episode excluded (prompted with its own task), and compare
the predicted chunk with the recorded one: active-arm / idle-arm TCP position error and gripper error at a few
horizons. "relative" = every neighbour's chunk minus that neighbour's own state, plus the query state (LeRobot's
RelativeActionsProcessorStep semantics, grippers kept absolute) -- what a relative-action kNN would return.
"""

import json
import sys

import numpy as np
import torch

from lerobot_policy_knn.modeling_knn import KNNPolicy, camera_key_name, mode_seeking_weights

POLICY = "/home/denis-office/git/lerobot/outputs/tri-arm-shoe-box-knn/policy"
SOURCES = "/home/denis-office/git/lerobot/outputs/tri-arm-shoe-box/episode_sources.json"
HORIZONS = (0, 7, 29)
GRIP = (9, 19)
ARM_XYZ = {"left": slice(0, 3), "right": slice(10, 13)}

policy = KNNPolicy.from_pretrained(POLICY).to("cuda").eval()
cfg = policy.config
arm_of = {s["episode_index"]: s["arm"] for s in json.load(open(SOURCES))}
g = torch.Generator().manual_seed(0)
n = policy.memory_size
rows_all = torch.randperm(n, generator=g)[:3000].cuda()
index = policy._index()
names = [camera_key_name(c) for c in cfg.active_cameras] + ["key_state"]
names = [nm for nm in names if hasattr(policy, nm)]
mask = torch.ones(20, device="cuda")
mask[list(GRIP)] = 0.0


def raw_state(rows: torch.Tensor) -> torch.Tensor:
    return policy.key_state[rows].float() * policy.state_std + policy.state_mean


@torch.no_grad()
def run(state_weight: float, k: int, bandwidth: float) -> dict:
    cfg.state_weight, cfg.k, cfg.action_mode_bandwidth = state_weight, k, bandwidth
    policy._retrieval_index.clear()
    index = policy._index()
    errs = {m: {h: {"active": [], "idle": [], "grip": []} for h in HORIZONS} for m in ("absolute", "relative", "hold")}
    for s in range(0, len(rows_all), 500):
        r = rows_all[s : s + 500]
        query = {nm: getattr(policy, nm)[r] for nm in names}
        task = index["key_task"][r] if cfg.task_conditioning != "none" else None
        nb, w, _ = policy.retrieve(query, policy.key_frame[r], task, exclude_episode=policy.key_episode[r])
        truth = policy.gather_chunks(r[:, None])[:, 0]  # (B, H, A)
        chunks = policy.gather_chunks(nb)  # (B, k, H, A)
        rel = chunks - (raw_state(nb)[..., :20] * mask)[:, :, None, :]
        preds = {}
        for mode, c in (("absolute", chunks), ("relative", rel)):
            ww = w
            if bandwidth > 0:
                motion = c[..., [i for i in range(20) if i not in GRIP]].flatten(2)
                ww = mode_seeking_weights(motion, w, bandwidth)
            preds[mode] = (ww[..., None, None] * c).sum(1)
        preds["relative"] = preds["relative"] + (raw_state(r)[:, :20] * mask)[:, None, :]
        preds["hold"] = raw_state(r)[:, None, :20].expand_as(truth)
        eps = policy.key_episode[r].tolist()
        for mode, p in preds.items():
            for h in HORIZONS:
                d = (p[:, h] - truth[:, h]).cpu().numpy()
                for i, ep in enumerate(eps):
                    act = arm_of[ep]
                    idle = "right" if act == "left" else "left"
                    errs[mode][h]["active"].append(np.linalg.norm(d[i, ARM_XYZ[act]]))
                    errs[mode][h]["idle"].append(np.linalg.norm(d[i, ARM_XYZ[idle]]))
                    errs[mode][h]["grip"].append(abs(d[i, GRIP[0] if act == "left" else GRIP[1]]))
    return {m: {h: {k2: float(np.median(v)) for k2, v in e.items()} | {"active_p90": float(np.percentile(e["active"], 90))}
                for h, e in he.items()} for m, he in errs.items()}  # fmt: skip


settings = [(1.0, 8, 0.05)] if len(sys.argv) < 2 else [tuple(map(float, x.split("/"))) for x in sys.argv[1:]]
for sw, k, bw in settings:
    res = run(sw, int(k), bw)
    print(f"state_weight {sw}, k {int(k)}, mode bandwidth {bw}  (median metres; gripper in [0,1] units)")
    for mode, he in res.items():
        print(f"  {mode:9s}", " | ".join(
            f"t+{h}: active {e['active']*100:4.1f} cm (p90 {e['active_p90']*100:4.1f}) idle {e['idle']*100:4.1f} cm grip {e['grip']:.2f}"
            for h, e in he.items()))  # fmt: skip
