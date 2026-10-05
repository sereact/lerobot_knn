"""The patched plugin's aggregate() on held-out frames must reproduce offline_eval's "relative" errors."""

import json

import lerobot_policy_knn
import numpy as np
import torch

from lerobot_policy_knn.modeling_knn import KNNPolicy, camera_key_name

print("plugin from", lerobot_policy_knn.__file__)
policy = KNNPolicy.from_pretrained("/home/denis-office/git/lerobot/outputs/tri-arm-shoe-box-knn/policy_relative")
policy = policy.to("cuda").eval()
cfg = policy.config
print("relative dims", cfg.relative_action_dims, "state_weight", cfg.state_weight, "k", cfg.k)
arm_of = {s["episode_index"]: s["arm"] for s in json.load(open("/home/denis-office/git/lerobot/outputs/tri-arm-shoe-box/episode_sources.json"))}
rows = torch.randperm(policy.memory_size, generator=torch.Generator().manual_seed(0))[:3000].cuda()
index = policy._index()
names = [camera_key_name(c) for c in cfg.active_cameras] + ["key_state"]
errs = {0: [], 7: [], 29: []}
with torch.no_grad():
    for s in range(0, len(rows), 500):
        r = rows[s : s + 500]
        query = {nm: getattr(policy, nm)[r] for nm in names}
        nb, w, _ = policy.retrieve(query, policy.key_frame[r], index["key_task"][r], exclude_episode=policy.key_episode[r])
        state = policy.key_state[r].float() * policy.state_std + policy.state_mean
        pred = policy.aggregate(nb, w, state)
        truth = policy.gather_chunks(r[:, None])[:, 0]
        for h in errs:
            d = (pred[:, h] - truth[:, h]).cpu().numpy()
            for i, ep in enumerate(policy.key_episode[r].tolist()):
                xyz = slice(0, 3) if arm_of[ep] == "left" else slice(10, 13)
                errs[h].append(np.linalg.norm(d[i, xyz]))
print("patched aggregate, active arm median:", {f"t+{h}": f"{np.median(v) * 100:.1f} cm" for h, v in errs.items()})
