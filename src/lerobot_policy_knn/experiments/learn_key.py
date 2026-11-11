"""Learn a lookup key on top of frozen ImageNet ResNet features: make close embeddings have close actions.

Local action consistency (``neighbor_metric``) predicts closed-loop kNN success, so this trains the key to maximise it
directly. A linear map is shared by the 16 grid cells of each camera's ResNet-18 layer-3 features (standardised, 256 -> 128
per cell); the objective is soft-kNN regression, the action chunk of a frame predicted from the softmax-weighted chunks of
its embedding neighbours in *other* demonstrations of the same task. Only the ``train`` episodes of outputs/vla/split.json
are used; the held-out 10% measure the result (LAC@8 and kNN R², memory = train episodes).

The learnt map is saved as ``key_norm="file:<path>"`` buffers (mean, std, dense block-diagonal projection per camera),
so the unchanged kNN policy uses it at test time.

    uv run python -m lerobot_policy_knn.experiments.learn_key --out=outputs/vla/learned_key_resnet.pt
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

from ..features import DEFAULT_CACHE_DIR, load_features, load_frame_index
from ..vla import DATASET_CAMERAS
from .common import MOTION, REPO_ID, SPLIT, VLA_OUT, action_chunks

CELLS, CHANNELS = 16, 256  # ResNet-18 layer 3, pooled to a 4x4 grid
HORIZON = 10


class CellKey(torch.nn.Module):
    """One linear map per camera, shared by its grid cells; the key is the unit-normalised concatenation."""

    def __init__(self, dim_out: int = 128):
        super().__init__()
        self.w = torch.nn.ParameterList(
            [
                torch.nn.Parameter(torch.eye(CHANNELS, dim_out) + 0.01 * torch.randn(CHANNELS, dim_out))
                for _ in DATASET_CAMERAS
            ]
        )
        self.log_tau = torch.nn.Parameter(torch.tensor(-2.3))  # log temperature of the soft kNN

    def forward(self, xs: list[torch.Tensor]) -> list[torch.Tensor]:
        return [
            F.normalize((x.view(len(x), CELLS, CHANNELS) @ w).flatten(1), dim=-1)
            for x, w in zip(xs, self.w, strict=True)
        ]


def raw_key(x: torch.Tensor) -> torch.Tensor:
    """The baseline key: per-cell unit norm of the (standardised) ResNet grid, what the policy uses."""
    return F.normalize(x.float().view(len(x), CELLS, CHANNELS), dim=-1).flatten(1) / CELLS**0.5


def distance(queries: list[torch.Tensor], candidates: list[torch.Tensor]) -> torch.Tensor:
    """(Q, C) mean over cameras of the squared distance between unit-norm keys."""
    return sum(2 - 2 * q @ c.T for q, c in zip(queries, candidates, strict=True)) / len(queries)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--out", default=str(VLA_OUT / "learned_key_resnet.pt"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    dev = args.device
    torch.manual_seed(0)
    frames = load_frame_index(REPO_ID)
    split = json.loads(SPLIT.read_text())
    memory = np.nonzero(frames.frame_index % 2 == 0)[0]
    in_train = np.isin(frames.episode_index[memory], split["train"])
    in_heldout = np.isin(frames.episode_index[memory], split["heldout"])

    # standardised raw features of every memory frame, with the statistics of the training demos
    xs, stats = [], []
    for cam in DATASET_CAMERAS:
        raw = load_features(DEFAULT_CACHE_DIR, REPO_ID, "resnet18_l3", cam, "grid4")
        x = torch.from_numpy(np.asarray(raw[memory]))
        x_train = x[torch.from_numpy(np.nonzero(in_train)[0])].float()
        mean, std = x_train.mean(0), x_train.std(0).clamp_min(1e-3)
        stats.append((mean, std))
        xs.append(((x.float() - mean) / std).half().to(dev))
    motion = torch.from_numpy(action_chunks(frames, memory, HORIZON)[..., MOTION])
    motion = motion.reshape(len(memory), -1).to(dev)
    task = torch.from_numpy(frames.task_index[memory]).to(dev)
    episode = torch.from_numpy(frames.episode_index[memory]).to(dev)
    is_train = torch.from_numpy(in_train).to(dev)
    is_heldout_query = torch.from_numpy(in_heldout & (frames.frame_index[memory] % 10 == 4)).to(dev)
    tasks = sorted(np.unique(frames.task_index).tolist())

    model = CellKey(args.dim).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)

    def keys(rows: torch.Tensor, learnt: bool = True) -> list[torch.Tensor]:
        return model([x[rows].float() for x in xs]) if learnt else [raw_key(x[rows]) for x in xs]

    @torch.no_grad()
    def evaluate(learnt: bool) -> dict:
        """LAC@8 and kNN R² of the held-out demos' queries against the training demos."""
        lac, sse, sst = [], 0.0, 0.0
        for t in tasks:
            cand = torch.nonzero((task == t) & is_train).squeeze(1)
            queries = torch.nonzero((task == t) & is_heldout_query).squeeze(1)
            if len(queries) == 0:
                continue
            nearest, idx = distance(keys(queries, learnt), keys(cand, learnt)).topk(8, largest=False)
            d_a = torch.cdist(motion[queries], motion[cand]).pow(2) / motion.shape[1]
            lac.append(1 - d_a.gather(1, idx).mean(1) / d_a.mean(1))
            weights = torch.softmax(-nearest / (0.1 * nearest.mean(1, keepdim=True)), 1)
            pred = (weights[..., None] * motion[cand][idx]).sum(1)
            sse += float(((pred - motion[queries]) ** 2).sum())
            sst += float(((motion[queries] - motion[queries].mean(0)) ** 2).sum())
        return {"lac@8": float(torch.cat(lac).mean()), "r2": 1 - sse / sst}

    before = evaluate(learnt=False)
    print("raw ResNet key (held-out demos):", before, flush=True)
    for step in range(args.steps):
        pool = torch.nonzero((task == tasks[step % len(tasks)]) & is_train).squeeze(1)
        queries = pool[torch.randperm(len(pool), device=dev)[:256]]
        cand = pool[torch.randperm(len(pool), device=dev)[:2048]]
        logits = -distance(keys(queries), keys(cand)) / model.log_tau.exp()
        # neighbours from other demos only
        logits = logits.masked_fill(episode[queries][:, None] == episode[cand][None, :], float("-inf"))
        loss = F.mse_loss(torch.softmax(logits, 1) @ motion[cand], motion[queries])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 500 == 0 or step == args.steps - 1:
            print(
                f"step {step} loss {loss.item():.5f} tau {model.log_tau.exp().item():.4f} "
                f"held-out {evaluate(learnt=True)}",
                flush=True,
            )
    after = evaluate(learnt=True)
    # dense block-diagonal projection per camera: key = normalize(standardised raw @ proj), as the kNN policy expects
    state = {}
    for cam, w, (mean, std) in zip(DATASET_CAMERAS, model.w, stats, strict=True):
        proj = torch.block_diag(*[w.detach().float().cpu()] * CELLS)
        state[cam] = {"mean": mean, "std": std, "proj": proj}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"cameras": state, "before": before, "after": after}, args.out)
    print("saved", args.out, "before", before, "after", after)


if __name__ == "__main__":
    main()
