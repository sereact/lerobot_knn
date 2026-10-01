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

CAMS = ("observation.images.image", "observation.images.image2")
CELLS, CH = 16, 256


class CellKey(torch.nn.Module):
    def __init__(self, dim_out: int = 128):
        super().__init__()
        self.w = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.eye(CH, dim_out) + 0.01 * torch.randn(CH, dim_out)) for _ in CAMS]
        )
        self.log_tau = torch.nn.Parameter(torch.tensor(-2.3))

    def forward(self, xs: list[torch.Tensor]) -> list[torch.Tensor]:
        return [
            F.normalize((x.view(len(x), CELLS, CH) @ w).flatten(1), dim=-1)
            for x, w in zip(xs, self.w, strict=True)
        ]


def dist(qa: list[torch.Tensor], ca: list[torch.Tensor]) -> torch.Tensor:
    return sum(2 - 2 * q @ c.T for q, c in zip(qa, ca, strict=True)) / len(qa)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--out", default="outputs/vla/learned_key_resnet.pt")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    dev = args.device
    torch.manual_seed(0)
    f = load_frame_index("lerobot/libero")
    split = json.loads(Path("outputs/vla/split.json").read_text())
    train_eps, held_eps = np.array(split["train"]), np.array(split["heldout"])
    mem = np.nonzero(f.frame_index % 2 == 0)[0]
    tr = mem[np.isin(f.episode_index[mem], train_eps)]
    xs = [torch.from_numpy(np.asarray(load_features(DEFAULT_CACHE_DIR, "lerobot/libero", "resnet18_l3", c, "grid4")[mem]))
          for c in CAMS]  # fmt: skip
    pos = np.full(len(f.frame_index), -1)
    pos[mem] = np.arange(len(mem))
    stats = []
    for i, x in enumerate(xs):
        xt = x[torch.from_numpy(pos[tr])].float()
        mean, std = xt.mean(0), xt.std(0).clamp_min(1e-3)
        stats.append((mean, std))
        xs[i] = ((x.float() - mean) / std).half().to(dev)
    act, end = torch.from_numpy(f.action), torch.from_numpy(f.episode_end)
    chunk = act[torch.minimum(torch.from_numpy(mem)[:, None] + torch.arange(10), end[mem][:, None])][..., :6]
    chunk = chunk.reshape(len(mem), -1).to(dev)
    task = torch.from_numpy(f.task_index[mem]).to(dev)
    ep = torch.from_numpy(f.episode_index[mem]).to(dev)
    is_train = torch.from_numpy(np.isin(f.episode_index[mem], train_eps)).to(dev)
    is_held_q = torch.from_numpy(np.isin(f.episode_index[mem], held_eps) & (f.frame_index[mem] % 10 == 4)).to(
        dev
    )
    tasks = sorted(np.unique(f.task_index).tolist())

    model = CellKey(args.dim).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)

    @torch.no_grad()
    def evaluate(use_model: bool) -> dict:
        lac, sse, sst = [], 0.0, 0.0
        for t in tasks:
            c = torch.nonzero((task == t) & is_train).squeeze(1)
            q = torch.nonzero((task == t) & is_held_q).squeeze(1)
            if len(q) == 0:
                continue
            if use_model:
                kc, kq = model([x[c].float() for x in xs]), model([x[q].float() for x in xs])
            else:  # the baseline key: per-cell unit norm of the raw ResNet grid (what the policy uses)

                def norm(z: torch.Tensor) -> torch.Tensor:
                    return F.normalize(z.float().view(len(z), CELLS, CH), dim=-1).flatten(1) / CELLS**0.5

                kc, kq = [norm(x[c]) for x in xs], [norm(x[q]) for x in xs]
            d = dist(kq, kc)
            dd, idx = d.topk(8, largest=False)
            da = torch.cdist(chunk[q], chunk[c]).pow(2) / chunk.shape[1]
            lac.append(1 - da.gather(1, idx).mean(1) / da.mean(1))
            w = torch.softmax(-dd / (0.1 * dd.mean(1, keepdim=True)), 1)
            pred = (w[..., None] * chunk[c][idx]).sum(1)
            sse += float(((pred - chunk[q]) ** 2).sum())
            sst += float(((chunk[q] - chunk[q].mean(0)) ** 2).sum())
        return {"lac@8": float(torch.cat(lac).mean()), "r2": 1 - sse / sst}

    before = evaluate(False)
    print("raw ResNet key (held-out demos):", before, flush=True)
    for step in range(args.steps):
        t = tasks[step % len(tasks)]
        pool = torch.nonzero((task == t) & is_train).squeeze(1)
        q = pool[torch.randperm(len(pool), device=dev)[:256]]
        c = pool[torch.randperm(len(pool), device=dev)[:2048]]
        kq, kc = model([x[q].float() for x in xs]), model([x[c].float() for x in xs])
        logits = -dist(kq, kc) / model.log_tau.exp()
        logits = logits.masked_fill(
            ep[q][:, None] == ep[c][None, :], float("-inf")
        )  # neighbours from other demos only
        pred = torch.softmax(logits, 1) @ chunk[c]
        loss = F.mse_loss(pred, chunk[q])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 500 == 0 or step == args.steps - 1:
            print(f"step {step} loss {loss.item():.5f} tau {model.log_tau.exp().item():.4f} held-out {evaluate(True)}",
                  flush=True)  # fmt: skip
    after = evaluate(True)
    # dense block-diagonal projection per camera: key = normalize(standardised raw @ proj), as the kNN policy expects
    state = {}
    for i, cam in enumerate(CAMS):
        w = model.w[i].detach().float().cpu()
        proj = torch.block_diag(*[w] * CELLS)
        state[cam] = {"mean": stats[i][0], "std": stats[i][1], "proj": proj}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"cameras": state, "before": before, "after": after}, args.out)
    print("saved", args.out, "before", before, "after", after)


if __name__ == "__main__":
    main()
