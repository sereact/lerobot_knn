"""Collect every number the poster shows into poster/data/poster_data.json (CPU only).

Sources (all produced by lerobot_policy_knn.experiments.*; missing ones are simply left out and the matching
poster panels render as "pending"):

* outputs/knn/report/data.json   baselines, official kNN / DiT evaluations, ablations (report_data.py)
* outputs/knn/final.json         robustness re-runs of the official configuration
* outputs/knn/compare.jsonl      k sweep, and closed-loop kNN runs with VLA embedders (tags vla_*)
* outputs/knn/memorization.json  diffusion transformer vs kNN, per chunk
* outputs/vla/results.jsonl      closed-loop success of the VLAs themselves
* outputs/vla/head_vs_knn_*.json does the VLA head behave like kNN in its own embedding space?
* outputs/vla/neighbor_metric*.json  do close embeddings have close actions (neighbor_metric.py)
* outputs/vla/learned_key_resnet.pt  the ResNet key optimised for that (learn_key.py): before / after metrics
* outputs/libero_plus/results.jsonl  LIBERO-trained policies on the perturbed LIBERO-plus variants
"""

import json
from pathlib import Path

import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "outputs").is_dir())  # the root holding outputs/
KNN = ROOT / "outputs/knn"
VLA = ROOT / "outputs/vla"
OUT = Path(__file__).resolve().parents[1] / "data" / "poster_data.json"
SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]

ABLATION_LABELS = {
    "no_language": "prompt ignored",
    "encoder_pixels": "raw 16×16 pixels",
    "demos_10": "10% of demos",
    "k1": "k = 1 (replay nearest demo)",
    "encoder_dinov2": "DINOv2-S instead of ResNet-18",
    "no_wrist": "agent-view camera only",
    "demos_25": "25% of demos",
    "no_agentview": "wrist camera only",
    "demos_50": "50% of demos",
    "no_mode": "plain weighted mean",
    "no_state": "no proprioceptive state",
    "no_time": "no time-step feature",
    "hard_filter": "hard task filter",
    "no_history": "no history matching",
    "no_ensemble": "no temporal ensembling",
}


def read_jsonl(path: Path) -> list[dict]:
    return (
        [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    )


def real_novelty_motion(horizon: int = 8, per_task: int = 64, seed: int = 0) -> list[float]:
    """Nearest-other-demonstration distance of real demonstration chunks (motion dims, leave-one-demo-out)."""
    f = np.load(KNN / "feature_cache/lerobot__libero/frame_index.npz")
    action, ep_end, task, episode = f["action"], f["episode_end"], f["task_index"], f["episode_index"]
    idx = np.minimum(np.arange(len(action))[:, None] + np.arange(horizon), ep_end[:, None])
    rng = np.random.default_rng(seed)
    out = []
    for t in np.unique(task):
        rows = np.nonzero(task == t)[0]
        cands = action[idx[rows]][..., :6].reshape(len(rows), -1).astype(np.float32)
        for r in rng.choice(rows, min(per_task, len(rows)), replace=False):
            q = action[idx[r]][..., :6].reshape(-1).astype(np.float32)
            d = ((cands - q) ** 2).mean(1)
            d[episode[rows] == episode[r]] = np.inf
            out.append(float(d.min()))
    return out


STUDY_VLAS = {  # tag:name of each family's closed-loop run on the shared 400 episodes
    "pi05_n10:pi05_libero",
    "smolvla_n10:smolvla_libero",
    "hub_other_n10:xvla@lerobot/xvla-libero",
    "hub_other_n10:pi0_fast@lerobot/pi0fast-libero",
    "hub_other_n10:pi0_libero",
    "vla_jepa_n7:vla_jepa@lerobot/VLA-JEPA-LIBERO",
    "evo1:evo1_libero",
    "molmoact2:molmoact2_libero",
    "groot_n17:groot_libero",  # merged from its per-suite checkpoints (PER_SUITE)
}
PER_SUITE = {"groot_libero": {s: f"groot_{s}" for s in SUITES}}  # merged name -> its checkpoint for each suite
OVERLAP_LOOKUPS = {"plain": "plain_resnet18_l3", "full": "00_base"}  # the lookup references (config stems)


def merged_name(name: str, suite: str) -> str | None:
    """The merged entry an episode of a per-suite checkpoint belongs to (None: another suite's episode), else name."""
    for merged, by_suite in PER_SUITE.items():
        if name in by_suite.values():
            return merged if by_suite[suite] == name else None
    return name


def merge_per_suite(results: list[dict], episodes: list[dict]) -> list[dict]:
    """Add one results row per PER_SUITE model whose four checkpoints all ran, computed from their episodes."""
    out = list(results)
    for merged, by_suite in PER_SUITE.items():
        for tag in {r["tag"] for r in episodes if r["name"] in by_suite.values()}:
            eps = [r for r in episodes if r["tag"] == tag and merged_name(r["name"], r["suite"]) == merged]
            if {r["suite"] for r in eps} != set(SUITES):
                continue
            first = next(r for r in results if r.get("tag") == tag and r.get("name") in by_suite.values())
            row = {"tag": tag, "name": merged, "init_states": first.get("init_states"),
                   "n_action_steps": first.get("n_action_steps"), "success": float(np.mean([r["success"] for r in eps]))}  # fmt: skip
            row.update({s: float(np.mean([r["success"] for r in eps if r["suite"] == s])) for s in SUITES})
            out.append(row)
    return out


def merge_plus_per_suite(episodes: list[dict]) -> dict[str, dict]:
    """LIBERO-plus results of each PER_SUITE model whose four checkpoints all ran (each on its own suite's variants)."""
    out = {}
    for merged, by_suite in PER_SUITE.items():
        eps = [r for r in episodes if r.get("tag") == "libero_plus" and r["policy"].startswith("vla:")
               and merged_name(r["policy"][4:], r["suite"]) == merged and r["policy"][4:] != merged]  # fmt: skip
        if {r["suite"] for r in eps} != set(SUITES):
            continue
        entry: dict = {"n_episodes": len(eps), "all": float(np.mean([r["success"] for r in eps]))}
        for field, prefix in (("category", "category/"), ("level", "level/"), ("suite", "suite/")):
            for value in sorted({r[field] for r in eps}, key=str):
                entry[f"{prefix}{value}"] = float(np.mean([r["success"] for r in eps if r[field] == value]))
        out[f"vla:{merged}"] = entry
    return out


def episode_overlap(vla_runs: list[dict]) -> dict:
    """Per VLA run, on the same 400 episodes (initial states 20-29): which episodes it and a training-free lookup
    solve; and per task, the success of each. Keys of ``episodes``: "suite/task_id/init_state"."""
    lookups: dict[str, dict[str, bool]] = {k: {} for k in OVERLAP_LOOKUPS}
    stems = {v: k for k, v in OVERLAP_LOOKUPS.items()}
    for r in read_jsonl(KNN / "compare_episodes.jsonl"):
        key = stems.get(Path(r["config_file"]).stem)
        if key and 20 <= r["init_state"] < 30:
            lookups[key].setdefault(f"{r['suite']}/{r['task_id']}/{r['init_state']}", r["success"])
    vlas: dict[str, dict[str, bool]] = {}
    for r in read_jsonl(VLA / "results_episodes.jsonl"):
        if (name := merged_name(r["name"], r["suite"])) is not None:
            vlas.setdefault(f"{r['tag']}:{name}", {}).setdefault(
                f"{r['suite']}/{r['task_id']}/{r['init_state']}", r["success"]
            )
    runs = {f"{r['tag']}:{r['name']}" for r in vla_runs}
    out: dict = {"lookups": {}, "vlas": {}}
    eps = sorted(lookups["plain"])
    for name, eps_ok in lookups.items():
        out["lookups"][name] = {"success": float(np.mean([eps_ok[e] for e in eps])) if eps else None,
                                "per_task": per_task(eps_ok, eps)}  # fmt: skip
    for run, eps_ok in vlas.items():
        if run not in runs or len(eps_ok) < len(eps) or not eps:
            continue
        entry = {"per_task": per_task(eps_ok, eps)}
        for name, ref in lookups.items():
            if len(ref) < len(eps):
                continue
            a = np.array([ref[e] for e in eps])
            b = np.array([eps_ok[e] for e in eps])
            entry[name] = {"both": int((a & b).sum()), "vla_only": int((b & ~a).sum()),
                           "lookup_only": int((a & ~b).sum()), "neither": int((~a & ~b).sum())}  # fmt: skip
        out["vlas"][run] = entry
    return out


def per_task(eps_ok: dict[str, bool], eps: list[str]) -> dict[str, float]:
    tasks: dict[str, list[bool]] = {}
    for e in eps:
        tasks.setdefault(e.rsplit("/", 1)[0], []).append(eps_ok[e])
    return {t: float(np.mean(v)) for t, v in sorted(tasks.items())}


def main() -> None:
    data: dict = {}
    report = json.loads((KNN / "report/data.json").read_text())
    data["baselines"] = [
        {"name": b["name"], "suites": b["suites"], "avg": float(np.mean(b["suites"])), "source": b["source"]}
        for b in report["baselines"]
    ]
    final = json.loads((KNN / "final.json").read_text())
    data["knn_official"] = {
        "avg": final["average"],
        "suites": [final["suites"][s]["pc"] for s in SUITES],
        "ci": [final["suites"][s]["ci"] for s in SUITES],
        "n": final["n_episodes"],
        "robustness": final.get("robustness", []),
    }
    dit = next((o for o in report["official"] if o["name"] == "dit_60k_all50"), None)
    if dit:
        data["dit_ours"] = {"avg": dit["overall"], "suites": [dit["groups"][s]["pc"] for s in SUITES]}

    ablations, base, base_suites = {}, [], []
    for r in report["compare"]:
        if r.get("tag") != "ablation":
            continue
        key = Path(r["config_file"]).stem
        if key == "00_base":
            base.append(100 * r["success"])
            base_suites.append([100 * r[s] for s in SUITES])
        elif key in ABLATION_LABELS:
            ablations[key] = {"label": ABLATION_LABELS[key], "success": 100 * r["success"],
                              "suites": [100 * r[s] for s in SUITES]}  # fmt: skip
    data["ablation_base"] = float(np.mean(base))
    data["ablation_base_suites"] = np.mean(base_suites, axis=0).tolist()
    data["ablations"] = ablations
    ksweep = {1: ablations["k1"]["success"]} if "k1" in ablations else {}
    for r in read_jsonl(KNN / "compare.jsonl"):
        if r.get("tag") == "k_sweep":
            ksweep[int(Path(r["config_file"]).stem[1:])] = 100 * r["success"]
    ksweep[13] = data["ablation_base"]  # the final configuration uses k = 13
    data["k_sweep"] = dict(sorted(ksweep.items()))
    data["demo_scaling"] = {
        pct: ablations[f"demos_{pct}"]["success"] for pct in (10, 25, 50) if f"demos_{pct}" in ablations
    }
    data["demo_scaling"][100] = data["ablation_base"]

    mem = json.loads((KNN / "memorization.json").read_text())
    recs = mem["records"]
    data["dit_memorization"] = {
        "n_chunks": len(recs),
        "dit_to_train": [r["nn_train_motion"] for r in recs],
        "dit_to_knn": [r["knn_policy_motion"] for r in recs],
        "knn_spread": [r["knn_neighbour_spread_motion"] for r in recs],
        "human_to_other_demos": real_novelty_motion(),
        "gripper_agreement": mem["summary"]["gripper_agreement"],
    }

    diag = KNN / "diagnose_libero_10.json"
    if diag.exists():
        eps = json.loads(diag.read_text())
        data["failure_diag"] = [{"success": e["success"], "peak_distance": max(e["distance"]), "task": e["task"]}
                                for e in eps]  # fmt: skip

    # --- VLA experiments (optional until they have run) ---
    # the study's VLAs: one working LIBERO fine-tune per family (tag:name of its closed-loop run)
    all_runs = merge_per_suite(read_jsonl(VLA / "results.jsonl"), read_jsonl(VLA / "results_episodes.jsonl"))
    vla_runs = [r for r in all_runs if f"{r.get('tag')}:{r.get('name')}" in STUDY_VLAS]
    if vla_runs:
        data["vla_success"] = {
            f"{r['tag']}:{r['name']}": {
                k: r[k]
                for k in ("name", "tag", "init_states", "n_action_steps", "success", *SUITES)
                if k in r
            }
            for r in vla_runs
            if not str(r.get("tag", "")).startswith("dose")
        }
    learnt = [r for r in read_jsonl(KNN / "compare.jsonl") if r.get("tag") == "learned_key"]
    if learnt:  # the full method with the learnt ResNet key
        data["knn_learned_key"] = [
            {"success": r["success"], "suites": [r.get(s) for s in SUITES]} for r in learnt
        ]
    knn_vla = [r for r in read_jsonl(KNN / "compare.jsonl") if str(r.get("tag", "")).startswith("vla")]
    if knn_vla:
        data["knn_with_vla_embedders"] = {
            Path(r["config_file"]).stem: {
                "tag": r["tag"],
                "init_states": r["init_states"],
                "success": r["success"],
                "suites": [r.get(s) for s in SUITES],
            }  # fmt: skip
            for r in knn_vla
        }
    # one entry per analysed VLA; extra-space files (run with --no_references) merge into it
    heads: dict[str, dict] = {}
    for path in sorted(VLA.glob("head_vs_knn_*.json")):
        res = json.loads(path.read_text())
        entry = heads.setdefault(res["vla"], {"spaces": {}})
        entry["spaces"].update(res["spaces"])
        for key in ("horizon", "n_chunks", "vla_success", "noise_floor", "nearest_training_chunk"):
            if key in res:
                entry[key] = res[key]
    if heads:
        data["head_vs_knn"] = heads
    quality = [
        r for r in read_jsonl(VLA / "embed_quality.jsonl") if str(r.get("tag", "")).endswith("_ladder_loeo")
    ]
    if quality:
        data["embed_quality"] = {}
        for r in quality:
            data["embed_quality"].setdefault(r["space"], {})[str(r["k"])] = {
                "r2": r["r2"],
                "gripper": r["gripper"],
            }
    for name in ("neighbor_metric", "neighbor_metric_heldout"):
        if (VLA / f"{name}.json").exists():
            data[name] = json.loads((VLA / f"{name}.json").read_text())
    if (VLA / "learned_key_resnet.pt").exists():
        import torch

        learnt = torch.load(VLA / "learned_key_resnet.pt", map_location="cpu", weights_only=True)
        data["learned_key"] = {"before": learnt["before"], "after": learnt["after"]}
    rows = read_jsonl(ROOT / "outputs/libero_plus/results.jsonl")
    # control: the language variants (LIBERO's original scenes) with their original instruction, in the LIBERO-plus env
    control = {r["policy"]: r["all"] for r in rows if r.get("tag") == "libero_plus_original_instruction"}
    if control:
        data["libero_plus_control"] = control
    plus = [r for r in rows if r.get("tag") == "libero_plus"]
    if plus:
        data["libero_plus"] = {r["policy"]: {k: v for k, v in r.items() if k not in ("tag", "policy", "time")}
                               for r in plus}  # fmt: skip
        data["libero_plus"].update(merge_plus_per_suite(read_jsonl(ROOT / "outputs/libero_plus/results_episodes.jsonl")))

    data["episode_overlap"] = episode_overlap(vla_runs)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, indent=1))
    print(f"wrote {OUT} ({', '.join(data)})")


if __name__ == "__main__":
    main()
