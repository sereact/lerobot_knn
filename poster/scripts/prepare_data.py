"""Collect every number the poster and the paper show into poster/data/poster_data.json (CPU only).

Inputs, all under outputs/ (written by the experiments in src/lerobot_policy_knn/experiments):
* knn/eval_official/*/eval_info.json  lerobot-eval runs of the kNN on the official LIBERO protocol
* knn/compare.jsonl, compare_episodes.jsonl  pool runs of kNN configurations (full50: the official protocol;
                                             ablations, k sweep, retry)
* knn/optuna.db                    the hyper-parameter search
* vla/results_episodes.jsonl       closed-loop LIBERO episodes of the VLAs (full50: the official protocol)
* vla/head_vs_knn_*.json           does a lookup predict a VLA's action chunks? (head_vs_knn.py)
* vla/neighbor_metric_heldout.json local action consistency of every key vs its lookup's success (neighbor_metric.py)
* vla/extra_keys.json              the backbone keys chosen by choose_keys.py
* vla/embedding_maps.json          2-D maps of every key space (embedding_maps.py), copied to data/embedding_maps.json
* libero_plus/results*.jsonl, sample.json, head_vs_knn_*.json  the LIBERO-plus runs (libero_plus_eval.py)
"""

import json
from pathlib import Path

import numpy as np

# the repository root: the first parent holding outputs/
ROOT = next(p for p in Path(__file__).resolve().parents if (p / "outputs").is_dir())
KNN = ROOT / "outputs/knn"
VLA = ROOT / "outputs/vla"
PLUS = ROOT / "outputs/libero_plus"
OUT = Path(__file__).resolve().parents[1] / "data" / "poster_data.json"
MAPS_OUT = OUT.with_name("embedding_maps.json")  # the 2-D maps, kept compact in their own file

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
# a VLA released as one checkpoint per suite: merged name -> suite -> checkpoint
PER_SUITE = {"groot_libero": {s: f"groot_{s}" for s in SUITES}}
OFFICIAL_EVALS = {  # lerobot-eval runs (50 episodes per task, episode j = initial state j): policy id -> run dir
    "knn:00_base": "knn_v2_39_all50",
    "knn:full_resnet18_l3_learned": "knn_full_learned_all50",
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def episode_key(r: dict, init_state: int | None = None) -> str:
    return f"{r['suite']}/{r['task_id']}/{r['init_state'] if init_state is None else init_state}"


def merged_name(name: str, suite: str) -> str | None:
    """The merged entry an episode of a per-suite checkpoint belongs to (None: another suite's episode), else name."""
    for merged, by_suite in PER_SUITE.items():
        if name in by_suite.values():
            return merged if by_suite[suite] == name else None
    return name


def libero_full() -> dict:
    """Every policy's per-episode LIBERO success on the official protocol (initial states 0-49, all 40 tasks):
    lerobot-eval runs of the kNN, and the full50 pool runs of the VLAs ("vla:<name>") and lookups
    ("knn:<config stem>"). Only complete runs (2,000 episodes) count."""
    eps: dict[str, dict[str, bool]] = {}
    for pid, run in OFFICIAL_EVALS.items():
        for t in json.loads((KNN / "eval_official" / run / "eval_info.json").read_text())["per_task"]:
            for j, ok in enumerate(t["metrics"]["successes"]):
                eps.setdefault(pid, {})[f"{t['task_group']}/{t['task_id']}/{j}"] = bool(ok)
    for r in read_jsonl(VLA / "results_episodes.jsonl"):
        if r.get("tag") == "full50" and (name := merged_name(r["name"], r["suite"])) is not None:
            eps.setdefault(f"vla:{name}", {}).setdefault(episode_key(r), r["success"])
    for r in read_jsonl(KNN / "compare_episodes.jsonl"):
        if r.get("tag") == "full50":
            eps.setdefault(f"knn:{Path(r['config_file']).stem}", {}).setdefault(episode_key(r), r["success"])
    out = {}
    for pid, e in eps.items():
        if len(e) < 2000:
            continue
        suites = [100 * float(np.mean([v for k, v in e.items() if k.startswith(s + "/")])) for s in SUITES]
        out[pid] = {"success": 100 * float(np.mean(list(e.values()))), "suites": suites, "episodes": e}
    return out


def libero_plus() -> tuple[dict, dict, dict]:
    """LIBERO-plus success of every policy (overall, per category / level / suite), the control runs (the language
    variants' scenes with their original instruction), and every policy's per-episode success."""
    rows = read_jsonl(PLUS / "results.jsonl")
    results = {
        r["policy"]: {k: v for k, v in r.items() if k not in ("tag", "policy", "time")}
        for r in rows
        if r.get("tag") == "libero_plus"
    }
    control = {r["policy"]: r["all"] for r in rows if r.get("tag") == "libero_plus_original_instruction"}
    episodes = [r for r in read_jsonl(PLUS / "results_episodes.jsonl") if r.get("tag") == "libero_plus"]
    for merged, by_suite in PER_SUITE.items():  # each per-suite checkpoint on its own suite's variants
        eps = [
            r
            for r in episodes
            if r["policy"].startswith("vla:")
            and r["policy"][4:] in by_suite.values()
            and merged_name(r["policy"][4:], r["suite"]) == merged
        ]
        entry: dict = {"n_episodes": len(eps), "all": float(np.mean([r["success"] for r in eps]))}
        for field in ("category", "level", "suite"):
            for value in sorted({r[field] for r in eps}, key=str):
                entry[f"{field}/{value}"] = float(np.mean([r["success"] for r in eps if r[field] == value]))
        results[f"vla:{merged}"] = entry
    per_episode: dict[str, dict[str, bool]] = {}
    for r in episodes:
        pol = r["policy"]
        if pol.startswith("vla:"):
            if (name := merged_name(pol[4:], r["suite"])) is None:
                continue
            pol = f"vla:{name}"
        per_episode.setdefault(pol, {}).setdefault(episode_key(r), r["success"])
    return results, control, per_episode


def plus_variant_task() -> dict[str, str]:
    """LIBERO-plus variant -> the LIBERO task it perturbs (variant names extend the task name, e.g. "..._tb_22")."""
    tasks = sorted({r["task"] for r in read_jsonl(VLA / "results_episodes.jsonl")})
    out = {}
    for v in json.loads((PLUS / "sample.json").read_text()):
        flat = v["name"].replace("_", " ")
        hits = [t for t in tasks if t == v["name"] or t in flat]
        if hits:
            out[f"{v['suite']}/{v['task_id']}"] = max(hits, key=len)
    return out


def optuna_summary(study: str = "knn_v2") -> dict:
    """The trials of the hyper-parameter search behind the final configuration, and fANOVA importances."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    storage = f"sqlite:///{KNN / 'optuna.db'}"

    def complete(name: str) -> list:
        trials = optuna.load_study(study_name=name, storage=storage).trials
        return [t for t in trials if t.state.name == "COMPLETE"]

    imp = optuna.importance.get_param_importances(
        optuna.load_study(study_name=study, storage=storage),
        evaluator=optuna.importance.FanovaImportanceEvaluator(seed=0),
    )
    return {
        "study": study,
        "n_trials": len(complete(study)),
        "n_trials_all_studies": sum(
            len(complete(s.study_name)) for s in optuna.get_all_study_summaries(storage)
        ),
        "importance": {k: float(v) for k, v in imp.items()},
        "trials": [{"value": t.value, "params": t.params} for t in complete(study)],
    }


def head_vs_knn() -> tuple[dict, dict]:
    """Per VLA, how well a lookup predicts its action chunks: clean LIBERO rollouts (every key space, references)
    and LIBERO-plus rollouts (R² at k = 8 per key space)."""
    clean: dict[str, dict] = {}
    # several files may hold one VLA (extra key spaces, run with --no_references): they merge into one entry
    for path in sorted(VLA.glob("head_vs_knn_*.json")):
        res = json.loads(path.read_text())
        entry = clean.setdefault(res["vla"], {"spaces": {}})
        entry["spaces"].update(res["spaces"])
        if "nearest_training_chunk" in res:
            entry["nearest_training_chunk"] = res["nearest_training_chunk"]
    plus = {}
    for path in sorted(PLUS.glob("head_vs_knn_*.json")):
        res = json.loads(path.read_text())
        plus[res["vla"]] = {
            "vla_success": res.get("vla_success"),
            "r2_k8": {sp: v["8"]["vla_r2"] for sp, v in res["spaces"].items() if "8" in v},
        }
    return clean, plus


def main() -> None:
    compare = read_jsonl(KNN / "compare.jsonl")
    plus, control, plus_episodes = libero_plus()
    heads, heads_plus = head_vs_knn()
    data = {
        "libero_full": libero_full(),
        "libero_plus": plus,
        "libero_plus_control": control,
        "plus_episodes": plus_episodes,
        "plus_variant_task": plus_variant_task(),
        # ablations and k sweep of the full method on the official protocol: config stem -> success (%)
        "ablations_full": {
            Path(r["config_file"]).stem: {
                "success": 100 * r["success"],
                "suites": [100 * r[s] for s in SUITES],
            }
            for r in compare
            if r.get("tag") in ("full50_ablation", "full50_ksweep") and r.get("init_states") == "0:50"
        },
        "retry": {
            Path(r["config_file"]).stem: r["success"] for r in compare if r.get("tag") == "retry_long2"
        },
        "optuna": optuna_summary(),
        "head_vs_knn": heads,
        "head_vs_knn_plus": heads_plus,
        "neighbor_metric": json.loads((VLA / "neighbor_metric_heldout.json").read_text()),
        "extra_keys": json.loads((VLA / "extra_keys.json").read_text())["keys"],
    }
    maps = json.loads((VLA / "embedding_maps.json").read_text())
    embedding_maps = {  # coordinates at the resolution a figure panel needs
        "suite": maps["suite"],
        "progress": [round(v, 2) for v in maps["progress"]],
        "spaces": {
            name: {
                **{k: v for k, v in sp.items() if k not in ("pca", "tsne")},
                **{m: [[round(x, 2), round(y, 2)] for x, y in sp[m]] for m in ("pca", "tsne")},
            }
            for name, sp in maps["spaces"].items()
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, indent=1))
    MAPS_OUT.write_text(json.dumps(embedding_maps, separators=(",", ":")))
    print(f"wrote {OUT} ({', '.join(data)}) and {MAPS_OUT}")


if __name__ == "__main__":
    main()
