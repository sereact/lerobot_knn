"""Per-episode source info of the tri-arm-gate-testing datasets (everything but capture quality)."""

import json
from pathlib import Path

ROOT = Path("/home/denis-office/git/lerobot/outputs/tri-arm-gate-testing")
first = True
rows = []
for d in sorted(ROOT.iterdir()):
    rc = json.loads((d / "original" / "meta" / "recording_configs.json").read_text())
    for ep in rc["episodes"]:
        si = ep["source_info"]
        if first:
            print("episode keys:", sorted(ep), "\nsource_info keys:", sorted(si))
            print({k: v for k, v in si.items() if k != "quality"})
            first = False
        rows.append((d.name, ep["episode_index"], {k: v for k, v in si.items() if k not in ("quality", "config_hash")}))
for name, idx, si in rows[:6]:
    print(name, idx, si)
keys = sorted({k for _, _, si in rows for k in si})
for k in keys:
    vals = {json.dumps(si.get(k))[:80] for _, _, si in rows}
    print(f"{k}: {len(vals)} distinct", sorted(vals)[:6])
