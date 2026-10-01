#!/usr/bin/env python
"""Serve a LeRobot policy to derroboter's inference node over its Zenoh policy protocol.

derroboter's inference node (``inference.policy_transport: zenoh_fb_shm``) sends one FlatBuffer request per control
tick to the Zenoh queryable ``services/policy/inference`` and executes the action chunk it gets back. This script
declares that queryable and answers it with a LeRobot policy (``--policy_path``: a local checkpoint or a Hub repo id;
plugin policies such as ``lerobot_policy_knn`` are discovered like in ``lerobot-eval``).

Request (built by derroboter's ``InferenceNode.get_observation``) -> policy input:

    observation.images.<cam>  uint8 [C, H, W] (or [T, C, H, W]: the newest is used)  -> float [1, C, H, W] in [0, 1]
    observation.state         [D] (or [T, D]: the newest is used)                      -> float [1, D]
    task                      str                                                      -> [str]

The reply is the predicted chunk, float32 [n_actions, action_dim], cut or padded (repeating its last action) to the
request's ``inference_config.n_actions``. The node re-queries every tick, so mostly the first rows are executed. The
action layout must be the one the node expects for its ``inference.action_type``, i.e. the layout of the derroboter
recordings (neo-cortex LeRobot export) the policy was trained on.

It runs as an ordinary process next to derroboter's Docker container: the container uses host networking, so its Zenoh
router is ``tcp/127.0.0.1:7447`` on the robot PC (``--connect``; ``ZENOH_CONFIG`` overrides it). derroboter itself
needs Python 3.10 and is not installed: the server needs Zenoh and FlatBuffers, and imports derroboter's payload codec
(one module, numpy/torch/flatbuffers only) from a derroboter checkout on the host (``--derroboter_src``):

    uv pip install eclipse-zenoh==1.9.0 flatbuffers==25.9.23

The tri-arm shoe-box kNN (station config with ``inference.action_type: dual_eef`` and this ``policy_zenoh_key``; the
node sends all three arms in ``observation.state``, the policy takes the two UR arms):

    .venv/bin/python scripts/derroboter_policy_server.py \\
        --policy_path=outputs/tri-arm-shoe-box-knn/policy_relative \\
        --key=services/policy/shoe_box_knn --state_slice=0:20 --max_jump_m=0.1 \\
        --task="take the shoe out of the box"

Check it from inside the container with ``venv/bin/python scripts/check_policy_server.py <key>``.
"""

import argparse
import importlib
import json
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch
from lerobot.configs import FeatureType, PreTrainedConfig
from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.import_utils import register_third_party_plugins

logger = logging.getLogger(__name__)

DEFAULT_KEY = "services/policy/inference"
PROTOCOL_MODULE = "derroboter.py.nodes.zenoh_fb_policy_protocol"


class PolicyHandler:
    """Decodes a derroboter policy request, runs the policy and encodes its action chunk."""

    def __init__(
        self,
        policy_path: str,
        device: str,
        protocol: ModuleType,
        rename_map: dict[str, str] | None = None,
        task: str | None = None,
        state_slice: slice | None = None,
        max_jump_m: float | None = None,
    ):
        register_third_party_plugins()
        cfg = PreTrainedConfig.from_pretrained(policy_path)
        cfg.device = device
        self.policy = get_policy_class(cfg.type).from_pretrained(policy_path, config=cfg)
        self.policy.eval()
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            cfg, pretrained_path=policy_path, preprocessor_overrides={"device_processor": {"device": device}}
        )
        self.protocol = protocol
        self.rename_map = rename_map or {}
        self.task = task
        self.state_slice = state_slice
        self.max_jump_m = max_jump_m
        self.cameras = [k for k, f in cfg.input_features.items() if f.type is FeatureType.VISUAL]
        state = cfg.input_features.get(OBS_STATE)
        self.state_dim = state.shape[0] if state is not None else None
        self.action_dim = cfg.output_features[ACTION].shape[0]
        self.lock = threading.Lock()  # one inference at a time, whatever thread Zenoh calls from
        self.n_requests = 0
        self.total_s = 0.0
        self.warned_rows = False
        logger.info(
            "Loaded %s policy from %s: cameras %s, state dim %s, action dim %d",
            cfg.type,
            policy_path,
            self.cameras,
            self.state_dim,
            self.action_dim,
        )

    def observation(self, request: dict[str, Any]) -> dict[str, Any]:
        request = {self.rename_map.get(k, k): v for k, v in request.items()}
        needed = self.cameras + ([OBS_STATE] if self.state_dim is not None else [])
        missing = [k for k in needed if k not in request]
        if missing:
            given = sorted(k for k in request if k.startswith("observation."))
            raise KeyError(f"request has no {missing} (it has {given}); map the names with --rename_map")
        obs: dict[str, Any] = {}
        for key in self.cameras:
            image = np.asarray(request[key])
            if image.ndim == 4:  # several image steps
                image = image[-1]
            obs[key] = torch.from_numpy(image).unsqueeze(0).float().div_(255.0)
        if self.state_dim is not None:
            state = np.asarray(request[OBS_STATE], dtype=np.float32)
            state = state.reshape(-1, state.shape[-1])[-1:]  # several state steps: the newest
            if self.state_slice is not None:
                state = state[:, self.state_slice]
            if state.shape[-1] != self.state_dim:
                raise ValueError(
                    f"{OBS_STATE} has {state.shape[-1]} values, the policy takes {self.state_dim}; "
                    "select them with --state_slice"
                )
            obs[OBS_STATE] = torch.from_numpy(np.ascontiguousarray(state))
        task = self.task if self.task is not None else str(request.get("task") or "")
        if not task:
            raise ValueError(
                "the node sent an empty task (nothing published on its `task` topic): pass --task"
            )
        obs["task"] = [task]
        return obs

    def check_jump(self, state: np.ndarray, action: np.ndarray) -> None:
        """Refuse a chunk whose first TCP target is more than max_jump_m from an arm's current TCP (derroboter's eef
        layout: 10 values per arm, xyz first, in the same order in the state and the action)."""
        for arm in range(min(len(state), len(action)) // 10):
            jump = float(np.linalg.norm(action[10 * arm : 10 * arm + 3] - state[10 * arm : 10 * arm + 3]))
            if jump > self.max_jump_m:
                raise ValueError(
                    f"first target of arm {arm} is {100 * jump:.1f} cm from its TCP "
                    f"(--max_jump_m {self.max_jump_m}): chunk not sent"
                )

    @torch.no_grad()
    def predict(self, obs: dict[str, Any]) -> np.ndarray:
        batch = self.preprocessor(obs)
        chunk = self.policy.predict_action_chunk(batch)
        if chunk.ndim == 2:  # (chunk_size, action_dim)
            chunk = chunk.unsqueeze(0)
        # post-process every action of the chunk, like LeRobot's async policy server
        actions = torch.stack([self.postprocessor(chunk[:, i]) for i in range(chunk.shape[1])], dim=1)
        return actions[0].float().cpu().numpy()

    def __call__(self, payload: bytes) -> bytes:
        start = time.perf_counter()
        request = self.protocol.decode_payload(payload, restore_torch=False)
        if not isinstance(request, dict):
            raise ValueError(f"expected a policy request dict, got {type(request).__name__}")
        obs = self.observation(request)
        with self.lock:
            actions = self.predict(obs)
        if self.max_jump_m is not None and OBS_STATE in obs:
            self.check_jump(obs[OBS_STATE][0].numpy(), actions[0])
        n_actions = int((request.get("inference_config") or {}).get("n_actions") or len(actions))
        if n_actions != len(actions):
            if not self.warned_rows:
                logger.warning(
                    "The policy predicts %d actions, the node asks for %d: %s",
                    len(actions),
                    n_actions,
                    "cutting the chunk" if n_actions < len(actions) else "repeating its last action",
                )
                self.warned_rows = True
            actions = actions[np.minimum(np.arange(n_actions), len(actions) - 1)]
        reply = self.protocol.encode_response(
            np.ascontiguousarray(actions, dtype=np.float32),
            metadata={"server": "lerobot", "policy": self.policy.name, "action_dim": self.action_dim},
        )
        self.n_requests += 1
        self.total_s += time.perf_counter() - start
        if self.n_requests == 1 or self.n_requests % 300 == 0:
            logger.info(
                "request %d: reply %s, mean handler time %.1f ms",
                self.n_requests,
                actions.shape,
                1000 * self.total_s / self.n_requests,
            )
        return reply


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--policy_path", required=True, help="local checkpoint dir or Hub repo id")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--key", default=DEFAULT_KEY, help="Zenoh key (the node's inference.policy_zenoh_key)"
    )
    parser.add_argument(
        "--connect",
        default="tcp/127.0.0.1:7447",
        help="derroboter's Zenoh router (its containers use host networking: localhost on the robot PC); "
        "ZENOH_CONFIG overrides it",
    )
    parser.add_argument(
        "--derroboter_src",
        default=os.environ.get("DERROBOTER_SRC", "~/git/derroboter/derroboter/src"),
        help="derroboter's src dir, for its policy payload codec",
    )
    parser.add_argument(
        "--rename_map",
        type=json.loads,
        default={},
        help='request key -> policy key, e.g. \'{"observation.images.front": "observation.images.image"}\'',
    )
    parser.add_argument("--task", default=None, help="prompt to use instead of the node's inference.task")
    parser.add_argument(
        "--state_slice",
        type=lambda s: slice(*(int(x) if x else None for x in s.split(":"))),
        default=None,
        help="START:END of the node's observation.state the policy takes, e.g. 0:20 for the two UR arms of the "
        "tri-arm cell (the node sends left, right and head arm)",
    )
    parser.add_argument(
        "--max_jump_m",
        type=float,
        default=None,
        help="refuse (error reply: the node holds) a chunk whose first target is farther than this from an arm's "
        "current TCP, in metres; derroboter eef layout (10 values per arm, xyz first) in state and action",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True
    )

    sys.path.insert(0, str(Path(args.derroboter_src).expanduser()))
    protocol = importlib.import_module(PROTOCOL_MODULE)
    session = open_session(args.connect)  # before loading the policy: a missing router fails fast

    handler = PolicyHandler(
        args.policy_path,
        args.device,
        protocol,
        args.rename_map,
        args.task,
        args.state_slice,
        args.max_jump_m,
    )

    def on_query(query: Any) -> None:
        try:
            payload = query.payload.to_bytes() if query.payload is not None else b""
            query.reply(query.key_expr, handler(payload))
        except Exception as exc:  # the node logs the error reply and keeps its current actions
            logger.exception("Request failed: %s", exc)
            query.reply_err(str(exc).encode())

    queryable = session.declare_queryable(args.key, on_query)
    logger.info("Serving on Zenoh key %s", args.key)

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    try:
        while not stop.wait(0.5):
            pass
    finally:
        queryable.undeclare()
        session.close()
        logger.info("Stopped after %d requests", handler.n_requests)


def open_session(connect: str) -> Any:
    """Zenoh session to derroboter's router: ``ZENOH_CONFIG`` if set, else a client of ``connect``, with derroboter's
    runtime settings (shared memory on, no discovery)."""
    import zenoh

    if path := os.environ.get("ZENOH_CONFIG"):
        config = zenoh.Config.from_file(path)
    else:
        config = zenoh.Config()
        config.insert_json5("mode", '"client"')
        config.insert_json5("connect/endpoints", json.dumps([connect]))
    for key, value in (
        ("transport/shared_memory/enabled", "true"),
        ("scouting/multicast/enabled", "false"),
        ("scouting/gossip/enabled", "false"),
    ):
        config.insert_json5(key, value)
    try:
        return zenoh.open(config)
    except Exception as exc:
        raise SystemExit(
            f"Cannot reach the Zenoh router at {connect} ({exc}): start derroboter's manager first, "
            "or point --connect / ZENOH_CONFIG at the station's router"
        ) from exc


if __name__ == "__main__":
    main()
