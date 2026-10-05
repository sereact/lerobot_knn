"""Unit tests of the KNN policy on a tiny synthetic memory (CPU, no downloads)."""

import numpy as np
import pytest
import torch

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.utils.constants import ACTION, OBS_STATE

from lerobot_policy_knn import KNNConfig, KNNPolicy, make_knn_pre_post_processors
from lerobot_policy_knn.features import FrameIndex
from lerobot_policy_knn.memory import build_memory

N_TASKS, N_EPISODES, N_FRAMES = 2, 3, 20
TASKS = ["pick up the bowl", "open the drawer"]


def make_frames() -> FrameIndex:
    rows = []
    for task in range(N_TASKS):
        for e in range(N_EPISODES):
            ep = task * N_EPISODES + e
            for f in range(N_FRAMES):
                rows.append((task, ep, f))
    task_index = np.array([r[0] for r in rows])
    episode_index = np.array([r[1] for r in rows])
    frame_index = np.array([r[2] for r in rows])
    n = len(rows)
    # state: position walks with the frame index, offset per episode; no rotation (axis-angle zero)
    state = np.zeros((n, 8), dtype=np.float32)
    state[:, 0] = frame_index * 0.1 + episode_index * 0.01
    state[:, 1] = task_index
    state[:, 6:] = 0.02  # gripper fingers: constant
    # action encodes (episode, frame) so retrieved chunks are identifiable
    action = np.zeros((n, 7), dtype=np.float32)
    action[:, 0] = episode_index
    action[:, 1] = frame_index
    action[:, 6] = np.where(frame_index < 10, -1.0, 1.0)
    ep_start = {ep: np.nonzero(episode_index == ep)[0].min() for ep in np.unique(episode_index)}
    episode_end = np.array([ep_start[ep] + N_FRAMES - 1 for ep in episode_index])
    return FrameIndex(
        state=state,
        action=action,
        episode_index=episode_index,
        frame_index=frame_index,
        task_index=task_index,
        episode_end=episode_end,
        episode_length=np.full(n, N_FRAMES),
        task_names=list(TASKS),
    )


def make_policy(**overrides) -> KNNPolicy:
    kwargs = {
        "device": "cpu",
        "camera_weights": {"observation.images.image": 0.0},
        "state_weight": 1.0,
        "k": 2,
        "chunk_size": 4,
        "n_action_steps": 2,
        "state_axis_angle_start": 3,
    }
    kwargs.update(overrides)
    cfg = KNNConfig(**kwargs)
    cfg.input_features = {OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(8,))}
    cfg.output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))}
    frames = make_frames()
    text = torch.nn.functional.normalize(torch.randn(N_TASKS, 16), dim=-1)
    buffers, _ = build_memory(cfg, frames, load_raw=None, task_text_emb=text)
    policy = KNNPolicy(cfg)
    policy.load_memory(buffers)
    return policy.eval()


def obs_batch(frames: FrameIndex, rows: list[int], tasks: list[str]) -> dict:
    return {OBS_STATE: torch.from_numpy(frames.state[rows]), "task": tasks}


def test_task_filter_and_one_per_episode():
    policy = make_policy(k=3)
    frames = make_frames()
    row = 5  # task 0, episode 0, frame 5
    query = policy.encode_observation(obs_batch(frames, [row], [TASKS[0]]))
    task = policy.task_ids([TASKS[0]])
    rows, weights, _ = policy.retrieve(query, task_ids=task)
    episodes = policy.key_episode[rows[0]].tolist()
    assert len(set(episodes)) == 3, "one neighbour per demonstration"
    assert all(ep < N_EPISODES for ep in episodes), "only demonstrations of the prompted task"
    assert torch.allclose(weights.sum(-1), torch.ones(1))


def test_exclude_episode():
    policy = make_policy(k=2)
    frames = make_frames()
    query = policy.encode_observation(obs_batch(frames, [5], [TASKS[0]]))
    rows, _, _ = policy.retrieve(
        query, task_ids=policy.task_ids([TASKS[0]]), exclude_episode=torch.tensor([0])
    )
    assert 0 not in policy.key_episode[rows[0]].tolist()


def test_chunk_queue_and_env_ids():
    policy = make_policy(k=1, chunk_size=4, n_action_steps=2)
    frames = make_frames()
    batch = obs_batch(frames, [3, 3 + N_FRAMES], [TASKS[0], TASKS[0]])
    assert policy.needs_observation([0, 1]) == [True, True]
    a0 = policy.select_action(batch, env_ids=[0, 1])
    assert a0.shape == (2, 7)
    assert policy.needs_observation([0, 1]) == [False, False]
    a1 = policy.select_action(batch, env_ids=[0, 1])
    # second action of the same chunk: next frame of the same demonstration
    assert torch.equal(a1[:, 1], a0[:, 1] + 1)
    assert policy.needs_observation([0, 1]) == [True, True]
    policy.reset_envs([0])
    assert 0 not in policy._slots and 1 in policy._slots


def test_gripper_sign_and_chunk_hold_at_episode_end():
    policy = make_policy(k=1, chunk_size=6, n_action_steps=6)
    frames = make_frames()
    last = N_FRAMES - 2  # chunk runs past the end of the demonstration
    chunk = policy.predict_action_chunk(obs_batch(frames, [last], [TASKS[0]]))
    assert chunk.shape == (1, 6, 7)
    assert set(chunk[0, :, 6].tolist()) <= {-1.0, 1.0}
    assert chunk[0, -1, 1].item() == N_FRAMES - 1, "actions are held at the episode end"


def test_consistency_prefers_continuation():
    policy = make_policy(k=1, consistency_weight=0.99, consistency_window=2, n_action_steps=1, chunk_size=2)
    frames = make_frames()
    first = policy.select_action(obs_batch(frames, [N_FRAMES + 4], [TASKS[0]]), env_ids=[0])
    ep = int(first[0, 0])
    # query closest to another demonstration (but not an exact memory frame, whose distance would be 0):
    # stickiness keeps following the first one
    batch = obs_batch(frames, [2 * N_FRAMES + 5], [TASKS[0]])
    batch[OBS_STATE][:, 0] += 0.003
    second = policy.select_action(batch, env_ids=[0])
    assert int(second[0, 0]) == ep
    policy.config.consistency_weight = 0.0
    policy.reset()
    policy.select_action(obs_batch(frames, [N_FRAMES + 4], [TASKS[0]]), env_ids=[0])
    assert int(policy.select_action(batch, env_ids=[0])[0, 0]) != ep, "without stickiness it switches"


def test_temporal_ensemble_shape():
    policy = make_policy(temporal_ensemble_coeff=0.1, n_action_steps=1)
    frames = make_frames()
    batch = obs_batch(frames, [3, 4], [TASKS[0], TASKS[1]])
    for _ in range(3):
        action = policy.select_action(batch)
        assert action.shape == (2, 7)


def test_save_load_roundtrip(tmp_path):
    policy = make_policy()
    frames = make_frames()
    batch = obs_batch(frames, [7, 30], [TASKS[0], TASKS[1]])
    expected = policy.predict_action_chunk(batch)
    policy.save_pretrained(tmp_path)
    pre, post = make_knn_pre_post_processors(policy.config)
    pre.save_pretrained(tmp_path)
    post.save_pretrained(tmp_path)
    loaded = KNNPolicy.from_pretrained(tmp_path)
    assert loaded.memory_size == policy.memory_size
    torch.testing.assert_close(loaded.predict_action_chunk(batch), expected)


@pytest.mark.parametrize("stride", [1, 3])
def test_memory_stride_keeps_full_rate_actions(stride):
    policy = make_policy(memory_stride=stride, k=1)
    assert policy.memory_size == N_TASKS * N_EPISODES * len(range(0, N_FRAMES, stride))
    assert policy.actions.shape[0] == N_TASKS * N_EPISODES * N_FRAMES


def test_shifted_rows_stay_in_episode():
    policy = make_policy()
    shifted = policy._shifted_rows(3)
    frames, episodes = policy.key_frame, policy.key_episode
    assert torch.equal(episodes[shifted], episodes)
    assert torch.equal(frames[shifted], (frames - 3).clamp_min(0))


def test_history_matching_runs_and_is_used():
    policy = make_policy(history_weight=1.0, k=1, chunk_size=2, n_action_steps=1)
    frames = make_frames()
    for f in range(4):
        action = policy.select_action(obs_batch(frames, [f + 5], [TASKS[0]]), env_ids=[0])
        assert action.shape == (1, 7)
    slot = policy._slots[0]
    assert slot.prev_query is not None and slot.prev_t == slot.t - 1
    assert policy.history_distance([slot]).shape == (1, policy.memory_size)


def test_norm_preserving_restores_mean_magnitude():
    policy = make_policy(norm_preserving=True, gripper_mode="mean")
    rows = torch.tensor([[0, N_FRAMES]])  # frames of two different demonstrations
    weights = torch.tensor([[0.5, 0.5]])
    out = policy.aggregate(rows, weights)
    chunks = policy.gather_chunks(rows)
    expected = chunks[..., :3].norm(dim=-1).mean(1)
    torch.testing.assert_close(out[..., :3].norm(dim=-1), expected)


def test_soft_task_conditioning_prefers_prompted_task():
    policy = make_policy(task_conditioning="soft", text_weight=100.0, k=3, one_per_episode=True)
    frames = make_frames()
    query = policy.encode_observation(obs_batch(frames, [5], [TASKS[1]]))
    rows, _, _ = policy.retrieve(query, task_ids=policy.task_ids([TASKS[1]]))
    assert all(ep >= N_EPISODES for ep in policy.key_episode[rows[0]].tolist())
    # without the text term, the state alone points at task 0
    policy.config.text_weight = 0.0
    rows, _, _ = policy.retrieve(query, task_ids=policy.task_ids([TASKS[1]]))
    assert any(ep < N_EPISODES for ep in policy.key_episode[rows[0]].tolist())


@pytest.mark.parametrize("norm", ["pair_median", "nn_median", "none"])
def test_distance_normalization_modes(norm):
    policy = make_policy(distance_normalization=norm)
    assert set(policy.config.modality_nn_scales) >= {"key_state", "text"}
    frames = make_frames()
    action = policy.select_action(obs_batch(frames, [3], [TASKS[0]]))
    assert action.shape == (1, 7)


def test_mode_seeking_ignores_minority_mode():
    from lerobot_policy_knn.modeling_knn import mode_seeking_weights

    # three neighbours go left (+1), two go right (-1): the mean (~0.2) is an action nobody took
    samples = torch.tensor([[[1.0], [1.05], [0.95], [-1.0], [-1.02]]])
    weights = torch.full((1, 5), 0.2)
    w = mode_seeking_weights(samples, weights, bandwidth=0.2)
    assert (w[0, :3].sum() > 0.99).item()
    torch.testing.assert_close((w[..., None] * samples).sum(1), torch.tensor([[1.0]]), atol=0.02, rtol=0)
    # a very wide kernel falls back to the plain mean
    w_wide = mode_seeking_weights(samples, weights, bandwidth=1e3)
    torch.testing.assert_close(w_wide, weights)
    # neighbours that agree keep their weights (plain mean) under an absolute bandwidth
    agree = torch.tensor([[[1.0], [1.05], [0.95], [1.02], [0.98]]])
    torch.testing.assert_close(
        mode_seeking_weights(agree, weights, bandwidth=0.5), weights, atol=0.01, rtol=0
    )


def test_mode_aggregation_in_policy():
    policy = make_policy(
        action_mode_bandwidth=0.1, k=4, temporal_ensemble_coeff=0.1, ensemble_mode_bandwidth=0.5
    )
    frames = make_frames()
    for _ in range(3):
        assert policy.select_action(obs_batch(frames, [3], [TASKS[0]])).shape == (1, 7)


def test_retry_backtracks_when_unseen():
    policy = make_policy(
        k=1, chunk_size=2, n_action_steps=1, retry_distance_threshold=3.0, retry_patience=2,
        retry_backtrack_steps=6, retry_max=1, retry_cooldown=0, retry_tabu=True,
    )  # fmt: skip
    frames = make_frames()
    assert torch.all(policy.task_distance_scale() > 0)
    executed = [policy.select_action(obs_batch(frames, [f], [TASKS[0]]), env_ids=[0])[0] for f in range(6)]
    slot = policy._slots[0]
    assert slot.retries == 0 and len(slot.executed) == 6
    far = obs_batch(frames, [5], [TASKS[0]])
    far[OBS_STATE][:, 0] += 50.0  # nothing like this state was ever demonstrated
    first_back = None
    for _ in range(3):
        a = policy.select_action(far, env_ids=[0])[0]
        if slot.retries:
            first_back = a
            break
    assert slot.retries == 1, "an unseen state triggers a retry"
    # the first rewind action is the newest executed action, negated
    assert first_back is not None
    assert policy.needs_observation([0]) == [False]
    assert slot.tabu, "the demos followed by the failed attempt are excluded afterwards"
    # after rewinding, retrieval resumes
    while slot.backtrack:
        policy.select_action(far, env_ids=[0])
    assert policy.needs_observation([0]) == [True]
    del executed


def test_retry_when_stuck():
    policy = make_policy(
        k=1, chunk_size=2, n_action_steps=1, retry_stuck_steps=5, retry_stuck_eps=0.01, retry_cooldown=0
    )
    frames = make_frames()
    still = obs_batch(frames, [3], [TASKS[0]])
    for _ in range(8):
        policy.select_action(still, env_ids=[0])
    assert policy._slots[0].retries == 1


def test_gripper_weight_separates_finger_states():
    policy = make_policy(state_weight=0.0, gripper_weight=1.0, k=1)
    frames = make_frames()
    batch = obs_batch(frames, [3], [TASKS[0]])
    batch[OBS_STATE][:, -2:] = 0.0  # fingers fully closed: never seen in the (constant 0.02) demos
    _, _, d_closed = policy.retrieve(policy.encode_observation(batch), task_ids=policy.task_ids([TASKS[0]]))
    _, _, d_normal = policy.retrieve(
        policy.encode_observation(obs_batch(frames, [3], [TASKS[0]])), task_ids=policy.task_ids([TASKS[0]])
    )
    assert d_closed[0, 0] > d_normal[0, 0]


def test_shift_and_tta_offsets():
    from lerobot_policy_knn.modeling_knn import _shift, _tta_offsets

    img = torch.arange(16.0).view(1, 1, 4, 4)
    right = _shift(img, 0, 1)
    assert torch.equal(right[..., 1:], img[..., :-1]) and torch.equal(right[..., 0], img[..., 0])
    down = _shift(img, 1, 0)
    assert torch.equal(down[..., 1:, :], img[..., :-1, :])
    assert _tta_offsets(0, 4) == [(0, 0)]
    assert len(_tta_offsets(4, 4)) == 5


def test_history_gap_uses_older_observation():
    policy = make_policy(history_weight=1.0, history_gap=3, k=1, chunk_size=2, temporal_ensemble_coeff=0.1)
    frames = make_frames()
    for f in range(6):
        policy.select_action(obs_batch(frames, [f + 2], [TASKS[0]]), env_ids=[0])
    slot = policy._slots[0]
    assert all(t >= slot.t - 1 - 3 for t, _ in slot.past), "the buffer only keeps what the gap needs"
    assert policy.history_distance([slot]) is not None
