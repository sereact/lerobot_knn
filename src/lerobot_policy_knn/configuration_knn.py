"""Configuration of the k-nearest-neighbour (kNN) action-chunk retrieval policy."""

from dataclasses import dataclass, field

from lerobot.configs import FeatureType, NormalizationMode, PreTrainedConfig
from lerobot.optim import AdamWConfig

DEFAULT_CAMERA_WEIGHTS = {"observation.images.image": 1.0, "observation.images.image2": 1.0}


@PreTrainedConfig.register_subclass("knn")
@dataclass
class KNNConfig(PreTrainedConfig):
    """Non-parametric policy: embed the observation, retrieve the k closest demonstration frames, and
    replay (a weighted average of) the action chunks that followed them.

    There are no trainable parameters. "Training" (``lerobot-knn-fit``) embeds every dataset frame with
    frozen encoders and stores the result as the policy's memory. Everything under *retrieval*,
    *aggregation* and *chunking* is an inference-time knob that can be overridden at evaluation time
    (``--policy.k=4`` ...). The *embedding* knobs must match the memory the policy was fitted with.

    Distance between a query ``q`` and a memory frame ``m`` is

        D(q, m) = sum_mod  w_mod * ||q_mod - m_mod||^2 / scale_mod

    over the modalities (one per camera, the proprioceptive state, and the time step), where
    ``scale_mod`` is the median squared distance between random frames of the same task (computed at fit
    time), so the weights are comparable across modalities.
    """

    # ---- Observation embedding (must match the fitted memory) ----
    image_encoder: str = "dinov2_small"
    image_resolution: int = 224
    image_pooling: str = "cls"
    # Retrieval weight of every camera. Cameras with weight 0 are neither stored nor embedded.
    camera_weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_CAMERA_WEIGHTS))
    state_weight: float = 1.0
    # Extra weight on the gripper finger positions (the last `gripper_state_dims` state dims, standardised):
    # the fingers only close all the way when a grasp missed, which no demonstration does mid-task.
    gripper_weight: float = 0.0
    gripper_state_dims: int = 2
    # Start index of an axis-angle triplet in the state (LIBERO: 3). It is replaced by the rotation
    # matrix before computing distances, avoiding the +-pi wrap-around of the axis-angle encoding.
    state_axis_angle_start: int | None = 3
    # Weight of |t_query - t_memory|, t being the step index since the start of the episode.
    time_weight: float = 0.0
    # "zscore": standardise every raw key dimension with memory statistics before the unit-norm (cosine) step.
    # VLM hidden states put most of their energy in a handful of "massive activation" channels, which otherwise
    # dominate the cosine distance.
    key_norm: str = "none"
    # Test-time augmentation: also embed this many copies shifted by +-query_tta_px pixels and average the
    # (unit-norm) keys of all views. 0 disables.
    query_tta: int = 0
    query_tta_px: int = 4
    # Keep every n-th dataset frame as a retrievable key (action chunks always use every frame).
    memory_stride: int = 1

    # ---- Task conditioning ----
    text_encoder: str = "clip_b16"
    # "filter": restrict retrieval to the demonstrations of the task whose prompt embedding is closest to
    #           the query prompt (exact prompts short-circuit the text encoder).
    # "soft":   retrieve from the whole memory, adding text_weight * ||e_prompt - e_task(m)||^2 / scale.
    # "none":   ignore the prompt and retrieve from the whole memory.
    task_conditioning: str = "filter"
    text_weight: float = 1.0
    # How modality distances are made comparable before weighting:
    # "pair_median": divide by the median distance between random frames of the same task;
    # "nn_median":   divide by the median nearest-neighbour distance to other demos of the same task;
    # "none":        raw distances.
    distance_normalization: str = "pair_median"

    # ---- Retrieval & aggregation ----
    k: int = 8
    # Neighbour weights are softmax(-D_i / (temperature * mean_k D)). Large -> uniform average.
    temperature: float = 1.0
    # Keep at most one neighbour (the closest frame) per demonstration episode.
    one_per_episode: bool = True
    # Trajectory stickiness: distances of frames continuing the previously matched demonstration (same
    # episode, within +-consistency_window frames of the expected frame) are scaled by (1 - weight).
    consistency_weight: float = 0.0
    consistency_window: int = 4
    # History matching: also compare the previous retrieval's observation (taken `gap` steps ago) with
    # the frame `gap` steps before each candidate in its demonstration, weighted by this factor.
    history_weight: float = 0.0
    # Steps between the current and the past observation used for history matching. 0 = the previous
    # retrieval (1 step back with temporal ensembling, which carries little motion information).
    history_gap: int = 0
    # Rescale averaged translation/rotation commands to the neighbours' mean magnitude.
    norm_preserving: bool = False
    # Mode-seeking aggregation: neighbours whose action chunks differ from the dominant chunk by more than
    # ~this RMS action difference (action units; LIBERO actions are in [-1, 1]) are down-weighted instead of
    # being averaged into an action none of them took. 0 disables (plain weighted mean).
    action_mode_bandwidth: float = 0.0
    # Same idea across the overlapping chunk predictions combined by temporal ensembling.
    ensemble_mode_bandwidth: float = 0.0
    # "sign": binarise the averaged gripper command; "mean": keep the weighted average.
    gripper_mode: str = "sign"
    gripper_index: int = -1
    # Action dims retrieved relative to the state (LeRobot's relative actions): every neighbour's chunk is re-anchored
    # on the query, chunk - state(neighbour key frame) + state(query), on the same state indices. For absolute
    # end-effector targets (e.g. derroboter's xyz + rot6d per arm, grippers left absolute): a neighbour recorded
    # elsewhere in the workspace then contributes its motion instead of its position. Empty: absolute actions.
    relative_action_dims: list[int] = field(default_factory=list)

    # ---- Retry / backtracking ----
    # When the observation looks unlike any demonstration (top-1 distance above this many times the task's
    # typical leave-one-demo-out nearest-neighbour distance, for `retry_patience` consecutive retrievals)
    # or the arm is stuck (end effector moved < retry_stuck_eps metres over retry_stuck_steps steps), the
    # policy rewinds: it replays its last `retry_backtrack_steps` actions negated and in reverse order, then
    # retrieves again (optionally excluding the demonstrations the failed attempt followed). 0 disables.
    retry_distance_threshold: float = 0.0
    retry_patience: int = 5
    retry_stuck_steps: int = 0
    retry_stuck_eps: float = 0.005
    retry_backtrack_steps: int = 20
    retry_max: int = 2
    retry_cooldown: int = 30
    retry_tabu: bool = True

    # ---- Chunking ----
    chunk_size: int = 16
    n_action_steps: int = 8
    # ACT-style temporal ensembling over overlapping chunks (queries every step when set).
    temporal_ensemble_coeff: float | None = None

    # ---- Memory layout (written by lerobot-knn-fit, do not edit) ----
    memory_shapes: dict[str, list[int]] = field(default_factory=dict)
    modality_scales: dict[str, float] = field(default_factory=dict)
    modality_nn_scales: dict[str, float] = field(default_factory=dict)
    task_names: list[str] = field(default_factory=list)
    dataset_repo_id: str | None = None

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        }
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot exceed chunk_size ({self.chunk_size})"
            )
        if self.distance_normalization not in ("pair_median", "nn_median", "none"):
            raise ValueError(f"Unknown distance_normalization '{self.distance_normalization}'")
        if self.task_conditioning not in ("filter", "soft", "none"):
            raise ValueError(f"Unknown task_conditioning '{self.task_conditioning}'")
        if self.gripper_mode not in ("sign", "mean"):
            raise ValueError(f"Unknown gripper_mode '{self.gripper_mode}'")
        if self.k < 1:
            raise ValueError("k must be >= 1")
        if self.relative_action_dims and self.state_axis_angle_start is not None:
            raise ValueError("relative_action_dims index the raw state: set state_axis_angle_start=None")

    @property
    def active_cameras(self) -> list[str]:
        return [cam for cam, w in self.camera_weights.items() if w > 0]

    def validate_features(self) -> None:
        missing = [cam for cam in self.active_cameras if cam not in self.input_features]
        if self.input_features and missing:
            raise ValueError(f"Cameras {missing} are weighted but not among the policy input features")
        if self.action_feature is None and self.output_features:
            raise ValueError("KNN policy requires an 'action' output feature")
        if self.input_features:
            visual = [k for k, v in self.input_features.items() if v.type is FeatureType.VISUAL]
            if self.active_cameras and not visual:
                raise ValueError("Cameras are weighted but the policy has no visual input feature")
        if not self.active_cameras and self.state_weight <= 0 and self.gripper_weight <= 0:
            raise ValueError("KNN policy needs at least one camera or a positive state/gripper weight")

    def get_optimizer_preset(self) -> AdamWConfig:
        # Nothing is trained; lerobot-knn-fit builds the memory instead.
        return AdamWConfig(lr=0.0)

    def get_scheduler_preset(self) -> None:
        return None

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
