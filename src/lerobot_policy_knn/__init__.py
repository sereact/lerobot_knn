"""k-nearest-neighbour action retrieval policy plugin for LeRobot (``--policy.type=knn``)."""

try:
    import lerobot  # noqa: F401
except ImportError as e:
    raise ImportError("lerobot is not installed. Please install lerobot to use this policy package.") from e

from .configuration_knn import KNNConfig
from .modeling_knn import KNNPolicy
from .processor_knn import make_knn_pre_post_processors

__all__ = [
    "KNNConfig",
    "KNNPolicy",
    "make_knn_pre_post_processors",
]
