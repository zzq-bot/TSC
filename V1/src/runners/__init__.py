REGISTRY = {}

from .episode_runner import EpisodeRunner
REGISTRY["episode"] = EpisodeRunner

from .parallel_runner import ParallelRunner
REGISTRY["parallel"] = ParallelRunner

from .dynamic_episode_runner import DynamicEpisodeRunner
REGISTRY["dynamic_episode"] = DynamicEpisodeRunner
