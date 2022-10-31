import logging
from gym.envs.registration import register
from .gym_wrapper import GymWrapper
from .multi_agent_env import MultiAgentEnv

logger = logging.getLogger(__name__)
# listing all custom domains which do not need a wrapper #
li_custom_domains = [
                     'tree_maze-v1',
                     'tree_maze-v2',
                     'tree_maze-v3',
                     'load_unload-v0',
                     'load_unload-v1',
                     'load_unload-v2',
                     'cookie-v0',
                     'cookie-v1',
                     'cookie-v2',
                     'cookie-v3',
                     'meuleau_maze-v0',
                     'meuleau_maze-v1',
                     'meuleau_maze-v2',
                     'basic_scheduler-v0',
                     'basic_scheduler-v1',
                     'toh_d3_r3-v0',
                     'toh_d3_r3-v1',
                     'toh_d3_r3-v2'
                     ]

register(
    id='tree_maze-v1',
    entry_point='lib_domain.gridworld:TreeMazeEnvV1',
)

register(
    id='tree_maze-v2',
    entry_point='lib_domain.gridworld:TreeMazeEnvV2',
)

register(
    id='tree_maze-v3',
    entry_point='lib_domain.gridworld:TreeMazeEnvV3',
)

register(
    id='load_unload-v0',
    entry_point='lib_domain.gridworld:LoadUnloadEnv',
)

register(
    id='load_unload-v1',
    entry_point='lib_domain.gridworld:LoadUnloadEnvV1',
)

register(
    id='load_unload-v2',
    entry_point='lib_domain.gridworld:LoadUnloadEnvV2',
)

register(
    id='meuleau_maze-v0',
    entry_point='lib_domain.gridworld.meuleau_maze:MeuleauMazeEnv',
)

register(
    id='meuleau_maze-v1',
    entry_point='lib_domain.gridworld.meuleau_maze:MeuleauMazeEnvV1',
)

register(
    id='meuleau_maze-v2',
    entry_point='lib_domain.gridworld.meuleau_maze:MeuleauMazeEnvV2',
)

register(
    id='cookie-v0',
    entry_point='lib_domain.gridworld:CookieEnv',
)

register(
    id='cookie-v1',
    entry_point='lib_domain.gridworld:CookieEnvV1',
)

register(
    id='cookie-v2',
    entry_point='lib_domain.gridworld:CookieEnvV2',
)

register(
    id='cookie-v3',
    entry_point='lib_domain.gridworld:CookieEnvV3',
)

register(
    id='basic_scheduler-v0',
    entry_point='lib_domain.scheduler:BasicSchedulerEnv',
)

register(
    id='basic_scheduler-v1',
    entry_point='lib_domain.scheduler:BasicSchedulerEnvV1',
)

register(
    id='toh_d3_r3-v0',
    entry_point='lib_domain.tower_of_hanoi:ToHd3r3Env',
)

register(
    id='toh_d3_r3-v1',
    entry_point='lib_domain.tower_of_hanoi:ToHd3r3EnvV1',
)

register(
    id='toh_d3_r3-v2',
    entry_point='lib_domain.tower_of_hanoi:ToHd3r3EnvV2',
)
