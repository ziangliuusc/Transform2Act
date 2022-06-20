from .hopper import HopperEnv
from .swimmer import SwimmerEnv
from .ant import AntEnv
from .gap import GapEnv
from .box2d_chain import Box2DChainEnv


env_dict = {
    'hopper': HopperEnv,
    'swimmer': SwimmerEnv,
    'ant': AntEnv,
    'gap': GapEnv,
    'box2d_chain': Box2DChainEnv,
}
