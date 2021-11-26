
from acme import specs
from acme import types

# Make __version__ accessible.
from acme._metadata import __version__

# Expose core interfaces.
from acme.core import Actor
from acme.core import Learner
from acme.core import Saveable
from acme.core import VariableSource
from acme.core import Worker

# Expose the environment loop.
from acme.environment_loop import EnvironmentLoop

from acme.specs import make_environment_spec

from gym.envs.registration import register

register(
    id='TimePilot-v0'
    #entry_point='myenv.myenv:MyEnv',
)
