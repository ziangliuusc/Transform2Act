
class Scene:
    TARGET_FPS = 60
    TIME_STEP = 1.0 / TARGET_FPS

    DENSITY = 1
    FRICTION = 0.3
    RESTITUTION = 0.8
    REWARD_SCALE = 1

    NUM_SIM_STEPS = 80
    FIXED_TRAJ = None

    def __init__(self, world, **kwargs):
        self.world = world
        self.handle_kwargs(**kwargs)

    def handle_kwargs(self, **kwargs):
        self.NUM_SIM_STEPS = kwargs.get('timesteps', self.NUM_SIM_STEPS)

    def reset(self):
        raise NotImplementedError

    def get_obs(self):
        raise NotImplementedError

    def close(self):
        pass

    def render(self):
        raise NotImplementedError
