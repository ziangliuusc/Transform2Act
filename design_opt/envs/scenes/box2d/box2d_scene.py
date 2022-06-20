import numpy as np

from Box2D import b2
from tool_design.envs.scenes.scene import Scene
from tool_design.envs.drawing import DrawFuncs


class Box2DScene(Scene):

    FINAL_POS = (14.5, 8.5)
    START_ANGLE = b2.pi / 2 - 0.2

    def __init__(self, world, **kwargs):
        super().__init__(world, **kwargs)
        self.drawer = DrawFuncs(w=480, h=480, ppm=20)
        self.drawer.install()

        if not self.return_frames:
            self.TIME_STEP *= self.speedup
            self.NUM_SIM_STEPS //= self.speedup

    def handle_kwargs(self, **kwargs):
        super().handle_kwargs(**kwargs)
        self.speedup = kwargs.get('speedup', 1)
        self.return_frames = kwargs.get('return_frames',
                                        False)  # whether the step function should return all frames from simulated traj

    def render(self):
        self.drawer.clear_screen()
        self.drawer.draw_world(self.world)
        return self.drawer.screen.copy()

    @property
    def manipulation_action_size(self):
        # By default, Box2D envs have no manipulation component
        return 0

    @staticmethod
    def vectorize_body_state(body):
        return np.concatenate([body.position, [body.angle]])

    @staticmethod
    def set_body_state(body, state):
        assert len(state) == 3, 'body state must be length 3!'
        body.position = state[:2]
        body.angle = state[2]