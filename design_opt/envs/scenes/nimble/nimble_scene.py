from tool_design.envs.scenes.scene import Scene


class NimbleScene(Scene):

    NUM_SIM_STEPS = 80

    def __init__(self, world, **kwargs):
        super().__init__(world, **kwargs)
        self.world.setTimeStep(self.TIME_STEP)

    def handle_kwargs(self, **kwargs):
        super().handle_kwargs(**kwargs)
        self.MANIP_SEGMENTS = kwargs.get('manip_segments', self.MANIP_SEGMENTS)
        self.gui= kwargs.get('gui', False)

    def render(self):
        pass
