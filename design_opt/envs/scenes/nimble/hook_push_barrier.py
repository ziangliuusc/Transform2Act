import numpy as np
import torch
import nimblephysics as nimble

from tool_design.envs.scenes.nimble.hook_push import HookPushNimble


class HookPushBarrierNimble(HookPushNimble):

    OBJECT_SETTINGS = [
        [1.0, 1.8],
        [1.0, 2.0],
        [0.8, 2.0],
    ]
    GOAL_POSITION = (0.5, 1.8)

    def handle_kwargs(self, **kwargs):
        super().handle_kwargs(**kwargs)

    def __init__(self, world, **kwargs):
        super().__init__(world, **kwargs)

    def setup_world(self):
        super().setup_world()

        def make_barrier(position):
            barrier = nimble.dynamics.Skeleton()
            barrier.setName('barrier')  # important for rendering shadows
            barrierJoint, floorBody = barrier.createWeldJointAndBodyNodePair()
            barrierOffset = nimble.math.Isometry3()
            barrierOffset.set_translation(position)
            barrierJoint.setTransformFromParentBodyNode(barrierOffset)
            barrierShape: nimble.dynamics.ShapeNode = floorBody.createShapeNode(nimble.dynamics.BoxShape(
                [0.6, 0.1, .1]))
            barrierVisual: nimble.dynamics.VisualAspect = barrierShape.createVisualAspect()
            barrierVisual.setColor([0.5, 0.5, 0.5])
            barrierVisual.setCastShadows(False)
            barrierShape.createCollisionAspect()
            return barrier

        barrier1 = make_barrier([0.9, 1.5, 0])
        barrier2 = make_barrier([1.95, 1.5, 0])
        self.world.addSkeleton(barrier1)
        self.world.addSkeleton(barrier2)

