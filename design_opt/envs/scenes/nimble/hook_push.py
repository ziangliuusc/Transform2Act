import numpy as np
import torch
import nimblephysics as nimble

from tool_design.envs.scenes.nimble.nimble_scene import NimbleScene


class HookPushNimble(NimbleScene):

    GOAL_POSITION = (0.3, 1.0)
    OBJECT_SETTINGS = [
        [1.0, 1.8],
        [1.0, 2.0],
        [0.8, 2.0],
    ]
    MANIP_SEGMENTS = 4
    NUM_SIM_STEPS = 120
    FIXED_TRAJ = [-1, -1] * MANIP_SEGMENTS

    def handle_kwargs(self, **kwargs):
        super().handle_kwargs(**kwargs)
        self.loss_fn = kwargs.get('loss_fn', 'min')
        self.bound_box_size = kwargs.get('bbox_size', 1.0)
        self.object_setting = kwargs.get('object_setting', 0)
        self.use_constraint = kwargs.get('constraint', False)
        self.constraint_factor = kwargs.get('constraint_factor', 100)
        self.randomize_object_position = kwargs.get('randomize_obj_position', False)
        self.random_obj_range = kwargs.get('random_obj_range', 2.0)
        print('random obj position: ', self.randomize_object_position)
        if self.loss_fn == 'track_demo':
            import pickle
            self.demo = pickle.load(open('obj_pos_1_demo_traj.pkl', 'rb'))

    def __init__(self, world, **kwargs):
        super().__init__(world, **kwargs)
        assert self.object_setting < len(self.OBJECT_SETTINGS), f'unknown object setting {self.object_setting}'
        self.object_position = self.OBJECT_SETTINGS[self.object_setting]
        self.setup_world()

    @property
    def manipulation_action_size(self):
        return self.MANIP_SEGMENTS * self.world.getActionSize()

    def setup_world(self):
        # Set up the projectile
        def make_projectile():
            projectile = nimble.dynamics.Skeleton()
            projectileJoint, projectileNode = projectile.createTranslationalJoint2DAndBodyNodePair()
            projectileJoint.setXYPlane()
            projectileNode.setRestitutionCoeff(0.8)
            projectileNode.setMass(3)
            projectileShape = projectileNode.createShapeNode(nimble.dynamics.SphereShape(0.05))
            projectileVisual = projectileShape.createVisualAspect()
            projectileShape.createCollisionAspect()
            projectileVisual.setColor([0.7, 0.7, 0.7])
            projectileVisual.setCastShadows(False)
            projectileJoint.setControlForceUpperLimit(0, 0)
            projectileJoint.setControlForceLowerLimit(0, 0)
            projectileJoint.setControlForceUpperLimit(1, 0)
            projectileJoint.setControlForceLowerLimit(1, 0)
            projectileJoint.setVelocityUpperLimit(0, 1000.0)
            projectileJoint.setVelocityLowerLimit(0, -1000.0)
            projectileJoint.setVelocityUpperLimit(1, 1000.0)
            projectileJoint.setVelocityLowerLimit(1, -1000.0)

            projectile.setPositions(np.array(self.object_position))
            return projectile

        self.projectile = make_projectile()
        self.world.addSkeleton(self.projectile)

        def make_floor():
            # Floor
            floor = nimble.dynamics.Skeleton()
            floor.setName('floor')  # important for rendering shadows

            floorJoint, floorBody = floor.createWeldJointAndBodyNodePair()
            floorOffset = nimble.math.Isometry3()
            floorOffset.set_translation([1.2, -0.7, 0])
            floorJoint.setTransformFromParentBodyNode(floorOffset)
            floorShape: nimble.dynamics.ShapeNode = floorBody.createShapeNode(nimble.dynamics.BoxShape(
                [3.5, 0.25, .5]))
            floorVisual: nimble.dynamics.VisualAspect = floorShape.createVisualAspect()
            floorVisual.setColor([0.5, 0.5, 0.5])
            floorVisual.setCastShadows(False)
            floorShape.createCollisionAspect()
            return floor

        floor = make_floor()
        self.world.addSkeleton(floor)
        # Target
        self.target = torch.Tensor(self.GOAL_POSITION).cpu()

        def make_target():
            target = nimble.dynamics.Skeleton()
            target.setName('target')  # important for rendering shadows

            targetJoint, targetBody = floor.createWeldJointAndBodyNodePair()
            targetOffset = nimble.math.Isometry3()
            targetOffset.set_translation([self.GOAL_POSITION[0], self.GOAL_POSITION[1], 0])
            targetJoint.setTransformFromParentBodyNode(targetOffset)
            targetShape = targetBody.createShapeNode(nimble.dynamics.BoxShape([0.1, 0.1, 0.1]))
            targetVisual = targetShape.createVisualAspect()
            targetVisual.setColor([0.8, 0.5, 0.5])
            return target

        target = make_target()
        self.world.addSkeleton(target)

        def make_constraint_box():
            constraint = nimble.dynamics.Skeleton()
            constraint.setName('constraint')  # important for rendering shadows

            constraintJoint, constraintBody = constraint.createWeldJointAndBodyNodePair()
            constraintOffset = nimble.math.Isometry3()
            constraintOffset.set_translation([1.45, 0.85, 0])
            constraintJoint.setTransformFromParentBodyNode(constraintOffset)
            constraintShape = constraintBody.createShapeNode(nimble.dynamics.BoxShape([self.bound_box_size,
                                                                                     self.bound_box_size,
                                                                                     self.bound_box_size]))
            if self.gui:
                constraintVisual = constraintShape.createVisualAspect()
                constraintVisual.setCastShadows(False)
                constraintVisual.setRGBA([0.8, 0.0, 0.0, 0.3])
            return constraint

        if self.use_constraint:
            constraint = make_constraint_box()
            self.world.addSkeleton(constraint)

        # Remove DOFs corresponding to projectile from action space
        self.world.removeDofFromActionSpace(0)
        self.world.removeDofFromActionSpace(1)

    def traj(self, actions, state, return_last_only=True, verbose=False):
        states = []
        actions = actions.reshape(self.MANIP_SEGMENTS, -1)
        if verbose:
            print(actions)
        for i in range(self.NUM_SIM_STEPS):
            time = (i // (self.NUM_SIM_STEPS // self.MANIP_SEGMENTS))
            act = actions[time] * 10
            state = nimble.timestep(self.world, state, act.cpu())
            if not return_last_only:
                states.append(state)
        if return_last_only:
            return state
        return states

    def reset_task(self):
        self.projectile.setPositions(np.array(self.object_position))
        self.projectile.setVelocities(np.array([0.0, 0.0]))

    def compute_bound_violation(self, world_state):
        state = world_state[2:4]
        return torch.clamp(torch.abs(state) - self.bound_box_size / 2, min=0, max=None)

    def state_loss(self, s, manip_loss=False):
        l2_loss = torch.linalg.norm(s[:2] - self.target)
        if self.use_constraint:
            return l2_loss + self.constraint_factor * torch.linalg.norm(self.compute_bound_violation(s))
        else:
            return l2_loss

    def shaped_reward(self, s, tool_position):
        l2_loss = torch.linalg.norm(tool_position[:2] - s[:2])
        return l2_loss

    def traj_loss(self, states, manip_loss=False, tool_position=None):
        state_losses = torch.stack([self.state_loss(s, manip_loss) for s in states])
        if self.loss_fn == 'min':
            return state_losses.min()
        elif self.loss_fn == 'min_scaled':
            return state_losses.min() * 10 # The nimble environment is a 1:10 replication of the Box2D 'hook_push' scene
        elif self.loss_fn == 'min_indicator':
            return state_losses.min() - 10 * torch.any(state_losses < 0.25)
        elif self.loss_fn == 'mean_indicator':
            return state_losses.mean() - 10 * torch.any(state_losses < 0.25)
        elif self.loss_fn == 'mean_indicator_scaled':
            return state_losses.mean() * 10 - 10 * torch.any(state_losses < 0.25)
        elif self.loss_fn == 'mean_indicator_scaled_shaped':
            shaped_rewards = torch.stack([self.shaped_reward(s, tp) for s, tp in zip(states, tool_position)])
            return 5 * shaped_rewards.min() + state_losses.mean() * 10 - 10 * torch.any(state_losses < 0.25)
        elif self.loss_fn == 'track_demo':
            return ((torch.stack(self.demo) - torch.stack(states)) ** 2).mean()
        else:
            raise NotImplementedError

    def get_random_object_position(self):
        # the selected object setting as the center, plus a random offset
        center_pos = np.array(self.OBJECT_SETTINGS[self.object_setting])
        scale = self.random_obj_range
        return center_pos + scale * np.random.uniform(low=-1.0, high=1.0, size=2)

    def reset(self):
        # Randomize object positions once per episode
        if self.randomize_object_position:
            self.object_position = self.get_random_object_position()

    def get_obs(self):
        return self.object_position.copy()