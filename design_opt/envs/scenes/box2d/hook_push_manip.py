import numpy as np
import Box2D
from Box2D import b2

from tool_design.envs.scenes.box2d.box2d_scene import Box2DScene
from tool_design.envs.scenes.box2d.utils import *
from tool_design.envs.scenes.box2d.hook_push import HookPush


class HookPushManip(HookPush):

    MANIP_SEGMENTS = 4
    REWARD_SCALE = 10

    def handle_kwargs(self, **kwargs):
        super().handle_kwargs(**kwargs)
        self.motor_scale = kwargs.get('motor_scale', 100)
        self.bound_box_size = kwargs.get('bbox_size', 5.0)
        self.constraint_factor = kwargs.get('constraint_factor', 0.1)

    def __init__(self, world, **kwargs):
        super().__init__(world, **kwargs)

    @property
    def manipulation_action_size(self):
        return self.MANIP_SEGMENTS * 3

    def compute_bound_violation(self, bodies):
        violations = []
        for body in bodies:
            violation = np.clip(np.abs(body.position - np.array(self.FINAL_POS)) - self.bound_box_size / 2, 0, None)
            violations.append(np.linalg.norm(violation)**2)
        min_violation = np.array(violations).min()
        # take minimum violation, because as long as one end of the tool stays in bounds, it's okay
        return self.constraint_factor * min_violation

    def run_sim_traj(self, actions, bodies, hinge_pos, record_every=-1):
        # reset object body position
        self.stop_all_bodies(bodies)
        self.reset_object_bodies()

        # Returns a list of frames containing every 'record_every' frames
        frames = []
        rewards = []
        goals_reached = [False] * len(self.object_bodies)
        #heuristic_reward = -1e10
        heuristic_reward = 0
        for step in range(self.NUM_SIM_STEPS):
            #heuristic_reward = max(heuristic_reward, self.heuristic_reward_weight * self.heuristic_reward())
            idx = int(step // (self.NUM_SIM_STEPS / self.MANIP_SEGMENTS))
            force = actions[idx*3:(idx+1)*3] * self.motor_scale
            for body in bodies:
                body.ApplyForce(force=(float(force[0]), float(force[1])),
                                point=body.position, wake=True)
            bodies[0].ApplyTorque(torque=float(force[2]) * 5, wake=True)
            self.world.Step(self.TIME_STEP, 6, 2)
            self.world.ClearForces()
            if record_every > 0 and step % record_every == 0:
                frames.append(self.render())
            # compute reward (negative l2 object-goal distance) + solving bonus
            l2_reward, success = self.l2_distance_reward()
            for i in range(len(success)):
                if success[i]:
                    goals_reached[i] = True
            reward = l2_reward - self.compute_bound_violation(bodies)
            rewards.append(reward)
        mean_reward = np.array(rewards).mean() + heuristic_reward
        success = all(goals_reached)
        bonus = 10 * np.array(goals_reached).sum()
        self.stop_all_bodies(bodies)
        self.reset_object_bodies()
        return mean_reward + bonus, frames, success

    @property
    def obs_dim(self):
        obs_dim = 2
        return obs_dim

    def get_obs(self):
        obs = [[]]
        obs.append(self.object_position)
        return np.concatenate(obs)