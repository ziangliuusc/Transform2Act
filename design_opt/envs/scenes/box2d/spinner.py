import numpy as np
from Box2D import b2

from tool_design.envs.scenes.box2d.box2d_scene import Box2DScene
from tool_design.envs.scenes.box2d.utils import *
import gc

class Spinner(Box2DScene):

    SPINNER_POS = [(6, 12), (16, 12), (11, 5)]
    SPOKE_SIZE = (3, 0.25)
    FINAL_POS = (11, 9.5)
    START_ANGLE = b2.pi / 2

    DENSITY = 2

    def handle_kwargs(self, **kwargs):
        super().handle_kwargs(**kwargs)
        self.num_spokes = kwargs.get('spokes', 6)
        self.num_spinners = kwargs.get('num_spinners', 3)
        self.randomize_spoke_number = kwargs.get('randomize_obj_position', False)

    def __init__(self, world, **kwargs):
        super().__init__(world, **kwargs)

        #self.create_ground_body()
        # Create the body anchor which will be used to hold the tool
        self.hingeBody = self.world.CreateStaticBody(position=(0, 0), angle=0)

        self.spinner_centers = []
        self.spinner_bodies, self.spinner_joints, self.spinner_body_states = [], [], []
        for i in range(self.num_spinners):
            spinnerCenter = self.world.CreateDynamicBody(position=self.SPINNER_POS[i], angle=0)
            self.spinner_centers.append(spinnerCenter)
            # spoke box is necessary!
            spoke_box = spinnerCenter.CreatePolygonFixture(box=(0.25, 0.25), density=self.DENSITY,
                                                                friction=self.FRICTION)
            spoke_box.filterData.groupIndex = -2
            spinnerPivot = self.world.CreateStaticBody(position=self.SPINNER_POS[i], angle=0)
            spinner_joint = self.world.CreateRevoluteJoint(
                bodyA=spinnerCenter,
                bodyB=spinnerPivot,
                anchor=spinnerPivot.worldCenter,
                collideConnected=False,
                enableMotor=True,
                maxMotorTorque=100.0,
                motorSpeed = 0.0,

            )
        self.create_spinners()

        self.indicator = self.world.CreateStaticBody(position=self.FINAL_POS)
        self.indicatorCircle = self.indicator.CreateCircleFixture(radius=0.5, density=1, friction=0.3, isSensor=True)
        self.indicatorCircle.userData = {'color': Color.ORANGE}

        for body in self.all_bodies:
            body.sleepingAllowed = False

    @property
    def all_bodies(self):
        return [self.hingeBody, self.indicator] + self.spinner_centers + sum(self.spinner_bodies, [])

    def reset_spinner(self):
        for spinner_bodies, spinner_body_states in zip(self.spinner_bodies, self.spinner_body_states):
            for b, state in zip(spinner_bodies, spinner_body_states):
                self.set_body_state(b, state)

    def destroy_spinners(self):
        for spinner_bodies, spinner_joints in zip(self.spinner_bodies, self.spinner_joints):
            for j in spinner_joints:
                self.world.DestroyJoint(j)
            for b in spinner_bodies:
                b.DestroyFixture(b.fixtures[0])
                self.world.DestroyBody(b)
            del spinner_joints, spinner_bodies
        del self.spinner_body_states, self.spinner_joints, self.spinner_bodies
        gc.collect()
        self.spinner_joints, self.spinner_bodies, self.spinner_body_states = [], [], []

    def create_spinners(self):
        for spinnerCenter in self.spinner_centers:
            spinner_bodies, spinner_joints, body_states = self.create_spinner(spinnerCenter)
            self.spinner_bodies.append(spinner_bodies)
            self.spinner_joints.append(spinner_joints)
            self.spinner_body_states.append(body_states)

    def create_spinner(self, spinnerCenter):
        spokes, joints = [], []
        for i in range(self.num_spokes // 2):
            angle = i / self.num_spokes * 2*b2.pi
            spoke = self.world.CreateDynamicBody(position=spinnerCenter.position, angle=angle)
            spoke_box = spoke.CreatePolygonFixture(box=self.SPOKE_SIZE, density=self.DENSITY, friction=self.FRICTION)
            spoke_box.filterData.groupIndex = -2
            if i == 0:
                j = self.world.CreateWeldJoint(bodyA=spoke, bodyB=spinnerCenter,
                                           localAnchorA=[0, 0],
                                           localAnchorB=[0, 0],
                                           collideConnected=False)
            else:
                j = self.world.CreateWeldJoint(bodyA=spoke, bodyB=spokes[-1],
                                           localAnchorA=[0, 0],
                                           localAnchorB=[0, 0],
                                           collideConnected=False)
            spokes.append(spoke)
            joints.append(j)

        spinner_body_states = [self.vectorize_body_state(b) for b in spokes]
        return spokes, joints, spinner_body_states

    def close(self):
        super().close()
        for body in self.all_bodies:
            self.world.DestroyBody(body)

    def create_ground_body(self):
        # And a static body to hold the ground shape
        ground_body = self.world.CreateStaticBody(
            position=(0, 0),
            shapes=b2.polygonShape(box=(50, 1)),
        )

    def reset_object_bodies(self):
        self.reset_spinner()

    def stop_all_bodies(self, bodies):
        for body in self.all_bodies:
            body.linearVelocity = (0, 0)
            body.angularVelocity = 0

    def run_sim_traj(self, bodies, hinge_pos, record_every=-1):
        # reset object body position
        self.stop_all_bodies(bodies)
        self.reset_object_bodies()

        # Returns a list of frames containing every 'record_every' frames
        self.start_fixed_traj(bodies, hinge_pos)
        frames = []
        rewards = []
        goal_reached = False
        # heuristic_reward = -1e10
        heuristic_reward = 0
        for step in range(self.NUM_SIM_STEPS):
            # heuristic_reward = max(heuristic_reward, self.heuristic_reward_weight * self.heuristic_reward())
            self.world.Step(self.TIME_STEP, 10, 10)
            if record_every > 0 and step % record_every == 0:
                frames.append(self.render())
            # compute reward (negative l2 object-goal distance) + solving bonus
            l2_reward, success = self.angular_vel_reward()
            if success:
                goal_reached = True
            reward = l2_reward
            rewards.append(reward)
        mean_reward = np.array(rewards).mean() + heuristic_reward
        bonus = 10 * np.array(goal_reached).sum()
        success = all([goal_reached])
        self.stop_all_bodies(bodies)
        self.end_fixed_traj()
        return mean_reward + bonus, frames, success

    def angular_vel_reward(self):
        # TODO MAKE BETTER
        angular_vels = []
        for center in self.spinner_centers:
            angular_vel = -center.angularVelocity / 10.0
            angular_vels.append(angular_vel)
        success = (1 if all([a > 1 for a in angular_vels]) else 0)
        return np.array(angular_vels).mean() * self.REWARD_SCALE, success

    def start_fixed_traj(self, bodies, hinge_pos):
        body = bodies[0]
        self.hingeBody.position = body.GetWorldPoint(hinge_pos)
        self.hingeBody.angle = 0
        self.hinge_joint = self.world.CreateRevoluteJoint(
            bodyA=body,
            bodyB=self.hingeBody,
            anchor=self.hingeBody.worldCenter,
            enableLimit=False,
            maxMotorTorque=1000.0,
            motorSpeed=-1000.0,
            enableMotor=True,
        )

    def end_fixed_traj(self):
        self.world.DestroyJoint(self.hinge_joint)

    def reset(self):
        self.reset_spinner()
        if self.randomize_spoke_number:
            self.destroy_spinners()
            self.num_spokes = np.random.randint(1, 5) * 2
            self.create_spinners()

    @property
    def obs_dim(self):
        obs_dim = 0
        if self.randomize_spoke_number:
            obs_dim += 1
        return obs_dim

    def get_obs(self):
        obs = [[]]
        if self.randomize_spoke_number:
            obs.append([self.num_spokes])
        return np.concatenate(obs)


if __name__ == '__main__':
    import gym

    env = gym.make('Tool-ImgPolicyVariancesFixed-v1', scene='spinner', return_frames=True,
                   x_scale_factor=0.4, y_scale_factor=0.4, randomize_obj_position=True)
    env.reset()
    frames = []

    # test_optimal_actions = [
    #     {
    #         'object_id': 1,
    #         'joint_target': 0,
    #         'joint_anchor_params': [-1.0, -.9989, 1.0, -1, 0]
    #     },
    #     {
    #         'object_id': 2,
    #         'joint_target': 1,
    #         'joint_anchor_params': [-1.0, -.9989, 1.0, -1, 0.2],
    #     },
    # ]
    # test_long_stick_actions = [
    #     {
    #         'object_id': 1,
    #         'joint_target': 0,
    #         'joint_anchor_params': [-1.0, -.9989, 1.0, -1, 0]
    #     },
    #     {
    #         'object_id': 2,
    #         'joint_target': 1,
    #         'joint_anchor_params': [-1.0, -.9989, 1.0, -1, 0],
    #     },
    # ]

    for traj in range(1000):
        env.reset()
        for step in range(2):
            act = env.action_space.sample()
            # obs, rew, done, info = env.step(test_long_stick_actions[step])
            obs, rew, done, info = env.step(act)
            # from moviepy.editor import ImageSequenceClip

            # clip = ImageSequenceClip(info['frames'], fps=20)
            # clip.write_gif(f'random_trajs/random_traj_{traj}_step{step}.gif', fps=20)
            # print(rew)
