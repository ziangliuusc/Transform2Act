import numpy as np
from Box2D import b2

from tool_design.envs.scenes.box2d.box2d_scene import Box2DScene
from tool_design.envs.scenes.box2d.utils import *


class Wedge(Box2DScene):
    LEVER_POSITION = (10, 5)
    LEVER_SIZE = (5, 0.25)
    BALL_1_POS = (6, 6)
    #BALL_2_POS = (14, 12)
    BALL_2_POS = (11, 12)
    GOAL_POSITION = (6, 15)
    FINAL_POS = (10, 3)
    START_ANGLE = b2.pi / 2

    def handle_kwargs(self, **kwargs):
        self.randomize_object_position = kwargs.get('randomize_obj_position', False)
        super().handle_kwargs(**kwargs)

    def __init__(self, world, **kwargs):
        super().__init__(world, **kwargs)

        self.ball_2_position = self.BALL_2_POS
        self.ball_1_position = self.BALL_1_POS
        #self.create_ground_body()
        # Create the body anchor which will be used to hold the tool
        self.hingeBody = self.world.CreateStaticBody(position=(0, 0), angle=0)

        self.indicator = self.world.CreateStaticBody(position=self.FINAL_POS)
        self.indicatorCircle = self.indicator.CreateCircleFixture(radius=0.5, density=1, friction=0.3, isSensor=True)
        self.indicatorCircle.userData = {'color': Color.ORANGE}

        self.goal_bodies, self.object_bodies = self.create_objects_goals()

        self.obj2Body = self.world.CreateDynamicBody(position=self.ball_2_position)
        obj2Circle = self.obj2Body.CreateCircleFixture(radius=1, density=1, friction=0.3,
                                                       restitution=self.RESTITUTION)
        obj2Circle.userData = {'color': Color.PURPLE}

        self.leverBody = self.world.CreateDynamicBody(position=self.LEVER_POSITION, angle=0)
        leverFixture = self.leverBody.CreatePolygonFixture(box=self.LEVER_SIZE, density=self.DENSITY,
                                                           friction=self.FRICTION)

        for body in self.all_bodies:
            body.sleepingAllowed = False

    @property
    def all_bodies(self):
        return [self.hingeBody, self.indicator, self.obj2Body, self.leverBody] + \
                self.goal_bodies + self.object_bodies

    def close(self):
        super().close()
        for body in self.all_bodies:
            self.world.DestroyBody(body)

    def create_objects_goals(self):
        # Create target indicator
        goalBody = self.world.CreateStaticBody(position=self.GOAL_POSITION)
        goalCircle = goalBody.CreateCircleFixture(radius=1.5, density=1, friction=0.3, isSensor=True)
        goalCircle.userData = {'color': Color.LIGHT_GREEN}

        # Create object of interest
        objectBody = self.world.CreateDynamicBody(position=self.ball_1_position)
        objectCircle = objectBody.CreateCircleFixture(radius=1, density=1, friction=0.3,
                                                      restitution=self.RESTITUTION)
        objectCircle.userData = {'color': Color.BLUE}

        goal_bodies = [goalBody]
        object_bodies = [objectBody]

        return goal_bodies, object_bodies

    def create_ground_body(self):
        # And a static body to hold the ground shape
        ground_body = self.world.CreateStaticBody(
            position=(0, 0),
            shapes=b2.polygonShape(box=(50, 1)),
        )

    def reset_object_bodies(self):
        self.object_bodies[0].position = self.ball_1_position
        self.obj2Body.position = self.ball_2_position
        self.leverBody.angle = 0
        self.leverBody.position = self.LEVER_POSITION

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
        goals_reached = [False] * len(self.object_bodies)
        # heuristic_reward = -1e10
        heuristic_reward = 0
        for step in range(self.NUM_SIM_STEPS):
            # heuristic_reward = max(heuristic_reward, self.heuristic_reward_weight * self.heuristic_reward())
            self.world.Step(self.TIME_STEP, 10, 10)
            if record_every > 0 and step % record_every == 0:
                frames.append(self.render())
            # compute reward (negative l2 object-goal distance) + solving bonus
            l2_reward, success = self.l2_distance_reward()
            for i in range(len(success)):
                if success[i]:
                    goals_reached[i] = True
            reward = l2_reward
            rewards.append(reward)
        mean_reward = np.array(rewards).mean() + heuristic_reward
        bonus = 10 * np.array(goals_reached).sum()
        success = all(goals_reached)
        self.stop_all_bodies(bodies)
        self.end_fixed_traj()
        return mean_reward + bonus, frames, success

    def l2_distance_reward(self):
        l2_norms, success = [], []
        for i, (object_body, goal_body) in enumerate(zip(self.object_bodies, self.goal_bodies)):
            l2_norm = np.linalg.norm(np.array(goal_body.position) - np.array(object_body.position))
            l2_norms.append(l2_norm)
            success.append(1 if l2_norm < 2.5 else 0)
        return -sum(l2_norms) * self.REWARD_SCALE, success

    def start_fixed_traj(self, bodies, hinge_pos):
        body = bodies[0]
        self.hingeBody.position = body.GetWorldPoint(hinge_pos)
        self.hingeBody.angle = 0
        self.hinge_joint = self.world.CreateWeldJoint(
            bodyA=body,
            bodyB=self.hingeBody,
            anchor=self.hingeBody.worldCenter,
            collideConnected=False,
        )
        # apply a force to the ball
        self.obj2Body.ApplyLinearImpulse(impulse=(0, -50), point=self.obj2Body.worldCenter, wake=True)

    def end_fixed_traj(self):
        self.world.DestroyJoint(self.hinge_joint)

    def get_random_object_position(self):
        random_pos = np.random.uniform(low=self.LEVER_POSITION[0], high=self.LEVER_POSITION[0] + self.LEVER_SIZE[0])
        return tuple([random_pos, self.BALL_2_POS[1]])

    def get_random_obj_ball_position(self):
        random_pos = np.random.uniform(low=self.LEVER_POSITION[0] - self.LEVER_SIZE[0], high=self.LEVER_POSITION[0])
        return tuple([random_pos, self.BALL_1_POS[1]])

    def reset(self):
        if self.randomize_object_position:
            self.ball_2_position = self.get_random_object_position()
            self.ball_1_position = self.get_random_obj_ball_position()
        self.reset_object_bodies()

    @property
    def obs_dim(self):
        obs_dim = 0
        if self.randomize_object_position:
            obs_dim += 4
        return obs_dim

    def get_obs(self):
        obs = [[]]
        if self.randomize_object_position:
            obs.append(self.ball_2_position)
            obs.append(self.ball_1_position)
        return np.concatenate(obs)


if __name__ == '__main__':
    import gym

    env = gym.make('Tool-ImgPolicyVariancesFixed-v1', scene='wedge', return_frames=True,
                   x_scale_factor=0.1, y_scale_factor=0.1, randomize_obj_position=True)
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

    for traj in range(10):
        env.reset()
        for step in range(2):
            act = env.action_space.sample()
            # obs, rew, done, info = env.step(test_long_stick_actions[step])
            obs, rew, done, info = env.step(act)
            from moviepy.editor import ImageSequenceClip

            clip = ImageSequenceClip(info['frames'], fps=20)
            clip.write_gif(f'random_trajs/random_traj_{traj}_step{step}.gif', fps=20)
            print(rew)
