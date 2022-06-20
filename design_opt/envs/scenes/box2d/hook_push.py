import numpy as np
import Box2D
from Box2D import b2

from tool_design.envs.scenes.box2d.box2d_scene import Box2DScene
from tool_design.envs.scenes.box2d.utils import *


class HookPush(Box2DScene):

    OBJECT_POSITION = (10, 18)
    GOAL_POSITION = (3, 10)
    OBJECT_2_POSITION = (12, 14)
    GOAL_2_POSITION = (10, 10)

    OBJECT_SETTINGS = [
        [10, 18],
        [10, 20],
        [8, 20],
    ]

    OBSTACLE_POSITIONS = [(7, 18, b2.pi / 2)]
    OBSTACLE_SIZES = [(3, 0.25)]

    def handle_kwargs(self, **kwargs):
        super().handle_kwargs(**kwargs)
        self.randomize_object_position = kwargs.get('randomize_obj_position', False)
        print('Randomize object position: ', self.randomize_object_position)
        self.heuristic_reward_weight = kwargs.get('heuristic_reward_weight', 0)
        self.random_obj_range = kwargs.get('random_obj_range', -7.0)
        self.use_obstacles = kwargs.get('obstacles', False)
        self.RESTITUTION = kwargs.get('restitution', self.RESTITUTION)
        self.object_setting = kwargs.get('object_setting', 0)

        self.two_balls = kwargs.get('two_balls', False)
        self.second_object_weight = kwargs.get('second_object_weight', 1.0)
        self.move_start_right = kwargs.get('move_start_right', False)

    def __init__(self, world, **kwargs):
        super().__init__(world, **kwargs)

        if self.move_start_right:
            self.FINAL_POS = (17.5, 7.5)

        self.create_ground_body()
        self.create_obstacles()
        # Create the hinge joint which will be used for the scripted trajectory
        self.hingeBody = self.world.CreateStaticBody(position=(0, 0), angle=0)
        self.obstacleBodies = self.create_obstacles()

        self.indicator = self.world.CreateStaticBody(position=self.FINAL_POS)
        self.indicatorCircle = self.indicator.CreateCircleFixture(radius=0.5, density=1, friction=0.3, isSensor=True)
        self.indicatorCircle.userData = {'color': Color.ORANGE}

        self.goal_bodies, self.object_bodies = self.create_objects_goals()

        for body in [self.hingeBody, self.indicator] + self.goal_bodies + self.object_bodies:
            body.sleepingAllowed = False

    def close(self):
        super().close()
        for body in [self.hingeBody, self.indicator] + self.goal_bodies + self.object_bodies:
            self.world.DestroyBody(body)

    def create_objects_goals(self):

        goal_bodies, object_bodies = [], []
        # Create target indicator
        goalBody = self.world.CreateStaticBody(position=self.GOAL_POSITION)
        goalCircle = goalBody.CreateCircleFixture(radius=1.5, density=1, friction=0.3, isSensor=True)
        goalCircle.userData = {'color': Color.LIGHT_GREEN}

        print(f'----------------------------')
        print(f'Setting object setting to {self.object_setting}')
        print(f'----------------------------')

        self.object_position = self.OBJECT_SETTINGS[self.object_setting]
        # Create object of interest
        objectBody = self.world.CreateDynamicBody(position=self.object_position)
        objectCircle = objectBody.CreateCircleFixture(radius=1, density=1, friction=0.3,
                                                                restitution=self.RESTITUTION)
        objectCircle.userData = {'color': Color.BLUE}

        goal_bodies = [goalBody]
        object_bodies = [objectBody]

        if self.two_balls:
            goalBody2 = self.world.CreateStaticBody(position=self.GOAL_2_POSITION)
            goalCircle2 = goalBody2.CreateCircleFixture(radius=1.5, density=1, friction=0.3, isSensor=True)
            goalCircle2.userData = {'color': Color.PINK}

            self.object2_position = self.OBJECT_2_POSITION
            # Create object of interest
            objectBody2 = self.world.CreateDynamicBody(position=self.object2_position)
            objectCircle2 = objectBody2.CreateCircleFixture(radius=1, density=1, friction=0.3)
            objectCircle2.userData = {'color': Color.PURPLE}

            goal_bodies.append(goalBody2)
            object_bodies.append(objectBody2)

        return goal_bodies, object_bodies

    def create_ground_body(self):
        # And a static body to hold the ground shape
        ground_body = self.world.CreateStaticBody(
            position=(0, 0),
            shapes=b2.polygonShape(box=(50, 1)),
        )

    def get_random_object_position(self):
        random_offset = np.random.uniform(low=self.random_obj_range, high=0)
        object_position = list(self.OBJECT_POSITION)
        object_position[1] += random_offset
        return object_position

    def reset_object_bodies(self):
        self.object_bodies[0].position = self.object_position
        if self.two_balls:
            self.object_bodies[1].position = self.object2_position

    def create_obstacles(self):
        if not self.use_obstacles:
            return []
        else:
            bodies = []
            assert len(self.OBSTACLE_POSITIONS) == len(self.OBSTACLE_SIZES), \
                'Number of obstacle positions and sizes must be equal'
            for obstacle_pos, size in zip(self.OBSTACLE_POSITIONS, self.OBSTACLE_SIZES):
                bodies.append(self.world.CreateStaticBody(position=obstacle_pos[:2], angle=obstacle_pos[2]))
                fixture = bodies[-1].CreatePolygonFixture(box=size, density=self.DENSITY, friction=self.FRICTION,
                                                          restitution=self.RESTITUTION)
                fixture.userData = {'color': Color.RED}
            return bodies

    def stop_all_bodies(self, bodies):
        for body in bodies + self.object_bodies + self.goal_bodies:
            body.linearVelocity = (0, 0)
            body.angularVelocity = 0

    def run_sim_traj(self, actions, bodies, hinge_pos, record_every=-1):
        # reset object body position
        self.stop_all_bodies(bodies)
        self.reset_object_bodies()

        # Returns a list of frames containing every 'record_every' frames
        hinge_joint = self.start_fixed_traj(bodies, hinge_pos)
        frames = []
        rewards = []
        goals_reached = [False] * len(self.object_bodies)
        #heuristic_reward = -1e10
        heuristic_reward = 0
        for step in range(self.NUM_SIM_STEPS):
            #heuristic_reward = max(heuristic_reward, self.heuristic_reward_weight * self.heuristic_reward())
            self.world.Step(self.TIME_STEP, 6, 2)
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
        success = all(goals_reached)
        bonus = 10 * np.array(goals_reached).sum()
        self.stop_all_bodies(bodies)
        self.end_fixed_traj()
        return mean_reward + bonus, frames, success

    def l2_distance_reward(self):
        l2_norms, success = [], []
        for i, (object_body, goal_body) in enumerate(zip(self.object_bodies, self.goal_bodies)):
            l2_norm = np.linalg.norm(np.array(goal_body.position) - np.array(object_body.position))
            success.append(1 if l2_norm < 2.5 else 0)
            if i == 1:
                l2_norm *= self.second_object_weight
            # if self.second_object_weight > 1 and i == 0:
            #     l2_norm = 0
            l2_norms.append(l2_norm)
        return -sum(l2_norms) * self.REWARD_SCALE, success

    def start_fixed_traj(self, bodies, hinge_pos):
        body = bodies[0]
        self.hingeBody.position = body.GetWorldPoint(hinge_pos)
        self.hingeBody.angle = 0
        self.hinge_joint = self.world.CreateRevoluteJoint(
            bodyA=body,
            bodyB=self.hingeBody,
            anchor=self.hingeBody.worldCenter,
            lowerAngle=-0.9 * Box2D.b2_pi,
            upperAngle=0.1 * Box2D.b2_pi,
            enableLimit=True,
            maxMotorTorque=1000.0,
            motorSpeed=-1000.0,
            enableMotor=True,
        )

    def end_fixed_traj(self):
        self.world.DestroyJoint(self.hinge_joint)

    def reset(self):
        if self.randomize_object_position:
            self.object_position = self.get_random_object_position()
        self.reset_object_bodies()

    @property
    def obs_dim(self):
        obs_dim = 0
        if self.randomize_object_position:
            obs_dim += 2
        return obs_dim

    def get_obs(self):
        obs = [[]]
        if self.randomize_object_position:
            obs.append(self.object_position)
        return np.concatenate(obs)



if __name__ == '__main__':
    import gym
    env = gym.make('Tool-ImgPolicyVariancesFixed-v1', debug=True, two_balls=True,
                   return_frames=True, scene='hook_push', y_scale_factor=0.3)
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
            #obs, rew, done, info = env.step(test_long_stick_actions[step])
            obs, rew, done, info = env.step(act)
            from moviepy.editor import ImageSequenceClip

            clip = ImageSequenceClip(info['frames'], fps=20)
            clip.write_gif(f'random_trajs/random_traj_{traj}_step{step}.gif', fps=20)
            print(rew)

