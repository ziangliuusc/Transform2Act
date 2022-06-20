import numpy as np
from Box2D import b2

from tool_design.envs.scenes.box2d.box2d_scene import Box2DScene
from tool_design.envs.scenes.box2d.utils import *


class Contain(Box2DScene):

    BALL_POSITIONS = [(16, 10), (14, 10), (15, 9), (15, 11)]
    FINAL_POS = (15, 7)
    BALL_CENTER = (15, 10)
    START_ANGLE = b2.pi / 2

    def handle_kwargs(self, **kwargs):
        self.randomize_object_position = kwargs.get('randomize_obj_position', False)
        super().handle_kwargs(**kwargs)

    def __init__(self, world, **kwargs):
        super().__init__(world, **kwargs)

        #self.create_ground_body()
        # Create the body anchor which will be used to hold the tool
        self.hingeBody = self.world.CreateStaticBody(position=(0, 0), angle=0)

        self.indicator = self.world.CreateStaticBody(position=self.BALL_CENTER)
        self.indicatorCircle = self.indicator.CreateCircleFixture(radius=0.5, density=1, friction=0.3, isSensor=True)
        self.indicatorCircle.userData = {'color': Color.ORANGE}

        self.goal_bodies, self.object_bodies = self.create_objects_goals()

        for body in self.all_bodies:
            body.sleepingAllowed = False

    @property
    def all_bodies(self):
        return [self.hingeBody, self.indicator] + \
                self.goal_bodies + self.object_bodies

    def close(self):
        super().close()
        for body in self.all_bodies:
            self.world.DestroyBody(body)

    def create_objects_goals(self):
        # Create target indicator
        goalBody = self.world.CreateStaticBody(position=self.BALL_CENTER)
        goalCircle = goalBody.CreateCircleFixture(radius=1.5, density=1, friction=0.3, isSensor=True)
        goalCircle.userData = {'color': Color.LIGHT_GREEN}

        object_bodies = []

        # Create object of interest
        for pos in self.BALL_POSITIONS:
            objectBody = self.world.CreateDynamicBody(position=pos)
            objectCircle = objectBody.CreateCircleFixture(radius=0.7, density=1, friction=0.3,
                                                          restitution=self.RESTITUTION)
            objectCircle.userData = {'color': Color.BLUE}
            object_bodies.append(objectBody)

        goal_bodies = [goalBody]

        return goal_bodies, object_bodies

    def create_ground_body(self):
        # And a static body to hold the ground shape
        ground_body = self.world.CreateStaticBody(
            position=(0, 0),
            shapes=b2.polygonShape(box=(50, 1)),
        )

    def reset_object_bodies(self):
        for pos, body in zip(self.BALL_POSITIONS, self.object_bodies):
            body.position = pos

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
        for i, object_body in enumerate(self.object_bodies):
            l2_norm = np.linalg.norm(np.array(self.BALL_CENTER) - np.array(object_body.position))
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
        directions = np.array([(-1, 1), (-1, -1), (1, 1), (1, -1)]) * 50
        for body in self.object_bodies:
            direction = np.random.randint(0, 4)
            body.ApplyLinearImpulse(impulse=(float(directions[direction][0]), float(directions[direction][1])), point=body.worldCenter, wake=True)

    def end_fixed_traj(self):
        self.world.DestroyJoint(self.hinge_joint)

    def reset(self):
        if self.randomize_object_position:
            self.goal_position = self.get_random_goal_pos()
        self.reset_object_bodies()

    @property
    def obs_dim(self):
        obs_dim = 0
        return obs_dim

    def get_obs(self):
        obs = [[]]
        return np.concatenate(obs)


if __name__ == '__main__':
    import gym

    env = gym.make('Tool-ImgPolicyVariancesFixed-v1', scene='bounce', return_frames=True,
                   x_scale_factor=0.1, y_scale_factor=0.1, randomize_obj_position=True)
    env.reset()
    frames = []

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
