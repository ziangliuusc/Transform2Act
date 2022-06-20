import numpy as np
import cv2
import gym
from gym import spaces
import Box2D
from Box2D import b2

from design_opt.envs.scenes import BOX2D_SCENES as SCENES


class Box2DEnvv2(gym.Env):

    # Env for creating reaching tools with a variable number of rectangles, and a _fixed_ sweeping motor policy.
    NUM_BODIES = 3
    BODY_DEFAULT_POSITIONS = [(15, 10), (18, 10), (19, 10), (20, 10), (21, 10), (22, 10)]
    BOX_SIZES = [(2, 0.25), (2, 0.25), (2, 0.25), (2, 0.25), (2, 0.25), (2, 0.25)]  # length, width
    #assert NUM_BODIES == len(BODY_DEFAULT_POSITIONS) == len(BOX_SIZES)

    NUM_JOINTS = NUM_BODIES * (NUM_BODIES - 1) // 2

    # TODO handle object properties in a cleaner way
    DENSITY = 1
    FRICTION = 0.3
    RESTITUTION = 0.8

    def handle_kwargs(self, kwargs):
        self.penalize_joint_failure = kwargs.get('penalize_joint_fail', False)
        self.NUM_BODIES = kwargs.get('num_bodies', self.NUM_BODIES)
        self.scene_name = kwargs.get('scene', 'hook_push')
        self.return_frames = kwargs.get('return_frames',
                                        False)  # whether the step function should return all frames from simulated traj

    def __init__(self, **kwargs):
        super().__init__()
        self.handle_kwargs(kwargs)
        self.NUM_JOINTS = self.NUM_BODIES * (self.NUM_BODIES - 1) // 2

        # --- pybox2d world setup ---
        # Create the world
        self.world = b2.world(gravity=(0, 0), doSleep=False)

        self.scene = SCENES[self.scene_name](self.world, **kwargs)

        self.action_space = spaces.Dict({
            'object_id': spaces.Discrete(self.NUM_BODIES),
            'joint_target': spaces.Discrete(self.NUM_BODIES),
            'joint_anchor_params': spaces.Box(low=-1.0, high=1.0, shape=(5,)),
        })

        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(6*self.NUM_JOINTS + 3*self.NUM_BODIES + self.scene.obs_dim,), dtype=np.float32)

        self.bodies = []
        self.fixtures = []

        for i in range(self.NUM_BODIES):
            tool_body_i = self.world.CreateDynamicBody(position=self.BODY_DEFAULT_POSITIONS[i], angle=Box2D.b2_pi / 2)
            box_i = tool_body_i.CreatePolygonFixture(box=self.BOX_SIZES[i], density=self.DENSITY,
                                                     friction=self.FRICTION, restitution=self.RESTITUTION)
            box_i.filterData.groupIndex = -2
            tool_body_i.sleepingAllowed = False
            self.fixtures.append(box_i)
            self.bodies.append(tool_body_i)

        # joints grid, self.joints[i][j] is a joint from body i to j where i > j
        self.joints = [[None] * self.NUM_BODIES for _ in range(self.NUM_BODIES)]
        # keeps track of whether each body has been connected using a joint yet
        self.body_joint_mask = [False] * self.NUM_BODIES

    def nearest_distance_to_object(self, i):
        return Box2D.b2Distance(shapeA=self.objectCircle.shape, transformA=self.objectBody.transform,
                                shapeB=self.fixtures[i].shape, transformB=self.bodies[i].transform).distance

    def clip_action_space(self, action):
        if isinstance(action, dict):
            for key in action:
                assert key in self.action_space.spaces.keys(), f'Unknown action key {key}'
                a_space = self.action_space.spaces[key]
                if isinstance(a_space, spaces.Box):
                    action[key] = np.clip(action[key], a_min=a_space.low, a_max=a_space.high)
        else:
            action = np.clip(action, a_min=self.action_space.low, a_max=self.action_space.high)
        return action

    def step(self, action):
        if not isinstance(action, dict):
            action = spaces.unflatten(self.action_space, action)

        action = self.clip_action_space(action)

        target_object_id = action['object_id']
        assert target_object_id < len(self.bodies), \
           f'Target object was {target_object_id} but only {len(self.bodies)} bodies found in environment!'

        # next, determine the joint target
        joint_target_id = action['joint_target']
        joint_target_body = self.bodies[joint_target_id]

        self.bodies[0].angle = self.scene.START_ANGLE
        transform = -self.bodies[0].position + self.bodies[0].GetWorldPoint(
            (self.BOX_SIZES[0][0] - 0.05, 0)) + np.array(self.scene.FINAL_POS)
        self.bodies[0].transform = transform, self.scene.START_ANGLE
        joint_fail = False
        if joint_target_id < target_object_id and not self.body_joint_mask[target_object_id] and self.joints[target_object_id][joint_target_id] is None:  # if a valid joint can be created:
            joint_params = np.array(action['joint_anchor_params'])
            localAnchorA = np.array(self.BOX_SIZES[target_object_id]) * joint_params[:2]
            localAnchorB = np.array(self.BOX_SIZES[joint_target_id]) * joint_params[2:4]
            ref_angle = joint_params[4] * Box2D.b2_pi

            self.bodies[target_object_id].angle = ref_angle + self.bodies[joint_target_id].angle
            self.bodies[target_object_id].position = self.bodies[target_object_id].position - self.bodies[target_object_id].GetWorldPoint(localAnchorA) + \
                                      self.bodies[joint_target_id].GetWorldPoint(localAnchorB)

            self.joints[target_object_id][joint_target_id] = self.world.CreateWeldJoint(bodyA=self.bodies[target_object_id], bodyB=self.bodies[joint_target_id],
                                                    localAnchorA=localAnchorA,
                                                    localAnchorB=localAnchorB, userData={'bodyA_id': target_object_id, 'bodyB_id': joint_target_id},
                                                    collideConnected=False)
            self.body_joint_mask[target_object_id] = True
        else:
            joint_fail = True
        body_states = self.get_all_body_states()
        reward, frames, success = self.scene.run_sim_traj(self.bodies, hinge_pos=(-self.BOX_SIZES[0][0] + 0.05, 0), record_every=(1 if self.return_frames else -1))
        if self.penalize_joint_failure:
            if joint_fail:
                reward -= 10
        self.set_all_body_states(body_states)

        data = {'frames': frames}
        return self.get_obs(), reward, False, data

    def heuristic_reward(self):
        min_dist = 1e10
        for i in range(len(self.bodies)):
            min_dist = min(min_dist, self.nearest_distance_to_object(i))
        return -min_dist

    @staticmethod
    def set_body_state(body, state):
        assert len(state) == 3, 'body state must be length 3!'
        body.position = state[:2]
        body.angle = state[2]

    @staticmethod
    def vectorize_joint_state(joint):
        return np.concatenate([joint.GetLocalAnchorA(), joint.GetLocalAnchorB(), [joint.GetReferenceAngle()]])

    @staticmethod
    def vectorize_body_state(body):
        return np.concatenate([body.position, [body.angle]])

    def get_all_body_states(self):
        return [self.vectorize_body_state(body) for body in self.bodies]

    def set_all_body_states(self, states):
        assert len(self.bodies) == len(states), 'Must have same number of bodies as states to set!'
        for body, state in zip(self.bodies, states):
            self.set_body_state(body, state)

    def get_obs(self):
        state = np.zeros(self.observation_space.shape)
        for idx, body in enumerate(self.bodies):
            state[idx * 3:idx * 3 + 3] = self.vectorize_body_state(body)
        idx = 0
        for i in range(len(self.bodies)):
            for j in range(i+1, len(self.bodies)):
                if self.joints[j][i] is not None:
                    state[idx * 6 + len(self.bodies) * 3] = 1
                    state[idx * 6 + len(self.bodies) * 3 + 1:idx * 6 + len(self.bodies) * 3 + 6] = self.vectorize_joint_state(
                        self.joints[j][i])
                idx += 1
        if self.scene.obs_dim > 0:
            state[-self.scene.obs_dim:] = self.scene.get_obs()
        return state

    def stop_all_bodies(self):
        for body in self.bodies:
            body.linearVelocity = (0, 0)
            body.angularVelocity = 0

    def reset_bodies(self):
        # reset body positions
        for idx, body in enumerate(self.bodies):
            body.position = self.BODY_DEFAULT_POSITIONS[idx]
            body.angle = Box2D.b2_pi / 2
        self.stop_all_bodies()

    def reset(self):
        for joint_list in self.joints:
            for joint in joint_list:
                if joint is not None:
                    self.world.DestroyJoint(joint)
        self.joints = [[None] * self.NUM_BODIES for _ in range(self.NUM_BODIES)]
        self.scene.reset()
        self.reset_bodies()
        self.body_joint_mask = [False] * self.NUM_BODIES
        # delete joint
        return self.get_obs()

    def render(self, mode='human'):
        return self.scene.render()

    def close(self):
        pass


if __name__ == '__main__':
    # env = ReachPushEnv()
    env = gym.make('Tool-ReachPush-v2', return_frames=True, heuristic_reward_weight=1)
    env.reset()
    frames = []

    test_optimal_actions = [
        {
            'object_id': 1,
            'joint_target': 0,
            'joint_anchor_params': [-1.0, -.9989, 1.0, -1, 0]
        },
        {
            'object_id': 2,
            'joint_target': 1,
            'joint_anchor_params': [-1.0, -.9989, 1.0, -1, 0.2],
        },
    ]

    for traj in range(10):
        env.reset()
        for step in range(2):
            act = env.action_space.sample()
            obs, rew, done, info = env.step(act)
            from moviepy.editor import ImageSequenceClip

            clip = ImageSequenceClip(info['frames'], fps=20)
            clip.write_gif(f'random_trajs/random_traj_{traj}_step{step}.gif', fps=20)
            print(rew)

    env.reset()
    for i in range(2):

        obs, rew, done, info = env.step(test_optimal_actions[i])
        from moviepy.editor import ImageSequenceClip

        clip = ImageSequenceClip(info['frames'], fps=20)
        clip.write_gif(f'policy_hook_action{i}.gif', fps=20)
        print(rew)
    exit()

    env.reset()
    for i in range(1000):
        key = 0xFF & cv2.waitKey(int(env.TIME_STEP * 1000))  # milliseconds
        if key == 27:
            break
        env.drawer.clear_screen()
        env.drawer.draw_world(env.world)
        # Make Box2D simulate the physics of our world for one step.
        env.bodies[1].angle += 0.005
        env.world.Step(env.TIME_STEP, 10, 10)
        if i % 30 == 0:
            frames.append(env.render())
        # for contact_edge in env.goalBody.contacts:
        #     print(contact_edge.contact.touching)
        # Flip the screen and try to keep at the target FPS
        # cv2.imshow("world", env.drawer.screen)
    from moviepy.editor import ImageSequenceClip

    clip = ImageSequenceClip(frames, fps=20)
    clip.write_gif(f'angle_test.gif', fps=20)
    print(rew)
    exit()
