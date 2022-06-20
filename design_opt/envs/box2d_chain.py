import numpy as np
import cv2
import gym
from gym import spaces
import Box2D
from Box2D import b2

from design_opt.envs.scenes import BOX2D_SCENES as SCENES
from design_opt.envs.box2d_v2 import Box2DEnvv2


class Box2DChainEnv(Box2DEnvv2):

    CHAIN_MAX_LENGTH = 6
    CHAIN_WIDTH = 0.25

    def handle_kwargs(self, kwargs):
        super().handle_kwargs(kwargs)

    def __init__(self, **kwargs):
        self.handle_kwargs(kwargs)
        self.NUM_JOINTS = self.NUM_BODIES - 1

        # --- pybox2d world setup ---
        # Create the world
        self.world = b2.world(gravity=(0, 0), doSleep=False)

        self.scene = SCENES[self.scene_name](self.world, **kwargs)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.NUM_JOINTS * 2,))

        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(
            6 * self.NUM_JOINTS + 3 * self.NUM_BODIES + self.scene.obs_dim,), dtype=np.float32)

        self.bodies = []
        self.fixtures = []
        self.joints = []
        # keeps track of whether each body has been connected using a joint yet
        self.body_joint_mask = [False] * self.NUM_BODIES

    def get_body_dims(self, body):
        return tuple(map(lambda x: -x, body.fixtures[0].shape.vertices[0]))

    def step(self, action):
        data = dict()
        if not isinstance(action, dict):
            action = spaces.unflatten(self.action_space, action)

        action = self.clip_action_space(action)

        # next, determine the joint target
        link_remaining = self.CHAIN_MAX_LENGTH
        link_lengths = []
        for i in range(self.NUM_JOINTS):
            link_len = link_remaining * min(max((action[i] + 1) / 2, 0.2), 0.8)
            link_lengths.append(link_len)
            link_remaining = link_remaining - link_len
        link_lengths.append(link_remaining)
        link_angles = action[self.NUM_JOINTS:] * Box2D.b2_pi

        self.reset_bodies()

        # create bodies
        for i in range(self.NUM_BODIES):
            tool_body_i = self.world.CreateDynamicBody(position=self.BODY_DEFAULT_POSITIONS[i],
                                                       angle=Box2D.b2_pi / 2)
            box_i = tool_body_i.CreatePolygonFixture(box=(link_lengths[i], self.CHAIN_WIDTH), density=self.DENSITY,
                                                     friction=self.FRICTION)
            box_i.filterData.groupIndex = -2
            self.bodies.append(tool_body_i)
            self.fixtures.append(box_i)

        self.bodies[0].angle = self.scene.START_ANGLE
        transform = -self.bodies[0].position + self.bodies[0].GetWorldPoint(
            (self.get_body_dims(self.bodies[0])[0] - 0.05, 0)) + np.array(self.scene.FINAL_POS)
        self.bodies[0].transform = transform, self.scene.START_ANGLE

        for i in range(self.NUM_JOINTS):
            joint_target_id = i
            target_object_id = i+1
            joint_params = np.array([-1.0, 0, 1.0, 0, link_angles[i]])
            localAnchorA = np.array(self.get_body_dims(self.bodies[target_object_id])) * joint_params[:2]
            localAnchorB = np.array(self.get_body_dims(self.bodies[joint_target_id])) * joint_params[2:4]
            ref_angle = joint_params[4] * Box2D.b2_pi

            self.bodies[target_object_id].angle = ref_angle + self.bodies[joint_target_id].angle
            self.bodies[target_object_id].position = self.bodies[target_object_id].position - self.bodies[target_object_id].GetWorldPoint(localAnchorA) + \
                                                     self.bodies[joint_target_id].GetWorldPoint(localAnchorB)

            weld = self.world.CreateWeldJoint(bodyA=self.bodies[target_object_id], bodyB=self.bodies[joint_target_id],
                                              localAnchorA=localAnchorA,
                                              localAnchorB=localAnchorB,
                                              userData={'bodyA_id': target_object_id, 'bodyB_id': joint_target_id},
                                              collideConnected=False,
                                              frequencyHz=0,
                                              dampingRatio=0,
                                              )
            self.joints.append(weld)
            self.body_joint_mask[target_object_id] = True

        body_states = self.get_all_body_states()
        reward, frames, success = self.scene.run_sim_traj(self.bodies, hinge_pos=(-self.get_body_dims(self.bodies[0])[0] + 0.05, 0), record_every=(1 if self.return_frames else -1))
        data.update({'success': success})
        self.set_all_body_states(body_states)
        data.update({'frames': frames})

        return self.get_obs(), reward, False, data

    def get_obs(self):
        state = np.zeros(self.observation_space.shape)
        for idx, body in enumerate(self.bodies):
            state[idx * 3:idx * 3 + 3] = self.vectorize_body_state(body)
        idx = 0
        for i in range(self.NUM_JOINTS):
            if len(self.joints) > i:
                state[idx * 6 + len(self.bodies) * 3] = 1
                state[idx * 6 + len(self.bodies) * 3 + 1:idx * 6 + len(self.bodies) * 3 + 6] = self.vectorize_joint_state(
                    self.joints[i])
            idx += 1
        if self.scene.obs_dim > 0:
            state[-self.scene.obs_dim:] = self.scene.get_obs()
        return state

    def reset_bodies(self):
        for joint in self.joints:
            self.world.DestroyJoint(joint)
        for i, body in enumerate(self.bodies):
            body.DestroyFixture(self.fixtures[i])
            self.world.DestroyBody(body)
        self.bodies = []
        self.fixtures = []
        self.joints = []

    def reset(self):
        self.reset_bodies()
        self.body_joint_mask = [False] * self.NUM_BODIES
        self.scene.reset()
        # delete joint
        return self.get_obs()


if __name__ == '__main__':
    # env = ReachPushEnv()
    env = gym.make('Tool-Chain-v1', return_frames=True, num_bodies=3, speedup=2)
    env.reset()
    frames = []

    test_optimal_actions = [
        [-1/3, 0, 0, -1/2-0.1]
    ]

    for traj in range(10):
        env.reset()
        for step in range(2):
            act = env.action_space.sample()
            obs, rew, done, info = env.step(test_optimal_actions[0])
            #obs, rew, done, info = env.step(act)
            from moviepy.editor import ImageSequenceClip

            clip = ImageSequenceClip(info['frames'], fps=20)
            clip.write_gif(f'random_trajs/random_traj_{traj}_step{step}.gif', fps=20)
            print(rew)
    exit()
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
