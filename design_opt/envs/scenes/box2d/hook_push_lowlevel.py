import numpy as np
import Box2D

from tool_design.envs.scenes.box2d.hook_push import HookPush


class HookPushLL(HookPush):

    def run_sim_traj(self, bodies, hinge_pos, record_every=-1):
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
        torques = np.array([0.5841, 2.5699]) * -1000
        for step in range(self.NUM_SIM_STEPS):
            #heuristic_reward = max(heuristic_reward, self.heuristic_reward_weight * self.heuristic_reward())
            torque = torques[step//40]
            self.hinge_joint.motorSpeed = 1e5 * (-1 if torque < 0 else 1)
            self.hinge_joint.maxMotorTorque = abs(torque)
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


if __name__ == '__main__':
    import gym
    env = gym.make('Tool-ReachPush-v2', return_frames=True, scene='hook_push_ll')
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
            #act = env.action_space.sample()
            #obs, rew, done, info = env.step(test_long_stick_actions[step])
            obs, rew, done, info = env.step(test_optimal_actions[step])
            from moviepy.editor import ImageSequenceClip

            clip = ImageSequenceClip(info['frames'], fps=20)
            clip.write_gif(f'random_trajs/random_traj_{traj}_step{step}.gif', fps=20)
            print(rew)

