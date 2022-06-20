from tool_design.envs.scenes.box2d.hook_push import HookPush
from tool_design.envs.scenes.box2d.hook_push_manip import HookPushManip
from tool_design.envs.scenes.box2d.utils import get_com
from Box2D import b2


class HookPushBarrier(HookPush):

    OBSTACLE_POSITIONS = [(9, 15, 0), (19.5, 15, 0)]
    OBSTACLE_SIZES = [(3, 0.25), (3, 0.25)]

    OBJECT_POSITION = (10.5, 18)
    GOAL_POSITION = (5, 18)

    START_ANGLE = b2.pi / 2
    FINAL_POS = (15, 8.5)

    def handle_kwargs(self, **kwargs):
        # this scene always has obstacles
        kwargs['obstacles'] = True
        super().handle_kwargs(**kwargs)

    def __init__(self, world, **kwargs):
        super().__init__(world, **kwargs)

    def start_fixed_traj(self, bodies, hinge_pos):
        point = get_com(bodies)
        bodies[0].ApplyLinearImpulse(impulse=(-30, 0), point=point, wake=True)

    def end_fixed_traj(self):
        pass


class HookPushBarrierManip(HookPushBarrier, HookPushManip):
    pass


class HookPushBarrierBringCloseManip(HookPushBarrierManip):
    GOAL_POSITION = (15, 12)


if __name__ == '__main__':
    import gym
    #env = gym.make('Tool-ReachPush-v2', return_frames=True, heuristic_reward_weight=1, scene='hook_push_barrier')
    #env = gym.make('Tool-ImgPolicyVariancesFixed-v1', debug=True, return_frames=True, heuristic_reward_weight=1, scene='hook_push_barrier', y_scale_factor=0.3)
    env = gym.make('Tool-Chain-v1', return_frames=True, heuristic_reward_weight=1, scene='hook_push_barrier')
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

