from tool_design.envs.scenes.box2d.hook_push import HookPush
from tool_design.envs.scenes.box2d.hook_push_lowlevel import HookPushLL
from tool_design.envs.scenes.box2d.hook_push_manip import HookPushManip
from tool_design.envs.scenes.box2d.hook_push_barrier import HookPushBarrier, HookPushBarrierManip, HookPushBarrierBringCloseManip
from tool_design.envs.scenes.box2d.wedge import Wedge
from tool_design.envs.scenes.box2d.spinner import Spinner
from tool_design.envs.scenes.box2d.bounce import Bounce
from tool_design.envs.scenes.box2d.contain import Contain

from tool_design.envs.scenes.nimble.hook_push import HookPushNimble
from tool_design.envs.scenes.nimble.hook_push_barrier import HookPushBarrierNimble

BOX2D_SCENES = {
    'hook_push': HookPush,
    'hook_push_ll': HookPushLL,
    'hook_push_manip': HookPushManip,
    'hook_push_barrier': HookPushBarrier,
    'hook_push_barrier_manip': HookPushBarrierManip,
    'hook_push_barrier_bc_manip': HookPushBarrierBringCloseManip,
    'wedge': Wedge,
    'spinner': Spinner,
    'bounce': Bounce,
    'contain': Contain,
}


NIMBLE_SCENES = {
    'hook_push': HookPushNimble,
    'hook_push_barrier': HookPushBarrierNimble,
}