from Box2D import b2Vec2


class Color:
    # COLORS (BGR format)
    LIGHT_GREEN = (144, 238, 144)
    PINK = (255, 204, 229)
    BLUE = (0, 165, 255)
    PURPLE = (127, 0, 255)
    RED = (255, 0, 0)
    ORANGE = (232, 152, 23)


def get_com(bodies):
    # get all bodies which are connected to body 0
    connected = [0]
    while True:
        change = False
        for i, body in enumerate(bodies):
            if i not in connected:
                for joint in body.joints:
                    if joint.joint.userData['bodyA_id'] in connected or joint.joint.userData['bodyB_id'] in connected:
                        connected.append(i)
                        change = True
        if not change:
            break
    # compute COM of connected bodies
    bodies = [bodies[i] for i in connected]

    mass_sum = sum([b.mass for b in bodies])
    return sum([b.worldCenter * b.mass for b in bodies], b2Vec2()) / mass_sum