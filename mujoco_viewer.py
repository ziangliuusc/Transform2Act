
"""
Shows how to use render callback.
"""
import argparse
from mujoco_py import load_model_from_path, MjSim, MjViewer
from mujoco_py.modder import TextureModder
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot', default="hopper")
    args = parser.parse_args()
    model = load_model_from_path("assets/mujoco_envs/" + args.robot + ".xml")
    sim = MjSim(model)

    viewer = MjViewer(sim)

    t = 0

    while True:
        viewer.render()
        t += 1
        if t > 100 and os.getenv('TESTING') is not None:
            break
        
if __name__ == "__main__":
    main()