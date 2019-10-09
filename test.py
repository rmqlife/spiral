import matplotlib.pyplot as plt

import spiral.agents.default as default_agent
import spiral.agents.utils as agent_utils
import spiral.environments.libmypaint as libmypaint

import numpy as np

# The path to a TF-Hub module.
MODULE_PATH = "https://tfhub.dev/deepmind/spiral/default-wgangp-celebahq64-gen-19steps/agent4/1"
# The folder containing `libmypaint` brushes.
BRUSHES_PATH = "/mnt/c/Users/rmqli/spiral/third_party/mypaint-brushes-1.3.0/"

# Here, we create an environment.
env = libmypaint.LibMyPaint(episode_length=20,
                            canvas_width=64,
                            grid_width=32,
                            brush_type="classic/pen",
                            brush_sizes=[1, 2, 4, 8, 12, 24],
                            use_color=True,
                            use_pressure=True,
                            use_alpha=False,
                            background="white",
                            brushes_basedir=BRUSHES_PATH)


# Now we load the agent from a snapshot.
initial_state, step = agent_utils.get_module_wrappers(MODULE_PATH)

# Everything is ready for sampling.
state = initial_state()
noise_sample = np.random.normal(size=(10,)).astype(np.float32)

time_step = env.reset()
for t in range(10):
    # time_step = env.reset()
    time_step.observation["noise_sample"] = noise_sample
    action, state = step(time_step.step_type, time_step.observation, state)
    print(action)
    time_step = env.step(action)
    canvas = time_step.observation["canvas"]
    print('canvas', canvas.shape, np.min(canvas), np.max(canvas))
#     plt.imshow(time_step.observation["canvas"], interpolation="nearest")
#     plt.show()
# Show the sample.
# plt.close("all")
# plt.imshow(time_step.observation["canvas"], interpolation="nearest")