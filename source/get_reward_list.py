import os
from collections import deque
try:
    import cPickle as pickle
except ImportError:
    import pickle

from source import constants as c

dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "checkpoints")
os.makedirs(os.path.join(f"{dir_path}", "generator_rewards"), exist_ok=True)

write_all = True
merge = True
model = 556

rewards = []
models = [f"model_{str(model)}"]
if write_all:
    models = os.listdir(os.path.join(dir_path, "generator"))

for m in models:
    replay_memory = deque(maxlen=c.REPLAY_MEMORY_SIZE)
    try:
        pickle_file = os.path.join(dir_path, "generator", m, f"{c.REP_MEM}.pickle")
        with open(pickle_file, 'rb') as file:
            replay_memory = pickle.load(file)
    except:
        print("Unpickle unsuccessful for", m)
        continue

    if write_all and merge:
        for transition in replay_memory:
            # The third element (index 2) of each transition cotains reward
            rewards.append(transition[2])
    else:
        with open(f"{dir_path}/generator_rewards/rewards_{m}", 'w') as file:
            for transition in replay_memory:
                file.write(str(transition[2]) + "\n")

    print("Extracted rewards from", m)

if write_all and merge:
    with open(f"{dir_path}/generator_rewards/merged_rewards", 'w') as file:
        for reward in rewards:
            file.write(str(reward) + "\n")
