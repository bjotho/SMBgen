import plotly
import plotly.graph_objs as go
import os
from scipy import signal


x = []
y = []
dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "checkpoints")

with open(f"{dir_path}/generator_rewards/merged_rewards", 'r') as file:
    line = file.readline()
    i = 0
    while line != 0:
        try:
            y.append(float(line))
            i += 1
            x.append(i)
        except:
            break

        line = file.readline()


y = signal.savgol_filter(y, 53, 1)

plotly.offline.plot({
    "data": [
        go.Scatter(x=x, y=y, name='Generator Reward')
    ],
    "layout": go.Layout(title="Generator reward", xaxis=dict(title="Episode"), yaxis=dict(title="Episode reward"))
}, auto_open=True)
