import plotly.graph_objs as go
import numpy as np 
import torch

def get_unit_cam():
    f = 10
    unit_cam = torch.tensor([
        [0,0,0],
        [3,2,f],
        [3,-2,f],
        [-3,-2,f],
        [-3,2,f],
        [0,4,f]
    ])

    seq = torch.tensor([3,4,1,2,0,1,5,4,0,3,2])
    draw_cam = unit_cam[seq]

    return draw_cam.type(torch.float64)


# draw_cam = get_unit_cam()
# print(draw_cam)
# fig = go.Figure(data=go.Scatter3d(x=draw_cam[:,0], y=draw_cam[:,1], z=draw_cam[:,2], mode='lines'))
# fig.show() 