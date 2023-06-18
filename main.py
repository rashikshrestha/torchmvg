import numpy as np
import utils
import visualize
import torch
import plotly.graph_objs as go

pose1 = [0.11925659138672644, -0.26208388422867585, 0.03292307004664281, 0.9570820103299055, 1396703734.457429, -69135952.68710543, 783954463.3436801]
pose2 = [0.166102243328895, -0.34677252889507937, -0.3958056235426467, 0.8339644874554532, 1724796789.1439164, -481237824.022731, 1665158883.0118961]

pose1 = np.array(pose1)
pose2 = np.array(pose2)

q1, t1 = pose1[:4], pose1[4:]
q2, t2 = pose2[:4], pose2[4:]

print(q1,t1,q2,t2)

R1 = utils.qvec2rotmat(q1)
R2 = utils.qvec2rotmat(q2)

print(R1)
print(R2)

#! COnvert to tensors
R1 = torch.as_tensor(R1)
R2 = torch.as_tensor(R2)
t1 = torch.as_tensor(t1)
t2 = torch.as_tensor(t2)

print(t1.T.shape, t2.shape)

unit_cam = visualize.get_unit_cam()*100000000
print(R1.dtype, unit_cam.dtype)
cam1 = (torch.matmul(R1,unit_cam.T) + t1.reshape(3,1)).T
cam2 = (torch.matmul(R2,unit_cam.T) + t2.reshape(3,1)).T

print(cam1.shape)

cam1_plot = go.Scatter3d(x=cam1[:,0], y=cam1[:,1], z=cam1[:,2], mode='lines')
cam2_plot = go.Scatter3d(x=cam2[:,0], y=cam2[:,1], z=cam2[:,2], mode='lines')

# fig = go.Figure()
# fig.add_scatter3d(cam1_plot)
# fig.add_scatter3d(cam2_plot)
fig = go.Figure(data=[cam1_plot, cam2_plot])
fig.show() 

