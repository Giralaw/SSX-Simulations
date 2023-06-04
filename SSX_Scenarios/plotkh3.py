import numpy as np
import matplotlib.pyplot as plt
import h5py
import time
import sys
import matplotlib.gridspec as gridspec
import plotly.graph_objects as go
import plotly
import plotly.io as pio
pio.renderers.default="png" # disables browser

if(len(sys.argv) != 2):
    print ('usage: plotkh3.py <timestep>')
    sys.exit(0)
else:
    it = int(sys.argv[1])

with h5py.File("analysis_tasks/analysis_tasks_s1.h5", mode='r') as file:
    S = file['tasks']['S']
    t = S.dims[0]['sim_time']
    x = S.dims[1][0]
    y = S.dims[2][0]
    z = S.dims[3][0]

    nx = 96
    ny = 48
    nz = 128

    X, Y, Z = np.mgrid[0:nx, 0:ny, 0:nz]

    fig = go.Figure(data=go.Isosurface(colorbar=dict(title='scalar'),
                                       x=X.flatten(),
                                       y=Y.flatten(),
                                       z=Z.flatten(),
                                       value=S[it,:,:,:].flatten(),
                                       isomin=0.01,
                                       isomax=0.99,
                                       colorscale='jet',
                                       surface_count=5,
                                       opacity=0.3,
                                       showscale=False,
                                       caps=dict(x_show=False,y_show=False,z_show=False)))
    
    camera = dict(
        eye=dict(x=-2.0,y=-1.5,z=1.25)
        )

    fig.update_layout(scene_camera=camera,title='isosurfaces',
                      margin=dict(t=0,l=0,b=0),
                      scene=dict(xaxis=dict(title='streamwise'),
                                 yaxis=dict(title='spanwise'),
                                zaxis=dict(title='vertical')))
    fig.update_layout(scene_aspectmode='manual',
                      scene_aspectratio=dict(x=1,y=1,z=1))
    
    fname = "kh3iso.png"
    fig.write_image(fname)
