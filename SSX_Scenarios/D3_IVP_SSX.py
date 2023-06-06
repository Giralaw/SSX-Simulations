"""SSX_model_A.py

This is the *simplest* model we will consider for modeling spheromaks evolving in the SSX wind tunnel.

Major simplifications fall in two categories

Geometry
--------
We consider a square duct using parity bases (sin/cos) in all directions. (RealFourier in D3)

Equations
---------
The equations themselves are those from Schaffner et al (2014), with the following simplifications

* hall term off
* constant eta instead of Spitzer
* no wall recycling term
* no mass diffusion

For this first model, rather than kinematic viscosity nu and thermal
diffusivity chi varying with density rho as they should, we are here
holding them *constant*. This dramatically simplifies the form of the
equations in Dedalus.

We use the vector potential, and enforce the Coulomb Gauge, div(A) = 0.

File formerly called D3_SSX_A_2_spheromaks
- when looking for older versions, check both current name and that name.


Dedalus 3 edits made by Alex Skeldon. Direct all queries to askeldo1@swarthmore.edu (prior to June 2024).
"""

import time
import numpy as np
import os
import sys
import dedalus.public as d3
from dedalus.extras import flow_tools


from D3_two_spheromaks import spheromak_pair, parity

import logging
logger = logging.getLogger(__name__)


# for optimal efficiency: nx should be divisible by mesh[0], ny by mesh[1], and
# nx should be close to ny. Bridges nodes have 128 cores, so mesh[0]*mesh[1]
# should be a multiple of 128.
nx = 16 #formerly 32 x 32 x 160? Current plan is 64 x 64 x 320 or 640
ny = 16
nz = 160 # try power of two for nz? e.g. 512?
r = 1
length = 10

# for 3D runs, you can divide the work up over two dimensions (x and y).
# The product of the two elements of mesh *must* equal the number
# of cores used.
mesh = [2,2]
#mesh = [16,16]
data_dir = "scratch" #change each time or overwrite

kappa = 0.01
mu = 0.05 #Determines Re_k ; 0.05 -> Re_k = 20 (try 0.005?)
eta = 0.001 # Determines Re_m ; 0.001 -> Re_m = 1000
rho0 = 1 #rho0 is redefined later, and generally has a whole
gamma = 5./3.
eta_sp = 2.7 * 10**(-4)
eta_ch = 4.4 * 10**(-3)
v0_ch = 2.9 * 10**(-2)
chi = kappa/rho0
nu = mu/rho0
#diffusivities for heat (kappa -> chi), momentum (viscosity) (mu -> nu), current (eta)
# life time of currents regulated by resistivity
# linearization time of temperature goes like e^-t/kappa

#Coords, dist, bases
coords = d3.CartesianCoordinates('x', 'y','z')
dist = d3.Distributor(coords, dtype=np.float64, mesh = mesh)

xbasis = d3.RealFourier(coords['x'], size=nx, bounds=(-r, r))
ybasis = d3.RealFourier(coords['y'], size=ny, bounds=(-r, r))
zbasis = d3.RealFourier(coords['z'], size=nz, bounds=(0, length))

# Fields
t = dist.Field(name='t')
v = dist.VectorField(coords, name='v', bases=(xbasis, ybasis, zbasis))
A = dist.VectorField(coords, name='A', bases=(xbasis, ybasis, zbasis))
lnrho = dist.Field(name='lnrho', bases=(xbasis, ybasis, zbasis))
T = dist.Field(name='T', bases=(xbasis, ybasis, zbasis))
phi = dist.Field(name='phi', bases=(xbasis, ybasis, zbasis))
tau_A = dist.Field(name='tau_A')
# ex, ey, ez = coords.unit_vector_fields(dist)

# Coulomb Gauge implies J = -Laplacian(A)
# j = dist.VectorField(coords, name ='j', bases = (xbasis, ybasis, zbasis))
j = -d3.lap(A)
J2 = j@j
rho = np.exp(lnrho)
B = d3.curl(A)

# CFL substitutions
Va = B/np.sqrt(rho)
Cs = np.sqrt(gamma*T)

#Problem
SSX = d3.IVP([v, A, lnrho, T, phi, tau_A], time=t, namespace=locals())

#resistivity
#SSX.add_equation("eta1 = eta_sp/(sqrt(T)**3) + (eta_ch/sqrt(rho))*(1 - exp((-v0_ch*sqrt(J2))/(3*rho*sqrt(gamma*T))))")

# Continuity
SSX.add_equation("dt(lnrho) + div(v) = - v@grad(lnrho)")

# Momentum
SSX.add_equation("dt(v) + grad(T) - nu*lap(v) = T*grad(lnrho) - v@grad(v) + cross(j,B)/rho")

# MHD equations: A
SSX.add_equation("dt(A) + grad(phi) = - eta*j + cross(v,B)")

#these two should replace the commented equations below?
SSX.add_equation("div(A) + tau_A = 0")
SSX.add_equation("integ(phi) = 0")

# Energy
SSX.add_equation("dt(T) - (gamma - 1) * chi*lap(T) = - (gamma - 1) * T * div(v)  - v@grad(T) + (gamma - 1)*eta*J2")

solver = SSX.build_solver(d3.RK222) #switch to RK222? (formerly 443)

logger.info("Solver built")

# Initial timestep
dt = 1e-4

# Integration parameters
solver.stop_sim_time = 20 #historically 20
solver.stop_wall_time = np.inf #e.g. 60*60*3 would limit runtime to three hours
solver.stop_iteration = np.inf

x,y,z = dist.local_grids(xbasis,ybasis,zbasis)
fullGrid = x*y*z

# Initial condition parameters
R = r
L = R
lambda_rho = 0.4 # half-width of transition region for initial conditions
lambda_rho1 = 0.1
rho_min = 0.011
T0 = 0.1
delta = 0.1 # The strength of the perturbation. PSST 2014 has delta = 0.1 .

# Spheromak initial condition
# The vector potential is subject to some perturbation. This distorts all the magnetic field components in the same direction.
aa = spheromak_pair(xbasis,ybasis,zbasis, coords, dist)
for i in range(3):
    A['g'][i] = aa['g'][i] *(1 + delta*x*np.exp(-z**2) + delta*x*np.exp(-(z-10)**2))


# Frame for meta params in D3 with RealFourier
# (need to get multi-basis syntax for specifying coeff values in dir prods down first).

A = parity(A,0)
v = parity(v,1)
T = parity(T,0,scalar=True)
lnrho = parity(lnrho,0,scalar=True)
phi = parity(phi,1,scalar=True)


#initial velocity
max_vel = 0.1
##vz['g'] = -np.tanh(6*z - 6)*max_vel/2 + -np.tanh(6*z - 54)*max_vel/2

# Maybe write your versions of how you think these hardcodings should go,
# then ask jeff to have a call just to specifically Discuss this issue
#should always use local grid - never loop over things like this - talk to Jeff about in future call.
for i in range(x.shape[0]):
    xVal = x[i,0,0]
    for j in range(y.shape[1]):
       yVal = y[0,j,0]
       for k in range(z.shape[2]):
           zVal = z[0,0,k]
           fullGrid[i][j][k] = -np.tanh(6*zVal-6)*(1-rho_min)/2 -np.tanh(6*(10-zVal)-6)*(1-rho_min)/2 + 1 #density in the z direction with tanh transition

#ignoring this section for now
##########################################################################################################################################
#--------------------------------------density in the z direction with cosine transisition ----------------------------------------------#
##########################################################################################################################################
#           if((zVal <= 1 - lambda_rho or zVal >= 9 + lambda_rho)):
#               fullGrid[i][j][k] = 1
#           elif((zVal >= 1 - lambda_rho and zVal <= 1 + lambda_rho)):
#               fullGrid[i][j][k] = (1 + rho_min)/2 + (1 - rho_min)/2*np.sin((1-zVal) * np.pi/(2*lambda_rho))
#           elif(zVal <= 9 + lambda_rho and zVal >= 9 - lambda_rho):
#               fullGrid[i][j][k] = (1 + rho_min)/2 + (1 - rho_min)/2*np.sin((zVal - 9) * np.pi/(2*lambda_rho))
#           else:
#               fullGrid[i][j][k] = rho_min

##########################################################################################################################################
#-------------------------------enforcing circular crosssection of density---------------------------------------------------------------#
##########################################################################################################################################

for i in range(x.shape[0]):
    xVal = x[i,0,0]
    for j in range(y.shape[1]):
       yVal = y[0,j,0]
       for k in range(z.shape[2]):
            zVal = z[0,0,k]
            r = np.sqrt(xVal**2 + yVal**2)
##fullGrid[i][j][k] = np.tanh(40*r+40)*(fullGrid[i][j][k]-rho_min)/2 + np.tanh(40*(1-r))*(fullGrid[i][j][k]-rho_min)/2 + rho_min #tanh transistion

##########################################################################################################################################
#-----------------------------------------------sinusodial transition-----------------------------------------------------------------------------#
##########################################################################################################################################
            if(r <= 1 - lambda_rho1):
               fullGrid[i][j][k] = fullGrid[i][j][k]
            elif((r >= 1 - lambda_rho1 and r <= 1 + lambda_rho1)):
               fullGrid[i][j][k] = (fullGrid[i][j][k] + rho_min)/2 + (fullGrid[i][j][k] - rho_min)/2*np.sin((1-r) * np.pi/(2*lambda_rho1))
            else:
               fullGrid[i][j][k] = rho_min

#figure out what the whole deal with rho0 is - also def as const at start
#probably better way to rewrite this without the rho0 field
rho0 = dist.Field(name='rho0', bases=(xbasis, ybasis, zbasis))
rho0['g'] = fullGrid

lnrho['g'] = np.log(rho0['g'])
T['g'] = T0 * rho0['g']**(gamma - 1)

##eta1['g'] = eta_sp/(np.sqrt(T['g'])**3 + (eta_ch/np.sqrt(rho0['g']))*(1 - np.exp((-v0_ch)/(3*rho0['g']*np.sqrt(gamma*T['g']))))

# analysis output
##data_dir = './'+sys.argv[0].split('.py')[0]
wall_dt_checkpoints = 2
output_cadence = 0.1 # This is in simulation time units

fh_mode = 'overwrite'

# load state for restart - does it matter where to put it?
# also, does the virtual file work for restarting
# solver.load_state("scratch/load_data_two/load_data_two_s1.h5")
# solver.load_state("scratch/load_data_two/load_data_two_s1/load_data_two_s1_p1.h5")

#handle data output dirs
# I'm realizing the else statement doesn't necessarily work so well for the Bridges job submitting scheme...
if dist.comm.rank == 0:
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    # else:
    #     ow = input("this directory already exists. Would you like to overwrite it? (y/n) ")
    #     if ow == 'n':
    #         name = input("what would you like to name the new directory? ('n' to cancel script) ")
    #         if name == 'n':
    #             print("please press ctrl-c.")
    #             quit()
    #         else:
    #             os.mkdir(name)

# Will revisit tomorrow
checkpoint = solver.evaluator.add_file_handler(os.path.join(data_dir,'checkpoints2'), max_writes=10, wall_dt=wall_dt_checkpoints, mode = fh_mode)
checkpoint.add_tasks(solver.state)


field_writes = solver.evaluator.add_file_handler(os.path.join(data_dir,'fields_two'), max_writes = 500, sim_dt = output_cadence, mode = fh_mode)
# trying to just put j for third one yields issues - because j not variable in problem?
field_writes.add_task(v)
field_writes.add_task(B, name = 'B')
field_writes.add_task(d3.curl(B), name='j')
#Supposed to enforce positive rho, but still seeing negative numbers in h5 reader
field_writes.add_task(np.exp(lnrho), name = 'rho')
field_writes.add_task(T)
# field_writes.add_task(eta1)

# complaint about floats not having a dtype - can comment this out, but is probably nice
# to have parameters in h5 file with rest of scenario
# parameter_writes = solver.evaluator.add_file_handler(os.path.join(data_dir,'parameters_two'), max_writes = 1, sim_dt = output_cadence, mode = fh_mode)
# parameter_writes.add_task(mu)
# parameter_writes.add_task(eta)
# parameter_writes.add_task(nu)
# parameter_writes.add_task(chi)
# parameter_writes.add_task(gamma)

#is there a way to make it so that sim_dt = solver.stop_sim_time/max_writes?
load_writes = solver.evaluator.add_file_handler(os.path.join(data_dir,'load_data_two'), max_writes = 50, sim_dt = 4*output_cadence, mode = fh_mode)
load_writes.add_task(v)
load_writes.add_task(A)
load_writes.add_task(lnrho)
load_writes.add_task(T)
load_writes.add_task(phi)
load_writes.add_task(tau_A)

# Helicity
helicity_writes = solver.evaluator.add_file_handler(os.path.join(data_dir,'helicity'), max_writes=500, sim_dt = 2*output_cadence, mode=fh_mode)
helicity_writes.add_task(d3.integ(A@B), name="total_helicity")
helicity_writes.add_task(A@B, name="helicity_at_pos")

# Flow properties
flow = flow_tools.GlobalFlowProperty(solver, cadence = 1)
flow.add_property(np.sqrt(v@v) / nu, name = 'Re_k')
flow.add_property(np.sqrt(v@v) / eta, name = 'Re_m')
flow.add_property(np.sqrt(v@v), name = 'flow_speed')
flow.add_property(np.sqrt(v@v) / np.sqrt(T), name = 'Ma') # Mach number; T going negative?
flow.add_property(np.sqrt(B@B) / np.sqrt(rho), name = 'Al_v')
flow.add_property(T, name = 'temp')

char_time = 1. # this should be set to a characteristic time in the problem (the alfven crossing time of the tube, for example)
CFL_safety = 0.3
CFL = flow_tools.CFL(solver, initial_dt = dt, cadence = 1, safety = CFL_safety,
                     max_change = 1.5, min_change = 0.005, max_dt = output_cadence, threshold = 0.05)
CFL.add_velocity(v)
CFL.add_velocity(Va)
#not sure how to turn Cs into a vector; or if that's still something that we ought to be doing
#CFL.add_velocity(Cs)
#CFL.add_velocities(( 'Cs', 'Cs', 'Cs'))

good_solution = True
# Main loop
try:
    logger.info('Starting loop')
    logger_string = 'kappa: {:.3g}, mu: {:.3g}, eta: {:.3g}, dt: {:.3g}'.format(kappa, mu, eta, dt)
    logger.info(logger_string)
    while solver.proceed:

        dt = CFL.compute_timestep()
        solver.step(dt)

        # enforce parities for appropriate dynamical variables at each timestep to prevent non-zero buildup
        A = parity(A,0)
        v = parity(v,1)
        T = parity(T,0,scalar=True)
        lnrho = parity(lnrho,0,scalar=True)
        phi = parity(phi,1,scalar=True)
            
        if (solver.iteration-1) % 1 == 0:
            logger_string = 'iter: {:d}, t/tb: {:.2e}, dt/tb: {:.2e}, sim_time: {:.4e}, dt: {:.2e}'.format(solver.iteration, solver.sim_time/char_time, dt/char_time, solver.sim_time, dt)
            ##logger_string += 'min_rho: {:.4e}'.format(lnrho['g'].min())
            Re_k_avg = flow.grid_average('Re_k')
            Re_m_avg = flow.grid_average('Re_m')
            v_avg = flow.grid_average('flow_speed')
            Al_v_avg = flow.grid_average('Al_v')
            logger_string += ' Max Re_k = {:.2g}, Avg Re_k = {:.2g}, Max Re_m = {:.2g}, Avg Re_m = {:.2g}, Max vel = {:.2g}, Avg vel = {:.2g}, Max alf vel = {:.2g}, Avg alf vel = {:.2g}, Max Ma = {:.1g}'.format(flow.max('Re_k'), Re_k_avg, flow.max('Re_m'),Re_m_avg, flow.max('flow_speed'), v_avg, flow.max('Al_v'), Al_v_avg, flow.max('Ma'))
            logger.info(logger_string)

            np.clip(lnrho['g'], -4.9, 2, out=lnrho['g'])
            np.clip(T['g'], 0.001, 1000, out=T['g'])
            np.clip(v['g'], -100, 100, out=v['g'])
            ##np.clip(lnrho['g'], -4.9, 2, out=lnrho['g'])

            if not np.isfinite(Re_k_avg):
                good_solution = False
                logger.info("Terminating run.  Trapped on Reynolds = {}".format(Re_k_avg))
            if not np.isfinite(Re_m_avg):
                good_solution = False
                logger.info("Terminating run. Trapped on magnetic Reynolds = {}".format(Re_m_avg))

except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()
