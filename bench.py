import subprocess
import pandas as pd
import numpy as np

from devito import div, grad, TimeFunction, Operator, configuration, Eq, solve
from devito import configuration
from examples.seismic import Model
from examples.seismic import Receiver
from examples.seismic import TimeAxis

import numpy as np
import time


cpu_plat = configuration['platform'].name

def update(conf, mode):
    if mode == 'cpu':
        conf['compiler'] = 'gcc-9'
        conf['platform'] = cpu_plat
        conf['language'] = 'openmp'
    else:
        conf['compiler'] = 'nvc'
        conf['platform'] = 'nvidiaX'
        conf['language'] = 'openacc'

nd = [2, 3]
vp_modes = ['constant', 'array']
sizes = {('kwavecpu', 2): [64, 128, 256, 512], ('kwavegpu', 2): [64, 128, 256, 512],
         ('kwavecpu', 3): [64, 128, 256, 512], ('kwavegpu', 3): [64, 128, 256, 512]}
numrun = 3

nb_size = 10
so = 8
x_kwave = 1e-3
medium_vel = 1500



hdw = {'kwavecpu': "cpu", 'kwavegpu': "gpu"}
bench_k = pd.DataFrame(columns=["hdw", "shape", "mean", "std"])
bench_d = pd.DataFrame(columns=["hdw", "shape", "mean", "std"])
bench_ds = pd.DataFrame(columns=["hdw", "shape", "mean", "std"])

def bench_devito(ndim, grid_size, hdw, numrun, vtype, scale=1.25):
    grid_size = int(scale * grid_size)
    update(configuration, hdw)
    new_grid_size = (grid_size - 2*nb_size)
    shape   = tuple([new_grid_size]*ndim)
    spacing = tuple([x_kwave/new_grid_size]*ndim)
    origin  = tuple([0.]*ndim)

    v   = medium_vel
    b = 1
    if vtype == "array":
        v = v*np.ones(shape)
        v[..., 0:shape[1]//2] = medium_vel*1.5
        b = 1/(v - .5)
    
    model = Model(vp=v, b=b, origin=origin, shape=shape, spacing=spacing,
                  space_order=so, nbl=nb_size, bcs="damp")

    t0 = 0.   
    dt = model.critical_dt
    tn = scale*150*dt
    time_range  = TimeAxis(start=t0, stop=tn, step=dt)
    
    #make smoothed source with blackman filter
    point_src = np.zeros(model.grid.shape)
    point_src[[grid_size//2]*ndim] = 1
    total_time = 0

    #MAKE A TIME FUNCTION FOR ACOUSTIC PRESSURE
    u = TimeFunction(name='u', grid=model.grid, space_order=so, time_order=2)
    u.data[0][:] = point_src # field at t0
    u.data[1][:] = point_src # field at t0

    # Create symbol for receivers
    rec = Receiver(name='rec', grid=model.grid, npoint=1, time_range=time_range)

    #get location in model space
    model_loc = tuple([grid_size//2]*ndim) #np.where(file_kwave_2d['sensor_mask'] == 1)

    # Put receiver 10 gridpoints away from center point source
    for i in range(ndim):
        rec.coordinates.data[:,i]  = model_loc[i]*model.spacing[i]

    # Create interpolation expression for receivers
    rec_term = rec.interpolate(expr=u)
    pde_u = model.m * model.b * u.dt2 - div(model.b * grad(u)) + model.damp * u.dt

    # The PDE representation is as on paper
    wave_stencil = Eq(u.forward, solve(pde_u, u.forward))

    op_wave = Operator([wave_stencil] + rec_term, subs=model.spacing_map)
    # precompile
    op_wave(dt=np.float32(dt), time_M=1)
    # Run and time
    rt = np.zeros(numrun)
    for i in range(numrun):    
        start = time.process_time()
        op_wave(dt=np.float32(dt))
        end = time.process_time()
        rt[i] = (end - start)
    mt, st = np.mean(rt), np.std(rt)
    print(f"Devito: ({grid_size}, {ndim}, {vtype}, {hdw}): Runtime is {mt:.4f} with std={st:.4f}")
    return {"hdw": hdw, "shape": (grid_size, ndim), "mean": mt, "std": st}


def bench_kwave(ndim, s, numrun, prog, vtype):
    shape = 'x'.join([str(s) for _ in range(ndim)])
    rt = np.zeros(numrun)
    for i in range(numrun):
        p = subprocess.check_output([prog, '-i', f'kwave_setups/{vtype}_{shape}.h5', '-o', 'test'])
        rt[i] = float(p.decode('utf-8').split('\n')[-5].split('s')[-2].split(' ')[-1])
    mt, st = np.mean(rt), np.std(rt)
    print(f"Kwave : ({s}, {ndim}, {vtype}, {hdw[prog]}): Runtime is {mt:.4f} with std={st:.4f}")
    return {"hdw": hdw[prog], "shape": (s, ndim), "mean": mt, "std": st}

for ndim in nd:
    for prog in ['kwavegpu']:#, 'kwavecpu']:
        for s in sizes[(prog, ndim)]:
            for vtype in vp_modes:
                try:
                    b_k = bench_kwave(ndim, s, numrun, prog, vtype)
                    bench_k = bench_k.append(b_k, ignore_index=True)
                except:
                    pass

                #try:
                b_d = bench_devito(ndim, s, hdw[prog], numrun, vtype)
                bench_d = bench_d.append(b_d, ignore_index=True)
                #except:
                #    pass

                try:
                    b_ds = bench_devito(ndim, s, hdw[prog], numrun, vtype, scale=1)
                    bench_ds = bench_ds.append(b_ds, ignore_index=True)
                except:
                    pass
 
                print("")
                try:
                    speedup = b_k["mean"]/b_d["mean"]
                    print(f"Speedup between devito and kwave = {speedup:.3f}")
                except:
                    pass
                
                try:
                    speedups = b_k["mean"]/b_ds["mean"]
                    print(f"Speedup between devito and kwave same size = {speedups:.3f} \n")
                except:
                    pass


bench_ds.to_pickle('bench_devito_same.pkl')
bench_d.to_pickle('bench_devito.pkl')
bench_k.to_pickle('bench_kwave.pkl')
