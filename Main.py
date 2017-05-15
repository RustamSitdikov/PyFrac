# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on Fri Dec 23 17:49:21 2016.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2017. All rights reserved.
See the LICENSE.TXT file for more details.
"""


# adding src folder to the path
import sys
if "win" in sys.platform:
    slash = "\\"
else:
    slash = "/"
if not '.' + slash + 'src' in sys.path:
    sys.path.append('.' + slash + 'src')

# imports
import numpy as np
from src.CartesianMesh import *
from src.Fracture import *
from src.LevelSet import *
from src.VolIntegral import *
from src.Elasticity import *
from src.Properties import *
from src.FractureFrontLoop import *

# creating mesh
Mesh = CartesianMesh(6, 6, 41, 41)

# solid properties
nu = 0.4
Eprime = 3.3e10 / (1 - nu ** 2)
K_Ic = 0.005e6
sigma0 = 0 * 1e6
Solid = MaterialProperties(Eprime, K_Ic, 0., sigma0, Mesh, grain_size=1e-5)

# injection parameters
Q0 = 0.09  # injection rate
well_location = np.array([0., 0.])
Injection = InjectionProperties(Q0, well_location, Mesh)

# fluid properties
Fluid = FluidProperties(1.1e-3, Mesh, turbulence=True)

# simulation properties
simulProp = SimulationParameters(tip_asymptote="T",
                                 output_time_period=0.05,
                                 plot_figure=True,
                                 save_to_disk=False,
                                 out_file_address=".\\Data\\TurbRoughInit2",
                                 plot_analytical=True,
                                 cfl_factor=0.4,
                                 analytical_sol="TR")


# initializing fracture
initRad = 3 # initial radius of fracture
Fr = Fracture(Mesh, Fluid, Solid) # create fracture object
Fr.initialize_radial_Fracture(initRad,
                              'radius',
                              'TR',
                              Solid,
                              Fluid,
                              Injection,
                              simulProp) # initializing

Fr.plot_fracture("complete", "footPrint", identify=np.asarray([839,840,768]))
plt.show()

# elasticity matrix
C = load_elasticity_matrix(Mesh, Solid.Eprime)

# starting time stepping loop
MaximumTimeSteps = 1000
i = 0
Tend = 1000.
Fr_k = copy.deepcopy(Fr)

while (Fr.time < Tend) and (i < MaximumTimeSteps):

    i = i + 1

    print('\n*********************\ntime = ' + repr(Fr.time))

    TimeStep = simulProp.CFLfactor * min(Fr.mesh.hx, Fr.mesh.hy) / np.max(Fr.v)
    status, Fr_k = attempt_time_step(Fr_k, C, Solid, Fluid, simulProp, Injection, TimeStep)

    # Fr.plot_fracture("complete", "width")
    # plt.show()
    Fr = copy.deepcopy(Fr_k)
    print("volume in numerical " + repr(sum(Fr.w) * Fr.mesh.EltArea) + " volume injected " + repr(
        Injection.injectionRate[1, 0] * Fr.time))

