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
Mesh = CartesianMesh(40, 40, 61, 61)

# solid properties
nu = 0.4
Eprime = 3.3e10 / (1 - nu ** 2)
K_Ic = 0.3e6*np.ones((Mesh.NumberOfElts,),dtype=np.float64)
# highKIC = np.where(abs(Mesh.CenterCoor[:,1])>10)[0]
# K_Ic[highKIC] = 4e6
# highKIC = np.where(abs(Mesh.CenterCoor[:,1])>20)[0]
# K_Ic[highKIC] = 0.3e6
sigma0 = 0 * 1e6
Solid = MaterialProperties(Eprime, K_Ic, 0., sigma0, Mesh)

# injection parameters
Q0 = 0.09  # injection rate
well_location = np.array([0., 0.])
Injection = InjectionProperties(Q0, well_location, Mesh)

# fluid properties
Fluid = FluidProperties(1.1e-3, Mesh, turbulence=False)

# simulation properties
simulProp = SimulationParameters(tip_asymptote="U",
                                 output_time_period=5,
                                 plot_figure=True,
                                 save_to_disk=False,
                                 out_file_address=".\\Data\\TurbLamTough",
                                 plot_analytical=True,
                                 cfl_factor=0.5,
                                 analytical_sol="M")


# initializing fracture
initRad = 6.5 # initial radius of fracture
Fr = Fracture(Mesh, Fluid, Solid) # create fracture object
Fr.initialize_radial_Fracture(initRad,
                              'radius',
                              'M',
                              Solid,
                              Fluid,
                              Injection,
                              simulProp) # initializing


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

    TimeStep = simulProp.CFLfactor * Fr.mesh.hx / np.mean(Fr.v)
    status, Fr_k = attempt_time_step(Fr_k, C, Solid, Fluid, simulProp, Injection, TimeStep)
    # Fr_k.plot_fracture("complete","width")
    Fr = copy.deepcopy(Fr_k)


