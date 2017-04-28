#
# This file is part of PyFrac.
#
# Created by Brice Lecampion on 03.04.17.
# Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory, 2016-2017.  All rights reserved.
# See the LICENSE.TXT file for more details. 
#

from src.VolIntegral import *
from src.Utility import *
from src.TipInversion import *
from src.ElastoHydrodynamicSolver import *
from src.LevelSet import *
from src.HFAnalyticalSolutions import *
import copy

errorMessages = ("Propagated not attempted",
                 "Time step successful",
                 "Evaluated level set is not valid",
                 "Front is not tracked correctly",
                 "Evaluated tip volume is not valid",
                 "Solution obtained from the elastohydrodynamic solver is not valid",
                 "Did not converge after max iterations",
                 "Tip inversion is not correct",
                 "Ribbon element not found in the enclosure of the tip cell",
                 "Filling fraction not correct"
                 )


def attempt_time_step(Frac, C, Material_properties, Fluid_properties, Simulation_Parameters, Injection_Parameters,
                      TimeStep):
    """
    This function advances the fracture by the given time step. In case of failure, reattempts are made with smaller time
    steps. A system exit is raised after maximum allowed reattempts. 
    Arguments:
        Frac (Fracture object):                             fracture object from the last time step 
        C (ndarray-float):                                  the elasticity matrix 
        Material_properties (MaterialProperties object):    material properties
        Fluid_properties (FluidProperties object):          fluid properties 
        Simulation_Parameters (SimulationParameters object): simulation parameters
        Injection_Parameters (InjectionProperties object):  injection properties
        TimeStep (float):                                   time step to be attempted 
    
    Return:
        int:   possible values:
                                    0       -- not propagated
                                    1       -- iteration successful
                                    2       -- evaluated level set is not valid
                                    3       -- front is not tracked correctly
                                    4       -- evaluated tip volume is not valid
                                    5       -- solution of elastohydrodynamic solver is not valid
                                    6       -- did not converge after max iterations
                                    7       -- tip inversion not successful
                                    8       -- Ribbon element not found in the enclosure of a tip cell
                                    9       -- Filling fraction not correct
                                    
        Fracture object:            fracture after advancing time step. 
    """
    print("Attempting time step of " + repr(TimeStep) + " sec...")
    # loop for reattempting time stepping in case of failure.
    for i in range(0, Simulation_Parameters.maxReattempts):
        # smaller time step to reattempt time stepping; equal to the given time step on first iteration
        smallerTimeStep = TimeStep * Simulation_Parameters.reAttemptFactor ** i

        status, Fr = FractureFrontLoop(Frac,
                                       C,
                                       Material_properties,
                                       Fluid_properties,
                                       Simulation_Parameters,
                                       Injection_Parameters,
                                       smallerTimeStep)
        if status == 1:
            print(errorMessages[status])

            # output
            if Simulation_Parameters.plotFigure or Simulation_Parameters.saveToDisk:
                output(Frac,
                       Fr,
                       Simulation_Parameters,
                       Material_properties,
                       Injection_Parameters,
                       Fluid_properties)

            return status, Fr
        else:
            print(errorMessages[status])

        print("Time step failed...")
        print("Reattempting with time step of " + repr(
            TimeStep * Simulation_Parameters.reAttemptFactor ** (i + 1)) + " sec")
    Frac.plot_fracture("complete", "footPrint")
    raise SystemExit("Propagation not successful. Exiting...")


def FractureFrontLoop(Frac, C, Material_properties, Fluid_properties, Simulation_Parameters, Injection_Parameters,
                      TimeStep):
    """ Propagate fracture one time step. The function injects fluid into the fracture, first by keeping the same
    footprint. This gives the first trial value of the width. The ElastoHydronamic system is then solved iteratively
    until convergence is achieved.
    
    Arguments:
        Frac (Fracture object):                             fracture object from the last time step 
        C (ndarray-float):                                  the elasticity matrix 
        Material_properties (MaterialProperties object):    material properties
        Fluid_properties (FluidProperties object):          fluid properties 
        Simulation_Parameters (SimulationParameters object): simulation parameters
        Injection_Parameters (InjectionProperties object):  injection properties
        TimeStep (float):                                   time step 
    
    Return:
        int:   possible values:
                                    0       -- not propagated
                                    1       -- iteration successful
                                    2       -- evaluated level set is not valid
                                    3       -- front is not tracked correctly
                                    4       -- evaluated tip volume is not valid
                                    5       -- solution of elastohydrodynamic solver is not valid
                                    6       -- did not converge after max iterations
                                    7       -- tip inversion not successful
                                    8       -- Ribbon element not found in the enclosure of a tip cell
                                    9       -- Filling fraction not correct
                                    
        Fracture object:            fracture after advancing time step. 
    """

    exitstatus = 0  # exit code to be returned

    # index of current time in the time series (first row) of the injection rate array
    indxCurTime = max(np.where(Frac.time >= Injection_Parameters.injectionRate[0, :])[0])
    CurrentRate = Injection_Parameters.injectionRate[1, indxCurTime]  # current injection rate

    Qin = np.zeros((Frac.mesh.NumberOfElts), float)
    Qin[Injection_Parameters.source_location] = CurrentRate # current injection over the domain


    f = open('log', 'a')

    vel_Ribbon = calculate_Velocity(Frac, "M-K", Material_properties, Fluid_properties)

    sgndDist_k = 1e10 * np.ones((Frac.mesh.NumberOfElts,), float)  # Initializing the cells with maximum
    # float value. (algorithm requires inf)
    sgndDist_k[Frac.EltChannel] = 0  # for cells inside the fracture

    sgndDist_k[Frac.EltRibbon] = Frac.sgndDist[Frac.EltRibbon] - TimeStep * vel_Ribbon

    SolveFMM(sgndDist_k,
             Frac.EltRibbon,
             Frac.EltChannel,
             Frac.mesh)

    # if some elements remain unevaluated by fast marching method. It happens with unrealistic fracture geometry.
    # todo: not satisfied with why this happens. need re-examining
    if max(sgndDist_k) == 1e10:
        exitstatus = 2
        return exitstatus, None

    print('Calculating the filling fraction of tip elements with the new fracture front location...')

    # gets the new tip elements, along with the length and angle of the perpendiculars drawn on front (also containing
    # the elements which are fully filled after the front is moved outward)
    (EltsTipNew, l_k, alpha_k, CellStatus) = reconstruct_front(sgndDist_k,
                                                               Frac.EltChannel,
                                                               Frac.mesh)

    # If the angle and length of the perpendicular are not correct
    nan = np.logical_or(np.isnan(alpha_k), np.isnan(l_k))
    if nan.any() or (l_k < 0).any() or (alpha_k < 0).any() or (alpha_k > np.pi / 2).any():
        exitstatus = 3
        return exitstatus, None

    # check if any of the tip cells has a neighbor outside the grid, i.e. fracture has reached the end of the grid.
    tipNeighb = Frac.mesh.NeiElements[EltsTipNew, :]
    for i in range(0, len(EltsTipNew)):
        if (np.where(tipNeighb[i, :] == EltsTipNew[i])[0]).size > 0:
            Frac.plot_fracture('complete', 'footPrint')
            f.write('Reached end of the grid. exiting....\n\n')
            raise SystemExit('Reached end of the grid. exiting....')

    Vel_k = -(sgndDist_k - Frac.sgndDist) / TimeStep

    # Calculate filling fraction of the tip cells for the current fracture position
    FillFrac_k = VolumeIntegral(EltsTipNew,
                                alpha_k,
                                l_k,
                                Frac.mesh,
                                'A',
                                Material_properties,
                                Frac.muPrime,
                                Vel_k[EltsTipNew]) / Frac.mesh.EltArea

    # todo !!! Hack: This check rounds the filling fraction to 1 if it is not bigger than 1 + 1e-6 (up to 6 figures)
    FillFrac_k[np.logical_and(FillFrac_k > 1.0, FillFrac_k < 1 + 1e-6)] = 1.0

    # if filling fraction is below zero or above 1+1e-6
    if (FillFrac_k > 1.0).any() or (FillFrac_k < 0.0 - np.finfo(float).eps).any():
        exitstatus = 9
        return exitstatus, None

    # some of the lists are redundant to calculate on each iteration
    # Evaluate the element lists for the trial fracture front
    (EltChannel_k,
     EltTip_k,
     EltCrack_k,
     EltRibbon_k,
     zrVertx_k,
     CellStatus_k) = UpdateLists(Frac.EltChannel,
                                 EltsTipNew,
                                 FillFrac_k,
                                 sgndDist_k,
                                 Frac.mesh)

    partlyFilledTip = np.arange(EltsTipNew.shape[0])[np.in1d(EltsTipNew, EltTip_k)]

    wTip = VolumeIntegral(EltTip_k,
                          alpha_k[partlyFilledTip],
                          l_k[partlyFilledTip],
                          Frac.mesh,
                          Simulation_Parameters.tipAsymptote,
                          Material_properties,
                          Frac.muPrime,
                          Vel_k[EltTip_k]) / Frac.mesh.EltArea

    if (wTip < 0).any():
        exitstatus = 4
        return exitstatus, None

    guess = np.zeros((EltChannel_k.size + EltTip_k.size,), float)
    # pguess = Fr_lstTmStp.p[EltsTipNew]

    guess[np.arange(EltChannel_k.size)] = TimeStep * sum(Qin) / EltChannel_k.size \
                                                    * np.ones((EltChannel_k.size,), float)
    DLkOff = np.zeros((Frac.mesh.NumberOfElts,), float)  # leak off set to zero

    # width of guess. Evaluated to calculate the current velocity at the cell edges
    wguess = np.copy(Frac.w)
    wguess[EltChannel_k] = wguess[EltChannel_k] + guess[np.arange(EltChannel_k.size)]
    wguess[EltTip_k] = wTip

    InCrack_k = np.zeros((Frac.mesh.NumberOfElts,), dtype=np.int8)
    InCrack_k[EltChannel_k] = 1
    InCrack_k[EltTip_k] = 1

    # velocity at the cell edges evaluated with the guess width. Used as guess values for the implicit velocity solver.
    vk = velocity(wguess,
                  EltCrack_k,
                  Frac.mesh,
                  InCrack_k,
                  Frac.muPrime,
                  C,
                  Material_properties.SigmaO)

    # typical value for pressure
    typValue = np.copy(guess)
    typValue[EltChannel_k.size + np.arange(EltTip_k.size)] = 1e5
    # todo too many arguments; properties class needs to be utilized
    arg = (
        EltChannel_k,
        EltTip_k,
        Frac.w,
        wTip,
        EltCrack_k,
        Frac.mesh,
        TimeStep,
        Qin,
        C,
        Frac.muPrime,
        Fluid_properties.density,
        InCrack_k,
        DLkOff,
        Frac.SigmaO,
        Fluid_properties.turbulence)

    # sloving the system of equations for [\delta] w in the channel elements and pressure in the tip elements
    (sol, vel) = Picard_Newton(Elastohydrodynamic_ResidualFun_ExtendedFP,
                               MakeEquationSystemExtendedFP,
                               guess,
                               typValue,
                               vk,
                               Simulation_Parameters.toleranceEHL,
                               Simulation_Parameters.maxSolverItr,
                               *arg)

    Fr_kplus1 = copy.deepcopy(Frac)

    Fr_kplus1.time += TimeStep

    Fr_kplus1.w[EltChannel_k] += sol[np.arange(EltChannel_k.size)]
    Fr_kplus1.w[EltTip_k] = wTip

    # check if the new width is valid
    if np.isnan(Fr_kplus1.w).any() or (Fr_kplus1.w < 0).any():
        exitstatus = 5
        return exitstatus, None

    Fr_kplus1.FillF = FillFrac_k[partlyFilledTip]
    Fr_kplus1.EltChannel = EltChannel_k
    Fr_kplus1.EltTip = EltTip_k
    Fr_kplus1.EltCrack = EltCrack_k
    Fr_kplus1.EltRibbon = EltRibbon_k
    Fr_kplus1.ZeroVertex = zrVertx_k

    # pressure evaluated by dot product of width and elasticity matrix
    Fr_kplus1.p[Fr_kplus1.EltCrack] = np.dot(C[np.ix_(Fr_kplus1.EltCrack, Fr_kplus1.EltCrack)],
                                             Fr_kplus1.w[Fr_kplus1.EltCrack])
    Fr_kplus1.sgndDist = sgndDist_k

    Fr_kplus1.alpha = alpha_k[partlyFilledTip]
    Fr_kplus1.l = l_k[partlyFilledTip]
    Fr_kplus1.v = vel_Ribbon

    Fr_kplus1.InCrack = InCrack_k

    # # check if the tip has laminar flow, to be consistent with tip asymptote.
    # ReNumb, check = turbulence_check_tip(vel, Fr_kplus1, Fluid_properties, return_ReyNumb=True)
    # # plot Reynold's number
    # plot_Reynolds_number(Fr_kplus1, ReNumb, 1)

    exitstatus = 1
    return exitstatus, Fr_kplus1

# ----------------------------------------------------------------------------------------------------------------------

def output(Fr_lstTmStp, Fr_advanced, simulation_parameters, material_properties, injection_parameters, fluid_properties):
    """
    This function plot the fracture footprint and/or save file to disk according to the given time period.
    
    Arguments:
        Fr_lstTmStp (Fracture object):                      fracture from last time step
        Fr_advanced (Fracture object):                      fracture after time step advancing
        simulation_parameters (SimulationParameters object): simulation parameters 
        material_properties (MaterialProperties object):    Material properties
         
    Returns: 
    """
    if (Fr_lstTmStp.time // simulation_parameters.outputTimePeriod !=
                Fr_advanced.time // simulation_parameters.outputTimePeriod):
        # plot fracture footprint
        if simulation_parameters.plotFigure:
            # if ploting analytical solution enabled
            if simulation_parameters.plotAnalytical:
                Q0 = injection_parameters.injectionRate[1, 0] # injection rate at the time of injection
                if simulation_parameters.analyticalSol == "M":
                    (R, p, w, v) = M_vertex_solution_t_given(material_properties.Eprime,
                                                             Q0,
                                                             fluid_properties.muPrime,
                                                             Fr_lstTmStp.mesh,
                                                             Fr_advanced.time)

                elif simulation_parameters.analyticalSol == "K":
                    (R, p, w, v) = K_vertex_solution_t_given(material_properties.Kprime[
                                                             injection_parameters.source_location],
                                                             material_properties.Eprime,
                                                             Q0,
                                                             Fr_lstTmStp.mesh,
                                                             Fr_advanced.time)

                fig = Fr_advanced.plot_fracture('complete',
                                                'footPrint',
                                                analytical=R,
                                                mat_Properties=material_properties)
                # fig = Fr_advanced.plot_fracture('complete',
                #                                 'width')
            else:
                fig = Fr_advanced.plot_fracture('complete',
                                                'footPrint',
                                                mat_Properties = material_properties)
            plt.show()

        # save fracture to disk
        if simulation_parameters.saveToDisk:
            simulation_parameters.lastSavedFile += 1
            Fr_advanced.SaveFracture(simulation_parameters.outFileAddress + "file_"
                                     + repr(simulation_parameters.lastSavedFile))


def turbulence_check_tip(vel, Fr, fluid, return_ReyNumb=False):
    """
    This function calculate the Reynolds number at the cell edges and check if any to the edge between the ribbon cells
    and the tip cells are turbulent (i.e. the Reynolds number is greater than 2100).
    
    Arguments:
        vel (ndarray-float):                    the array giving velocity of each edge of the cells in domain 
        Fr (Fracture object):                   the fracture object to be checked
        fluid (FluidProperties object):         fluid properties object 
        return_ReyNumb (boolean, default False): if true, Reynolds number at all cell edges will be returned 
    
    Returns:
        ndarray-float:      Reynolds number of all the cells in the domain; row-wise in the following order : 0--left,
                            1--right, 2--bottom, 3--top
        boolean             true if any of the edge between the ribbon and tip cells is turbulent (i.e. Reynolds number
                            is more than 2100)
    """
    # width at the adges by averaging
    wLftEdge = (Fr.w[Fr.EltRibbon] + Fr.w[Fr.mesh.NeiElements[Fr.EltRibbon, 0]]) / 2
    wRgtEdge = (Fr.w[Fr.EltRibbon] + Fr.w[Fr.mesh.NeiElements[Fr.EltRibbon, 1]]) / 2
    wBtmEdge = (Fr.w[Fr.EltRibbon] + Fr.w[Fr.mesh.NeiElements[Fr.EltRibbon, 2]]) / 2
    wTopEdge = (Fr.w[Fr.EltRibbon] + Fr.w[Fr.mesh.NeiElements[Fr.EltRibbon, 3]]) / 2

    Re = np.zeros((4, Fr.EltRibbon.size, ), dtype=np.float64)
    Re[0, :] = 4 / 3 * fluid.density * wLftEdge * vel[0, Fr.EltRibbon] / fluid.viscosity
    Re[1, :] = 4 / 3 * fluid.density * wRgtEdge * vel[1, Fr.EltRibbon] / fluid.viscosity
    Re[2, :] = 4 / 3 * fluid.density * wBtmEdge * vel[2, Fr.EltRibbon] / fluid.viscosity
    Re[3, :] = 4 / 3 * fluid.density * wTopEdge * vel[3, Fr.EltRibbon] / fluid.viscosity

    ReNum_Ribbon = []
    # adding Reynolds number of the edges between the ribbon and tip cells to a list
    for i in range(0,Fr.EltRibbon.size):
        for j in range(0,4):
            # if the current neighbor (j) of the ribbon cells is in the tip elements list
            if np.where(Fr.mesh.NeiElements[Fr.EltRibbon[i], j] == Fr.EltTip)[0].size>0:
                ReNum_Ribbon = np.append(ReNum_Ribbon, Re[j, i])

    if return_ReyNumb:
        wLftEdge = (Fr.w[Fr.EltCrack] + Fr.w[Fr.mesh.NeiElements[Fr.EltCrack, 0]]) / 2
        wRgtEdge = (Fr.w[Fr.EltCrack] + Fr.w[Fr.mesh.NeiElements[Fr.EltCrack, 1]]) / 2
        wBtmEdge = (Fr.w[Fr.EltCrack] + Fr.w[Fr.mesh.NeiElements[Fr.EltCrack, 2]]) / 2
        wTopEdge = (Fr.w[Fr.EltCrack] + Fr.w[Fr.mesh.NeiElements[Fr.EltCrack, 3]]) / 2

        Re = np.zeros((4, Fr.mesh.NumberOfElts,), dtype=np.float64)
        Re[0, Fr.EltCrack] = 4 / 3 * fluid.density * wLftEdge * vel[0, Fr.EltCrack] / fluid.viscosity
        Re[1, Fr.EltCrack] = 4 / 3 * fluid.density * wRgtEdge * vel[1, Fr.EltCrack] / fluid.viscosity
        Re[2, Fr.EltCrack] = 4 / 3 * fluid.density * wBtmEdge * vel[2, Fr.EltCrack] / fluid.viscosity
        Re[3, Fr.EltCrack] = 4 / 3 * fluid.density * wTopEdge * vel[3, Fr.EltCrack] / fluid.viscosity

        return Re, (ReNum_Ribbon > 2100.).any()
    else:
        return (ReNum_Ribbon > 2100.).any()