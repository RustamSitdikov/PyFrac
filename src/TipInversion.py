# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on Tue Nov  1 15:22:00 2016.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2017. All rights reserved.
See the LICENSE.TXT file for more details.

Tip inversion for different flow regimes. The functions take width opening and gives distance from tip calculated using the given propagation regime
"""

# imports
import math
import numpy as np
from scipy.optimize import brentq
from src.Utility import *
import matplotlib.pyplot as plt


def TipAsym_viscStor_Res(dist, *args):
    """Residual function for viscocity dominate regime, without leak off"""

    (wEltRibbon, Kprime, Eprime, muPrime, Cbar, DistLstTSEltRibbon, dt) = args
    return wEltRibbon - (18 * 3 ** 0.5 * (dist - DistLstTSEltRibbon) / dt * muPrime / Eprime) ** (1 / 3) * dist ** (
        2 / 3)


# ----------------------------------------------------------------------------------------------------------------------

def TipAsym_viscLeakOff_Res(dist, *args):
    """Residual function for viscosity dominated regime, with leak off"""
    (wEltRibbon, Kprime, Eprime, muPrime, Cbar, DistLstTSEltRibbon, dt) = args
    return wEltRibbon - 4 / (15 * np.tan(np.pi / 8)) ** 0.25 * (Cbar * muPrime / Eprime) ** 0.25 * ((
                                                        dist - DistLstTSEltRibbon) / dt) ** 0.125 * dist ** (5 / 8)


# ----------------------------------------------------------------------------------------------------------------------

def f(K, Cb, C1):
    return 1 / (3 * C1) * (
        1 - K ** 3 - 3 * Cb * (1 - K ** 2) / 2 + 3 * Cb ** 2 * (1 - K) - 3 * Cb ** 3 * np.log((Cb + 1) / (Cb + K)))


# ----------------------------------------------------------------------------------------------------------------------

def TipAsym_Universal_delt_Res(dist, *args):
    """More precise function to be minimized to find root for universal Tip assymptote (see Donstov and Pierce)"""

    (wEltRibbon, Kprime, Eprime, muPrime, Cbar, DistLstTSEltRibbon, dt) = args

    Vel = (dist - DistLstTSEltRibbon) / dt
    Kh = Kprime * dist ** 0.5 / (Eprime * wEltRibbon)
    Ch = 2 * Cbar * dist ** 0.5 / (Vel ** 0.5 * wEltRibbon)
    sh = muPrime * Vel * dist ** 2 / (Eprime * wEltRibbon ** 3)

    g0 = f(Kh, 0.9911799823 * Ch, 10.392304845)
    delt = 10.392304845 * (1 + 0.9911799823 * Ch) * g0

    C1 = 4 * (1 - 2 * delt) / (delt * (1 - delt)) * np.tan(math.pi * delt)
    C2 = 16 * (1 - 3 * delt) / (3 * delt * (2 - 3 * delt)) * np.tan(3 * math.pi * delt / 2)
    b = C2 / C1

    return sh - f(Kh, Ch * b, C1)


# ----------------------------------------------------------------------------------------------------------------------

def TipAsym_Universal_zero_Res(dist, *args):
    """Function to be minimized to find root for universal Tip assymptote (see Donstov and Pierce 2017)"""
    (wEltRibbon, Kprime, Eprime, muPrime, Cbar, DistLstTSEltRibbon, dt) = args

    Vel = (dist - DistLstTSEltRibbon) / dt
    Kh = Kprime * dist ** 0.5 / (Eprime * wEltRibbon)
    Ch = 2 * Cbar * dist ** 0.5 / (Vel ** 0.5 * wEltRibbon)
    g0 = f(Kh, 0.9911799823 * Ch, 6 * 3 ** 0.5)
    sh = muPrime * Vel * dist ** 2 / (Eprime * wEltRibbon ** 3)

    return sh - g0


# -----------------------------------------------------------------------------------------------------------------------

def TipAsym_MKTransition_Res(dist, *args):
    """Residual function for viscocity to toughness regime with transition, without leak off"""
    (wEltRibbon, Kprime, Eprime, muPrime, Cbar, DistLstTSEltRibbon, dt) = args
    return wEltRibbon - (1 + 18 * 3 ** 0.5 * Eprime ** 2 * (
        dist - DistLstTSEltRibbon) / dt * muPrime * dist ** 0.5 / Kprime ** 3) ** (
                        1 / 3) * Kprime / Eprime * dist ** 0.5


# ----------------------------------------------------------------------------------------------------------------------

def FindBracket_dist(w, frac, mat_prop, fluid_prop, Kprime, dt, ResFunc):
    """
     Find the valid bracket for the root evaluation function. Also returns list of ribbon cells that are not
     propagating
     """

    stagnant = np.where(
        # propagation condition
        Kprime * (-frac.sgndDist[frac.EltRibbon]) ** 0.5 / (mat_prop.Eprime * w[frac.EltRibbon]) > 1)
    moving = np.arange(frac.EltRibbon.shape[0])[~np.in1d(frac.EltRibbon, frac.EltRibbon[stagnant])]

    a = -frac.sgndDist[frac.EltRibbon[moving]] * (1 + 1e5 * np.finfo(float).eps)
    b = 10 * (w[frac.EltRibbon[moving]] / (Kprime[moving] / mat_prop.Eprime)) ** 2

    for i in range(0, len(moving)):

        TipAsmptargs = (w[frac.EltRibbon[moving[i]]],
                        Kprime[moving[i]],
                        mat_prop.Eprime,
                        fluid_prop.muPrime,
                        mat_prop.Cprime[frac.EltRibbon[moving[i]]],
                        -frac.sgndDist[frac.EltRibbon[moving[i]]],
                        dt)

        Res_a = ResFunc(a[i], *TipAsmptargs)
        Res_b = ResFunc(b[i], *TipAsmptargs)

        cnt = 0
        mid = b[i]
        while Res_a * Res_b > 0:
            mid = (a[i] + 2 * mid) / 3  # weighted
            Res_a = ResFunc(mid, *TipAsmptargs)
            a[i] = mid
            cnt += 1
            if cnt >= 30:  # Should assume not propagating. not set to check how frequently it happens.
                raise SystemExit('Tip Inversion: front distance bracket cannot be found')

    return moving, a, b


# ----------------------------------------------------------------------------------------------------------------------

def TipAsymInversion(w, frac, matProp, fluidProp, simParmtrs, dt=None):
    """ 
    Evaluate distance from the front using tip assymptotics of the given regime, given the fracture width in the ribbon
    cells.
    Arguments:
        w (ndarray-float):                      fracture width
        frac (Fracture object):                 current fracture object
        matProp (MaterialProperties object):    Material properties
        simParmtrs (SimulationParameters object): Simulation parameters
        dt (float):                             time step
        
    Returns:
        ndarray-float:                          distance (unsigned) from the front for the ribbon cells.
    """

    if simParmtrs.tipAsymptote == 'U':
        ResFunc = TipAsym_Universal_zero_Res
    # ResFunc = TipAsym_Universal_delt_Res
    elif simParmtrs.tipAsymptote == 'Kt':
        return 0  # todo: to be implementd
    elif simParmtrs.tipAsymptote == 'M':
        ResFunc = TipAsym_viscStor_Res
    elif simParmtrs.tipAsymptote == 'Mt':
        ResFunc = TipAsym_viscLeakOff_Res
    elif simParmtrs.tipAsymptote == 'MK':
        ResFunc = TipAsym_MKTransition_Res
    elif simParmtrs.tipAsymptote == 'K':
        return w[frac.EltRibbon] ** 2 * (matProp.Eprime / matProp.Kprime[[frac.EltRibbon]]) ** 2

    Kprime = toughness(frac.EltRibbon, frac.l, frac.alpha, frac.ZeroVertex, matProp.KprimeFunc, frac.mesh)

    (moving, a, b) = FindBracket_dist(w, frac, matProp, fluidProp, Kprime, dt, ResFunc)
    dist = -frac.sgndDist[frac.EltRibbon]
    for i in range(0, len(moving)):
        TipAsmptargs = (w[frac.EltRibbon[moving[i]]], Kprime[moving[i]], matProp.Eprime,
                        frac.muPrime[frac.EltRibbon[moving[i]]], matProp.Cprime[frac.EltRibbon[moving[i]]],
                        -frac.sgndDist[frac.EltRibbon[moving[i]]], dt)
        try:
            dist[moving[i]] = brentq(ResFunc, a[i], b[i], TipAsmptargs)
        except RuntimeError:
            dist[moving[i]] = np.nan

    return dist


# -----------------------------------------------------------------------------------------------------------------------

def StressIntensityFactor(w, lvlSetData, EltTip, EltRibbon, stagnant, mesh, Eprime):
    """ See Donstov & Pierce Comput. Methods Appl. Mech. Engrn. 2017"""
    KIPrime = np.zeros((EltTip.size,), float)
    for i in range(0, len(EltTip)):
        if stagnant[i]:
            neighbors = mesh.NeiElements[EltTip[i]]
            enclosing = np.append(neighbors, np.asarray(
                [neighbors[2] - 1, neighbors[2] + 1, neighbors[3] - 1, neighbors[3] + 1]))  # eight enclosing cells

            InRibbon = np.asarray([], int)  # find neighbors in Ribbon cells
            for e in range(8):
                found = np.where(EltRibbon == enclosing[e])[0]
                if found.size > 0:
                    InRibbon = np.append(InRibbon, EltRibbon[found[0]])

            if InRibbon.size == 1:
                KIPrime[i] = w[InRibbon[0]] * Eprime / (-lvlSetData[InRibbon[0]]) ** 0.5
            elif InRibbon.size > 1:  # evaluate using least squares method
                KIPrime[i] = Eprime * (w[InRibbon[0]] * (-lvlSetData[InRibbon[0]]) ** 0.5 + w[InRibbon[1]] * (
                    -lvlSetData[InRibbon[1]]) ** 0.5) / (-lvlSetData[InRibbon[0]] - lvlSetData[InRibbon[1]])
            else:  # ribbon cells not found in enclosure, evaluating with the closest ribbon cell
                RibbonCellsDist = ((mesh.CenterCoor[EltRibbon, 0] - mesh.CenterCoor[EltTip[i], 0]) ** 2 + (
                    mesh.CenterCoor[EltRibbon, 1] - mesh.CenterCoor[EltTip[i], 1]) ** 2) ** 0.5
                closest = EltRibbon[np.argmin(RibbonCellsDist)]
                KIPrime[i] = w[closest] * Eprime / (-lvlSetData[closest]) ** 0.5

    return KIPrime

def toughness(Elts, l, alpha, zero_vrtx, Kprime, mesh):
    """
    This function gives the toughness at the closest tip for the given cells.
    
    Arguments:
        Elts (ndarray-int):             the elements for which the toughness is to be evaluated
        l (ndarray-float):              the length of the perpendicular drawn from the zero vertex on the fracture front 
        alpha (ndarray-float):          the angle inscribed by the perpendicular 
        zero_vrtx (ndarray-int):        the vertex from which the perpendicular is drawn
        Kprime (callable function):     the callable function returning the toughness given the coordinates
        mesh (CartesianMesh object):    the mesh object
    
    Return:
        ndarray-float:                  toughness at the front for the given elements
    """

    coor = np.zeros((len(Elts),2),)

    # obtaining the coordinates of front from the perpendicular drawn on the fracture front from the vertex of the given
    # elements
    for i in range(0,len(Elts)):
        if zero_vrtx[Elts[i]] == 0:
            coor[i, 0] = mesh.VertexCoor[mesh.Connectivity[Elts[i], 0], 0] + l[Elts[i]] * np.cos(alpha[Elts[i]])
            coor[i, 1] = mesh.VertexCoor[mesh.Connectivity[Elts[i], 0], 1] + l[Elts[i]] * np.sin(alpha[Elts[i]])
        elif zero_vrtx[Elts[i]] == 1:
            coor[i, 0] = mesh.VertexCoor[mesh.Connectivity[Elts[i], 1], 0] - l[Elts[i]] * np.cos(alpha[Elts[i]])
            coor[i, 1] = mesh.VertexCoor[mesh.Connectivity[Elts[i], 1], 1] + l[Elts[i]] * np.sin(alpha[Elts[i]])
        elif zero_vrtx[Elts[i]] == 2:
            coor[i, 0] = mesh.VertexCoor[mesh.Connectivity[Elts[i], 2], 0] - l[Elts[i]] * np.cos(alpha[Elts[i]])
            coor[i, 1] = mesh.VertexCoor[mesh.Connectivity[Elts[i], 2], 1] - l[Elts[i]] * np.sin(alpha[Elts[i]])
        elif zero_vrtx[Elts[i]] == 3:
            coor[i, 0] = mesh.VertexCoor[mesh.Connectivity[Elts[i], 3], 0] + l[Elts[i]] * np.cos(alpha[Elts[i]])
            coor[i, 1] = mesh.VertexCoor[mesh.Connectivity[Elts[i], 3], 1] - l[Elts[i]] * np.sin(alpha[Elts[i]])

    return Kprime.ev(coor[:,0],coor[:,1])