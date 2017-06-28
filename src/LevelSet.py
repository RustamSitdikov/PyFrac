# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on Tue Dec 27 19:01:22 2016.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2017. All rights reserved.
See the LICENSE.TXT file for more details.
"""

# imports
import numpy as np


from src.Utility import *


def SolveFMM(InitlevelSet, EltRibbon, EltChannel, mesh):
    """
    solve Eikonal equation to get level set.
    
    Arguments:
        InitlevelSet (ndarray-float):       level set with initial values in ribbon cells
        EltRibbon (ndarray-int):            ribbon elements
        EltChannel (ndarray-int):           channel elements
        mesh (CartesianMesh object):        mesh object
    
    Returns:
        Does not return anything. The InitlevelSet is updated in place.
    """

    # for Elements radialy outward from ribbon cells
    Alive = np.copy(EltRibbon)
    NarrowBand = np.copy(EltRibbon)
    FarAway = np.delete(range(mesh.NumberOfElts), np.intersect1d(range(mesh.NumberOfElts), EltRibbon, None))
    # the maximum distance any point can have from another in the current mesh. This distance is used to detect the
    # cells that are not yet traversed, i.e. having infinity distance
    maxdist = 4 * (mesh.Lx ** 2 + mesh.Ly ** 2) ** 0.5

    while NarrowBand.size > 0:

        Smallest = int(NarrowBand[InitlevelSet[NarrowBand.astype(int)].argmin()])
        neighbors = mesh.NeiElements[Smallest]

        for neighbor in neighbors:
            if not neighbor in Alive:

                if neighbor in FarAway:
                    NarrowBand = np.append(NarrowBand, neighbor)
                    FarAway = np.delete(FarAway, np.where(FarAway == neighbor))

                NeigxMin = min(InitlevelSet[mesh.NeiElements[neighbor, 0]], InitlevelSet[mesh.NeiElements[neighbor, 1]])
                NeigyMin = min(InitlevelSet[mesh.NeiElements[neighbor, 2]], InitlevelSet[mesh.NeiElements[neighbor, 3]])
                beta = mesh.hx / mesh.hy
                delT = NeigyMin - NeigxMin

                theta = (mesh.hx ** 2 * (1 + beta ** 2) - beta ** 2 * delT ** 2) ** 0.5  # it goes to nan for fully
                # horizontal or fully vertical perpendiculars on the front

                if not np.isnan((NeigxMin + beta * NeigyMin + theta) / (1 + beta ** 2)):
                    InitlevelSet[neighbor] = (NeigxMin + beta ** 2 * NeigyMin + theta) / (1 + beta ** 2)
                else:  # the angle is either 0 or 90 degrees
                    # vertical propagation direction.
                    if NeigxMin > maxdist:  # used to check if very large value (level set value for unevaluated elements)
                        InitlevelSet[neighbor] = NeigyMin + mesh.hy
                    # horizontal propagation direction.
                    if NeigyMin > maxdist:
                        InitlevelSet[neighbor] = NeigxMin + mesh.hx
                # if abs(InitlevelSet[mesh.NeiElements[neighbor, 0]]/InitlevelSet[mesh.NeiElements[neighbor, 1]]-1)<1e-8:
                #     InitlevelSet[neighbor] = NeigyMin + mesh.hy
                # elif abs(InitlevelSet[mesh.NeiElements[neighbor, 2]]/InitlevelSet[mesh.NeiElements[neighbor, 3]]-1)<1e-8:
                #     InitlevelSet[neighbor] = NeigxMin + mesh.hx
        Alive = np.append(Alive, Smallest)
        NarrowBand = np.delete(NarrowBand, np.where(NarrowBand == Smallest))


    # for elements radialy inward from ribbon cells. The sign of the level set values(tip asymptote) in the ribbon cells
    # is inverted to run the fast marching algorithm. The sign is finally inverted back to assign the value in the level
    # set to be returned.

    RibbonInwardElts = np.copy(EltChannel)
    for i in range(len(EltRibbon)):
        RibbonInwardElts = np.delete(RibbonInwardElts, np.where(RibbonInwardElts==EltRibbon[i])[0])

    positive_levelSet = 1e10 * np.ones((mesh.NumberOfElts,), np.float64)
    positive_levelSet[EltRibbon] = -InitlevelSet[EltRibbon]
    Alive = np.copy(EltRibbon)
    NarrowBand = np.copy(EltRibbon)
    FarAway = np.copy(RibbonInwardElts)

    while NarrowBand.size > 0:

        Smallest = int(NarrowBand[positive_levelSet[NarrowBand.astype(int)].argmin()])
        neighbors = mesh.NeiElements[Smallest]

        for neighbor in neighbors:
            if not neighbor in Alive:

                if neighbor in FarAway:
                    NarrowBand = np.append(NarrowBand, neighbor)
                    FarAway = np.delete(FarAway, np.where(FarAway == neighbor))

                NeigxMin = min(positive_levelSet[mesh.NeiElements[neighbor, 0]],
                               positive_levelSet[mesh.NeiElements[neighbor, 1]])
                NeigyMin = min(positive_levelSet[mesh.NeiElements[neighbor, 2]],
                               positive_levelSet[mesh.NeiElements[neighbor, 3]])
                beta = mesh.hx / mesh.hy
                delT = NeigyMin - NeigxMin

                theta = (mesh.hx ** 2 * (1 + beta ** 2) - beta ** 2 * delT ** 2) ** 0.5

                if not np.isnan((NeigxMin + beta * NeigyMin + theta) / (1 + beta ** 2)):
                    positive_levelSet[neighbor] = (NeigxMin + beta ** 2 * NeigyMin + theta) / (1 + beta ** 2)
                else:  # the angle is either 0 or 90 degrees
                    # vertical propagation direction.
                    if NeigxMin > maxdist:  # used to check if very large value (level set value for unevaluated elements)
                        positive_levelSet[neighbor] = NeigyMin + mesh.hy
                    # horizontal propagation direction.
                    if NeigyMin > maxdist:
                        positive_levelSet[neighbor] = NeigxMin + mesh.hx

                # if abs(positive_levelSet[mesh.NeiElements[neighbor, 0]]/positive_levelSet[mesh.NeiElements[neighbor, 1]]-1)<1e-8:
                #     positive_levelSet[neighbor] = NeigyMin + mesh.hy
                # elif abs(positive_levelSet[mesh.NeiElements[neighbor, 2]]/positive_levelSet[mesh.NeiElements[neighbor, 3]]-1)<1e-8:
                #     positive_levelSet[neighbor] = NeigxMin + mesh.hx
        Alive = np.append(Alive, Smallest)
        NarrowBand = np.delete(NarrowBand, np.where(NarrowBand == Smallest))

    # assigning adjusted value to the level set to be returned
    InitlevelSet[RibbonInwardElts] = -positive_levelSet[RibbonInwardElts]
# -----------------------------------------------------------------------------------------------------------------------


def reconstruct_front(dist, EltChannel, EltRibbon, mesh):
    """
    Track the fracture front, the length of the perpendicular drawn on the fracture and the angle inscribed by the
    perpendicular.
    
    Arguments:
        dist (ndarray-float): the signed distance of the cells from the fracture front
        EltChannel (ndarray-int): list of Channel elements
        mesh (CartesianMesh object): the mesh of the fracture
        
    Returns:
        ndarray-int:            new tip elements
        ndarray-float:          the length of the perpendicular from the fracture front to the zero vertex
        ndarray-float:          the angle inscribed by the perpendicular from the fracture front to the zero vertex
        ndarray-int:            the type of the elements (1 for channel elements, 2 for tip elements, 0 for rest)
        ndarray-int:            Vertex from which the perpendicular is drawn (can have value from 0 to 3, where
                                0 signify bottom left, 1 signifying bottom right, 2 signifying top right and 3
                                signifying top left vertex)
    """

    # Elements that are not in channel
    EltRest = np.delete(range(mesh.NumberOfElts), EltChannel, None)
    ElmntTip = np.asarray([], int)
    l = np.zeros((mesh.NumberOfElts,), dtype=np.float64)
    alpha = np.zeros((mesh.NumberOfElts,), dtype=np.float64)
    # finding distance from front for ribbon elements also. It will be used to check the propagation condition.
    EltRest = np.append(EltRest, EltRibbon)
    zeroVrtx = 255 * np.ones((mesh.NumberOfElts,), int)     # Vertex from where the perpendicular is drawn (255 for
                                                            # unitialized)

    for i in range(0, len(EltRest)):
        neighbors = np.asarray(Neighbors(EltRest[i], mesh.nx, mesh.ny))


        # minx = min(dist[neighbors[0]], dist[neighbors[1]])
        # miny = min(dist[neighbors[2]], dist[neighbors[3]])
        if dist[neighbors[0]] < dist[neighbors[1]]:
            minx = dist[neighbors[0]]
            drctx = -1
        else:
            minx = dist[neighbors[1]]
            drctx = 1

        if dist[neighbors[2]] < dist[neighbors[3]]:
            miny = dist[neighbors[2]]
            drcty = -1
        else:
            miny = dist[neighbors[3]]
            drcty = 1
        # distance of the vertex (zero vertex, i.e. rotated distance) of the current cell from the front
        Pdis = -(minx + miny) / 2

        # if the vertex distance is positive, meaning the fracture has passed the vertex
        if Pdis >= 0:
            if not EltRest[i] in EltRibbon:
                ElmntTip = np.append(ElmntTip, EltRest[i])
            l[EltRest[i]] = Pdis

            # calculate angle imposed by the perpendicular on front (see Peirce & Detournay 2008)
            delDist = miny - minx
            beta = mesh.hx / mesh.hy


            theta = (mesh.hx ** 2 * (1 + beta ** 2) - beta ** 2 * delDist ** 2) ** 0.5

            if np.isnan((mesh.hx ** 2 * (1 + beta ** 2) - beta ** 2 * delDist ** 2) ** 0.5):
                theta = (abs(mesh.hx ** 2 * (1 + beta ** 2) - beta ** 2 * delDist ** 2)) ** 0.5

            # angle calculate with inverse of cosine trigonometric function
            a1 = np.arccos((theta + beta ** 2 * delDist) / (mesh.hx * (1 + beta ** 2)))
            # angle calculate with inverse of sine trigonometric function
            sinalpha = beta * (theta - delDist) / (mesh.hx * (1 + beta ** 2))
            a2 = np.arcsin(sinalpha)
            # angle calculated with tan
            # if minx<0 and miny<0:
            if (mesh.hx ** 2 * (1 + beta ** 2) - beta ** 2 * delDist ** 2)<0.0:
                a1 = np.pi/4
                # a3 = np.arccos(abs(minx-miny)/(mesh.hx**2+mesh.hy**2)**0.5)-np.arctan(mesh.hy/mesh.hx)
                # print(repr(a3))

            # !!!Hack. this check of zero or 90 degree angle works better
            if abs(1 - dist[neighbors[0]] / dist[neighbors[1]]) < 1e-8:
                a2 = np.pi / 2
            elif abs(1 - dist[neighbors[2]] / dist[neighbors[3]]) < 1e-8:
                a2 = 0

            # checks to remove numerical noise in angle calculation
            if a2 >= 0 and a2 <= np.pi / 2:
                alpha[EltRest[i]] = a2
            elif a1 >= 0 and a1 <= np.pi / 2:
                alpha[EltRest[i]] = a1
            elif a2 < 0 and a2 > -1e-6:
                alpha[EltRest[i]] = 0
            elif a2 > np.pi / 2 and a2 < np.pi / 2 + 1e-6:
                alpha[EltRest[i]] = np.pi / 2
            elif a1 < 0 and a1 > -1e-6:
                alpha[EltRest[i]] = 0
            elif a1 > np.pi / 2 and a1 < np.pi / 2 + 1e-6:
                alpha[EltRest[i]] = np.pi / 2
            else:
                alpha[EltRest[i]] = np.nan

            # assigning zerot vertex
            if drctx < 0 and drcty < 0:
                zeroVrtx[EltRest[i]] = 0
            if drctx > 0 and drcty < 0:
                zeroVrtx[EltRest[i]] = 1
            if drctx < 0 and drcty > 0:
                zeroVrtx[EltRest[i]] = 3
            if drctx > 0 and drcty > 0:
                zeroVrtx[EltRest[i]] = 2

    CellStatusNew = np.zeros((mesh.NumberOfElts), int)
    CellStatusNew[EltChannel] = 1
    CellStatusNew[ElmntTip] = 2

    return ElmntTip, l, alpha, CellStatusNew, zeroVrtx


# -----------------------------------------------------------------------------------------------------------------------

def UpdateLists(EltsChannel, EltsTipNew, FillFrac, levelSet, mesh):
    """
    This function update the Element lists, given the element lists from the last time step. EltsTipNew list can have 
    partially filled and fully filled elements. The function update lists accordingly.
    
    Arguments:
        EltsChannel (ndarray-int):      channel elements list
        EltsTipNew (ndarray-int):       list of the new tip elements, including fully filled cells that were tip cells
                                        in the last time step
        FillFrac (ndarray-float):       filling fraction of the new tip cells
        levelSet (ndarray-float):       current level set
        mesh (CartesianMesh object):    the mesh of the fracture
        
    Returns:
        (ndarray-int):                  new channel elements list
        (ndarray-int):                  new tip elements list
        (ndarray-int):                  new crack elements list
        (ndarray-int):                  new ribbon elements list
        (ndarray-int):                  specifies which region each element currently belongs to
    """

    # new tip elements contain only the partially filled elements
    eltsTip = EltsTipNew[np.where(FillFrac <= 0.999999)]

    # Tip elements flag to avoid search on each iteration
    inTip = np.zeros((mesh.NumberOfElts,), bool)
    inTip[eltsTip] = True
    i = 0
    while i < len(eltsTip):  # to remove a special case encountered in sharp edges and rectangular cells
        neighbors = np.asarray(Neighbors(eltsTip[i], mesh.nx, mesh.ny))
        if inTip[neighbors[0]] and inTip[neighbors[3]] and inTip[neighbors[3] - 1]:
            conjoined = np.asarray([neighbors[0], neighbors[3], neighbors[3] - 1, eltsTip[i]])
            mindist = np.argmin(mesh.distCenter[conjoined])
            inTip[conjoined[mindist]] = False
            eltsTip = np.delete(eltsTip, np.where(eltsTip == conjoined[mindist]))
            i -= 1
        i += 1

    newEltChannel = np.copy(EltsTipNew)  # new channel elements
    for i in range(0, len(eltsTip)):
        newEltChannel = np.delete(newEltChannel, np.where(newEltChannel == eltsTip[i]))

    eltsChannel = np.append(EltsChannel, newEltChannel)
    eltsCrack = np.append(eltsChannel, eltsTip)
    eltsRibbon = np.array([], int)

    # All the inner cells neighboring tip cells are added to ribbon cells
    for i in range(0, len(eltsTip)):
        neighbors = mesh.NeiElements[eltsTip[i]] #np.asarray(Neighbors(eltsTip[i], mesh.nx, mesh.ny))

        if levelSet[neighbors[0]] < levelSet[neighbors[1]]:# -1 < 0:
            eltsRibbon = np.append(eltsRibbon, neighbors[0])
        else:
            eltsRibbon = np.append(eltsRibbon, neighbors[1])

        if levelSet[neighbors[2]] < levelSet[neighbors[3]]:# -1 < 0:
            eltsRibbon = np.append(eltsRibbon, neighbors[2])
        else:
            eltsRibbon = np.append(eltsRibbon, neighbors[3])


    # Remove repetitions in the ribbon cells
    eltsRibbon = np.unique(eltsRibbon)
    for i in range(0, len(eltsTip)):
        eltsRibbon = np.delete(eltsRibbon, np.where(eltsRibbon == eltsTip[i]))

    to_delete = np.asarray([])
    for i in range(0, len(eltsRibbon)):
        neighbors = mesh.NeiElements[eltsRibbon[i]]
        enclosing = np.append(neighbors, np.asarray(
            [neighbors[2] - 1, neighbors[2] + 1, neighbors[3] - 1, neighbors[3] + 1]))
        if sum(np.in1d(enclosing,eltsRibbon))< 2:
            to_delete = np.append(to_delete, np.where(eltsRibbon == eltsRibbon[i])[0])
    eltsRibbon = np.delete(eltsRibbon, to_delete)

    for i in range(0,len(eltsRibbon)):
        if not eltsRibbon[i] in eltsChannel:
            print("not in ribbon")
    # Cells status list store the status of all the cells in the domain
    CellStatusNew = np.zeros((mesh.NumberOfElts), int)
    CellStatusNew[eltsChannel] = 1
    CellStatusNew[eltsTip] = 2
    CellStatusNew[eltsRibbon] = 3

    return eltsChannel, eltsTip, eltsCrack, eltsRibbon, CellStatusNew

    # -----------------------------------------------------------------------------------------------------------------------

def Eikonal_Res(Tij, *args):
    """quadratic Eikonal equation residual to be used by numerical root finder"""

    (Tleft, Tright, Tbottom, Ttop, Fij, dx, dy) = args
    return np.nanmax([(Tij - Tleft) / dx, 0]) ** 2 + np.nanmin([(Tright - Tij) / dx, 0]) ** 2 + np.nanmax(
        [(Tij - Tbottom) / dy, 0]) ** 2 + \
           np.nanmin([(Ttop - Tij) / dy, 0]) ** 2 - Fij ** 2


# -----------------------------------------------------------------------------------------------------------------------