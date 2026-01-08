import logging
import os
from typing import Optional
from typing import Annotated
import vtk
import sitkUtils
from qt import QFileDialog
from slicer import vtkMRMLScalarVolumeNode
import slicer   
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)

# check if package is in slicer env, if not install it.
def _ensure_package(pkg, import_name=None, version=None):
    import_name = import_name or pkg
    try:
        __import__(import_name)
    except ImportError:
        spec = pkg if version is None else f"{pkg}=={version}"
        logging.info(f"Installing {spec} for Slicer...")
        slicer.util.pip_install(spec)
        __import__(import_name)

_ensure_package("numpy")
_ensure_package("pandas")
_ensure_package("scipy")
_ensure_package("SimpleITK", import_name="SimpleITK")

import numpy as np
import pandas as pd
import SimpleITK as sitk
from qt import QTableWidgetItem

# Slicer Bullshit
@parameterNodeWrapper
class AirwayEnergyFractionParameterNode:
    inputVolume: vtkMRMLScalarVolumeNode
    imageThreshold: Annotated[float, WithinRange(-100, 500)] = 100
    invertThreshold: bool = False
    thresholdedVolume: vtkMRMLScalarVolumeNode
    invertedVolume: vtkMRMLScalarVolumeNode
class AirwayEnergyFractionLogic(ScriptedLoadableModuleLogic):
    def __init__(self) -> None:
        ScriptedLoadableModuleLogic.__init__(self)
    def getParameterNode(self):
        return AirwayEnergyFractionParameterNode(super().getParameterNode())
    def process(self,
                inputVolume: vtkMRMLScalarVolumeNode,
                outputVolume: vtkMRMLScalarVolumeNode,
                imageThreshold: float,
                invert: bool = False,
                showResult: bool = True) -> None:
        if not inputVolume or not outputVolume:
            raise ValueError("Input or output volume is invalid")
        import time
        startTime = time.time()
        logging.info("Processing started")
        cliParams = {
            "InputVolume": inputVolume.GetID(),
            "OutputVolume": outputVolume.GetID(),
            "ThresholdValue": imageThreshold,
            "ThresholdType": "Above" if invert else "Below",
        }
        cliNode = slicer.cli.run(slicer.modules.thresholdscalarvolume, None, cliParams, wait_for_completion=True, update_display=showResult)
        slicer.mrmlScene.RemoveNode(cliNode)
        stopTime = time.time()
        logging.info(f"Processing completed in {stopTime-startTime:.2f} seconds")
class AirwayEnergyFractionTest(ScriptedLoadableModuleTest):
    def setUp(self):
        slicer.mrmlScene.Clear()
    def runTest(self):
        pass
class AirwayEnergyFraction(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("AirwayEnergyFraction")
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "APPIL Tools")]
        self.parent.dependencies = []
        self.parent.contributors = ["Jacob Kitzmann"]
        self.parent.helpText = _('')
        self.parent.acknowledgementText = _('')

class AirwayEnergyFractionWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    def __init__(self, parent=None) -> None:
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None
        self.centroidFiducialNodesByVolumeId = {}  # volumeID -> fiducial node
    # -- slicer code --
    def cleanup(self) -> None:
        self.removeObservers()
    def enter(self) -> None:
        self.initializeParameterNode()
    def exit(self) -> None:
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None
    def onSceneStartClose(self, caller, event) -> None:
        self.setParameterNode(None)
    def onSceneEndClose(self, caller, event) -> None:
        if self.parent.isEntered:
            self.initializeParameterNode()
    def initializeParameterNode(self) -> None:
        self.setParameterNode(self.logic.getParameterNode())
        if not self._parameterNode.inputVolume:
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if firstVolumeNode:
                self._parameterNode.inputVolume = firstVolumeNode
    def setParameterNode(self, inputParameterNode: Optional[AirwayEnergyFractionParameterNode]) -> None:
        self._parameterNode = inputParameterNode
        if self._parameterNode:
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            

    def setup(self) -> None:
        # -- slicer code -- 
        ScriptedLoadableModuleWidget.setup(self)
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/AirwayEnergyFraction.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)
        uiWidget.setMRMLScene(slicer.mrmlScene)
        self.logic = AirwayEnergyFractionLogic()
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)
        self.initializeParameterNode()
        
        # -- my code -- 
            
        self.setupPointsTable()
        self.ui.pointPairButton.connect('clicked(bool)', self.onPointPairButton)
        self.ui.executeButton.connect('clicked (bool)', self.onExecuteButton)

        
    # initialize table that stores all of the points the user has entered
    def setupPointsTable(self):
        t = self.ui.pointsTable
        t.setRowCount(10)
        t.setColumnCount(2)
        t.setHorizontalHeaderLabels(['Point 1', 'Point 2'])

        t.verticalHeader().setVisible(True)
        t.horizontalHeader().setStretchLastSection(True)

        # Fill with empty items so setText later never crashes
        for r in range(t.rowCount):
            for c in range(t.columnCount):
                if t.item(r, c) is None:
                    t.setItem(r, c, QTableWidgetItem(""))

    # place 2 points to caputure an airway
    def onPointPairButton(self):
        volumeNode = self._parameterNode.inputVolume
        self._pairVolumeNode = volumeNode

        if volumeNode is None:
            slicer.util.errorDisplay("Select an input volume first.")
            return

        # if we dont have a node for the user to place, create one
        if not hasattr(self, "_pairFidNode") or self._pairFidNode is None:
            self._pairFidNode = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLMarkupsFiducialNode", "PointPair"
            )
        else: # otherwise use the one we already have
            self._pairFidNode.RemoveAllControlPoints()

        self._pairFidNode.SetLocked(False)

        # dont stack observers
        if getattr(self, "_pairObserverTag", None) is not None: 
            self._pairFidNode.RemoveObserver(self._pairObserverTag)
        self._pairObserverTag = self._pairFidNode.AddObserver(
            slicer.vtkMRMLMarkupsNode.PointModifiedEvent, self._onPairPointsModified
        )

        # create points for placement
        selectionNode = slicer.mrmlScene.GetNodeByID("vtkMRMLSelectionNodeSingleton")
        selectionNode.SetReferenceActivePlaceNodeID(self._pairFidNode.GetID())

        interactionNode = slicer.mrmlScene.GetNodeByID("vtkMRMLInteractionNodeSingleton")
        interactionNode.SetPlaceModePersistence(1)     
        interactionNode.SetCurrentInteractionMode(interactionNode.Place)

        slicer.util.showStatusMessage("Click 2 fiducial points in the viewer…", 3000)

    def _onPairPointsModified(self, caller, event):
        fid = self._pairFidNode
        if fid is None:
            return

        # safety
        if fid.GetNumberOfDefinedControlPoints() < 2:
            return

        # get positions
        ras0 = [0.0, 0.0, 0.0]
        ras1 = [0.0, 0.0, 0.0]
        fid.GetNthControlPointPositionWorld(0, ras0)
        fid.GetNthControlPointPositionWorld(1, ras1)

        # convert to IJK
        ijk0 = self._rasToIJK(self._pairVolumeNode, ras0)
        ijk1 = self._rasToIJK(self._pairVolumeNode, ras1)

        # put as strings into next empty row
        row = self._nextEmptyPairRow()
        s0 = f"{ijk0[0]},{ijk0[1]},{ijk0[2]}"
        s1 = f"{ijk1[0]},{ijk1[1]},{ijk1[2]}"
        self._setPairRowStrings(row, s0, s1)

        # exit place mode
        interactionNode = slicer.mrmlScene.GetNodeByID("vtkMRMLInteractionNodeSingleton")
        interactionNode.SetPlaceModePersistence(0)
        interactionNode.SetCurrentInteractionMode(interactionNode.ViewTransform)

        # stop observing
        if getattr(self, "_pairObserverTag", None) is not None:
            fid.RemoveObserver(self._pairObserverTag)
            self._pairObserverTag = None

        slicer.util.showStatusMessage(f"Saved point pair to row {row}.", 2000)


    # conversion
    def _rasToIJK(self, volumeNode, ras_world):
        ras_h = [ras_world[0], ras_world[1], ras_world[2], 1.0]
        m = vtk.vtkMatrix4x4()
        volumeNode.GetRASToIJKMatrix(m)

        ijk_h = [0.0, 0.0, 0.0, 1.0]
        m.MultiplyPoint(ras_h, ijk_h)

        return (int(round(ijk_h[0])), int(round(ijk_h[1])), int(round(ijk_h[2])))

    # table helper
    def _nextEmptyPairRow(self):
        t = self.ui.pointsTable
        for r in range(t.rowCount):
            a = t.item(r, 0).text().strip() if t.item(r, 0) else ""
            b = t.item(r, 1).text().strip() if t.item(r, 1) else ""
            if a == "" and b == "":
                return r
        # if none empty, append a row
        r = t.rowCount
        t.setRowCount(r + 1)
        # make sure items exist
        t.setItem(r, 0, QTableWidgetItem(""))
        t.setItem(r, 1, QTableWidgetItem(""))
        return r

    # table helper
    def _setPairRowStrings(self, row, s1, s2):
        t = self.ui.pointsTable
        # ensure items exist
        if t.item(row, 0) is None:
            t.setItem(row, 0, QTableWidgetItem(""))
        if t.item(row, 1) is None:
            t.setItem(row, 1, QTableWidgetItem(""))
        t.item(row, 0).setText(s1)
        t.item(row, 1).setText(s2)

    # pull coords from table for use in Analysis
    def _parseIJKString(self, s):
        if s is None:
            return None
        s = s.strip()
        if s == "":
            return None

        parts = [p.strip() for p in s.split(",")]
        if len(parts) != 3:
            return None

        try:
            i, j, k = (int(round(float(parts[0]))),
                    int(round(float(parts[1]))),
                    int(round(float(parts[2]))))
        except ValueError:
            return None

        return Point(i, j, k) # return point object of the values stored in the cell (assuming that it's valid)

    def getPointPairsFromTable(self):
        t = self.ui.pointsTable
        pairs = []

        for r in range(t.rowCount):
            item1 = t.item(r, 0)
            item2 = t.item(r, 1)

            s1 = item1.text() if item1 else ""
            s2 = item2.text() if item2 else ""

            p1 = self._parseIJKString(s1)
            p2 = self._parseIJKString(s2)

            # skip fully empty rows
            if p1 is None and p2 is None:
                continue

            # if one is missing/invalid, error out (or skip—your call)
            if p1 is None or p2 is None:
                raise ValueError(f"Row {r+1} is incomplete or invalid. Expected 'i,j,k' in both columns.")

            pairs.append((p1, p2))

        return pairs

    def onExecuteButton(self):
        try:
            point_pairs = self.getPointPairsFromTable()
        except ValueError as e:
            slicer.util.errorDisplay(str(e))
            return

        if not point_pairs:
            slicer.util.errorDisplay("No valid point pairs found in the table.")
            return
        
        
        analysis = Analysis()
        sitk_img = sitkUtils.PullVolumeFromSlicer(self._parameterNode.inputVolume) # bring slicer volume to itk image

        print()
        print('-'*30)

        results = [] # store for csv output

        for p1, p2 in point_pairs:
            for cutoff in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]:
                print("Pair:", (p1.x, p1.y, p1.z), (p2.x, p2.y, p2.z))
                roi, mask, bbox, rho = analysis.extractSphericalROI(sitk_img, p1, p2)

                frac, E_total, E_high, arr, arr_filt, mag_log, mag_filt_log = analysis.frequency_filter(roi, cutoff=cutoff)

                if self.ui.returnImagesCheckbox.isChecked():
                    base = self._ptname(p1)

                    self.pushToSlicer(roi, base)
                    self.pushToSlicer(sitk.GetImageFromArray(arr_filt), f"{base}_filtered_image")
                    self.pushToSlicer(sitk.GetImageFromArray(mag_log), f"{base}_frequency")
                    self.pushToSlicer(sitk.GetImageFromArray(mag_filt_log), f"{base}_filtered_frequency")

                print(f'Energy fraction: {frac}')
                print()

                results.append({
                    "p1_x": int(p1.x),
                    "p1_y": int(p1.y),
                    "p1_z": int(p1.z),
                    "p2_x": int(p2.x),
                    "p2_y": int(p2.y),
                    "p2_z": int(p2.z),
                    'cutoff': cutoff,
                    'recist_diameter': rho,
                    "energy_fraction": float(frac),
                })

        df = pd.DataFrame(results)

        out_path, _ = QFileDialog.getSaveFileName(
            None,
            "Save Energy Fraction Results",
            "",
            "CSV files (*.csv)"
        )

        if not out_path:
            print("Save cancelled.")
            return

        file_exists = os.path.exists(out_path)

        df.to_csv(
            out_path,
            mode='a' if file_exists else 'w',
            header=not file_exists,   # only write header once
            index=False
        )

        print(f"{'Appended to' if file_exists else 'Saved'} results at: {out_path}")


    def pushToSlicer(self, img, title):
        volumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        volumeNode.SetName(title)
        volumeNode.CreateDefaultDisplayNodes()
        sitkUtils.PushVolumeToSlicer(img, volumeNode)

    def _ptname(self, p):
        return f"{int(round(p.x))}_{int(round(p.y))}_{int(round(p.z))}"

class Point:
    def __init__(self, x, y, z):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

class Analysis:
    def __init__(self):
        pass

    # sitk image, Point(), Point()
    @staticmethod
    def extractSphericalROI(image, p1, p2, alpha=0.25):
        image = sitk.GetArrayFromImage(image)

        midpoint = Point((p1.x + p2.x)/2, (p1.y + p2.y)/2, (p1.z + p2.z)/2) # midpoint between fiducial point vectors
        rho = np.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2 + (p2.z - p1.z)**2) # the diameter in a sphere around the airway
        # print('midpoint', midpoint.x, midpoint.y, midpoint.z)
        # print('rho of spherical coord', rho)

        # create box around midpoint that encompasses the sphere
        shape = image.shape
        zmin = max(int(midpoint.z - rho), 0)
        zmax = min(int(midpoint.z + rho + 1), shape[0])
        ymin = max(int(midpoint.y - rho), 0)
        ymax = min(int(midpoint.y + rho + 1), shape[1])
        xmin = max(int(midpoint.x - rho), 0)
        xmax = min(int(midpoint.x + rho + 1), shape[2])
        
        # crop image to box
        sub_image = image[zmin:zmax, ymin:ymax, xmin:xmax].astype(np.float32, copy=False)

        # create grid for computation
        zz, yy, xx = np.ogrid[zmin:zmax, ymin:ymax, xmin:xmax]
        r = np.sqrt((zz - midpoint.z)**2 + (yy - midpoint.y)**2 + (xx - midpoint.x)**2).astype(np.float32, copy=False) # create ball on grid with rho

        # cutoff
        r0 = rho * (1.0 - alpha)
        soft_mask = np.ones_like(r, dtype=np.float32)

        soft_mask[r > rho] = 0.0 # zero out everything after cutoff
        taper = (r > r0) & (r <= rho) # boolean to determine outside cutoff
        soft_mask[taper] = 0.5 * (1.0 + np.cos(np.pi * (r[taper] - r0) / (rho - r0))) # fade out anything past cutoff

        sub_masked = sitk.GetImageFromArray(sub_image * soft_mask) # mask the image

        bbox = (zmin, zmax, ymin, ymax, xmin, xmax) # return the bounding box of the image to relate to full image if needed
        return sub_masked, soft_mask, bbox, rho
    
    # sitk image, cutoff for ideal HP filter
    @staticmethod
    def frequency_filter(image, cutoff=0.4):
        img = sitk.Cast(image, sitk.sitkFloat32)
        arr = sitk.GetArrayFromImage(img).astype(np.float32, copy=False)
        arr -= arr.mean(dtype=np.float32)

        # isotrpoic, even sided cube
        Z, Y, X = arr.shape
        m = min(Z, Y, X)
        if m % 2 != 0:
            m -= 1

        z0 = (Z - m) // 2
        y0 = (Y - m) // 2
        x0 = (X - m) // 2
        arr = arr[z0:z0+m, y0:y0+m, x0:x0+m]
        # np.fft.fftshift shifts frequency domain to center DC, radial frequencies, np.fft.fftn computes 3D DFT
        F = np.fft.fftshift(np.fft.fftn(arr))
        mag2 = (np.abs(F) ** 2).astype(np.float64, copy=False) # we care about the magnitudes (power since we're squaring for energy)

        f = np.fft.fftfreq(m, d=1.0) # convert to cycles per sample -- how many times does wave oscillate in one voxel

        # create grid of the frequencies
        fz = f[:, None, None]
        fy = f[None, :, None]
        fx = f[None, None, :]

        r = np.sqrt(fz**2 + fy**2 + fx**2) # create rho for spherical HP ideal mask
        r = np.fft.fftshift(r) # shift mask to align with fourier magnitudes

        f_c = cutoff * 0.5  # nyquist is 1/2 max frequency. our filter is a fraction of the nyquist rate, so 1/2 * cutoff, where cutoff is portion of image
        H = (r >= f_c).astype(np.float32)  # ideal

        E_total = mag2.sum(dtype=np.float64) # sum of power = energy
        E_high = (mag2 * (H**2)).sum(dtype=np.float64) # energy of masked
        frac = E_high / (E_total + 1e-12) # ef

        # visualization
        F_filt = F * H
        arr_filt = np.real(np.fft.ifftn(np.fft.ifftshift(F_filt))).astype(np.float32, copy=False)
        mag_log = np.log1p(np.abs(F)).astype(np.float32, copy=False)
        mag_filt_log = np.log1p(np.abs(F_filt)).astype(np.float32, copy=False)

        return frac, E_total, E_high, arr, arr_filt, mag_log, mag_filt_log