import logging
import os
from typing import Annotated
import vtk
import slicer   
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)
import sitkUtils
from qt import QFileDialog

from slicer import vtkMRMLScalarVolumeNode
import SimpleITK as sitk
import numpy as np
import pandas as pd
from scipy.signal.windows import tukey, hann


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
        self.parent.contributors = ["Jacob Kitmzann"]
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
        
        self.ui.dropSeedpointButton.connect('clicked(bool)', self.onDropSeedpointButton)
        self.ui.printCentroidButton.connect('clicked(bool)', self.onPrintCentroidButton)
        self.ui.createResultsButton.connect('clicked(bool)', self.onCreateResultsButton)
        

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
    def setParameterNode(self, inputParameterNode: AirwayEnergyFractionParameterNode | None) -> None:
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
        self._parameterNode = inputParameterNode
        if self._parameterNode:
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            
            
    # -- my code --
    def onDropSeedpointButton(self):
        volumeNode = self._parameterNode.inputVolume
        if not volumeNode:
            print("No input volume selected.")
            return

        centroidNode = self._getCentroidNodeForVolume(volumeNode, createIfMissing=True)

        markupsLogic = slicer.modules.markups.logic()
        markupsLogic.SetActiveList(centroidNode)

        # 0 = place exactly one fiducial
        markupsLogic.StartPlaceMode(0)



    def onPrintCentroidButton(self):
        volumeNode = self._parameterNode.inputVolume # works on current volume in combo box
        if not volumeNode:
            print("No active input volume.")
            return

        # don't want to make a new one if no centroids defined
        centroidNode = self._getCentroidNodeForVolume(volumeNode, createIfMissing=False)
        if not centroidNode or centroidNode.GetNumberOfControlPoints() == 0:
            print(f"No centroids defined for volume: {volumeNode.GetName()}")
            return

        # coordinate conversion
        rasToIjk = vtk.vtkMatrix4x4()
        volumeNode.GetRASToIJKMatrix(rasToIjk)

        n = centroidNode.GetNumberOfControlPoints()
        print(f"Volume: {volumeNode.GetName()} | Number of centroids: {n}")

        for i in range(n):
            ras = [0.0, 0.0, 0.0]
            centroidNode.GetNthControlPointPosition(i, ras)

            ijkH = [0.0, 0.0, 0.0, 0.0]
            rasToIjk.MultiplyPoint([ras[0], ras[1], ras[2], 1.0], ijkH)

            ijk = [int(round(ijkH[0])), int(round(ijkH[1])), int(round(ijkH[2]))]
            print(f"Centroid {i} IJK: {ijk}")

                
    def onCreateResultsButton(self):

        volumeNode = self._parameterNode.inputVolume
        if not volumeNode:
            print("No input volume selected.")
            return

        centroidNode = self._getCentroidNodeForVolume(volumeNode, createIfMissing=False)
        if not centroidNode or centroidNode.GetNumberOfControlPoints() == 0:
            print(f"No centroids defined for volume: {volumeNode.GetName()}")
            return

        sitk_img = sitkUtils.PullVolumeFromSlicer(volumeNode) # bring slicer volume to itk image

        # coord system conversion
        rasToIjk = vtk.vtkMatrix4x4()
        volumeNode.GetRASToIJKMatrix(rasToIjk)

        # convert
        centroids_ijk = []
        for i in range(centroidNode.GetNumberOfControlPoints()):
            ras = [0.0, 0.0, 0.0]
            centroidNode.GetNthControlPointPosition(i, ras)

            ijkH = [0.0, 0.0, 0.0, 0.0]
            rasToIjk.MultiplyPoint([ras[0], ras[1], ras[2], 1.0], ijkH)

            ijk = [int(round(v)) for v in ijkH[:3]]
            centroids_ijk.append(ijk)

        # initialize obj
        try:
            energyFraction = EnergyFraction(int(self.ui.windowEdit.text))
        except:
            print("No window defined")
            return

        # run pipeline
        energyFraction.set_centroids(centroids_ijk)
        energyFraction.set_input_volume(sitk_img)
        energyFraction.execute()

        # visualize rois created for debug if checked
        if self.ui.checkBox.isChecked():
            volName = self._sanitizeName(volumeNode.GetName() or "Volume")
            print(f"displaying rois for {volName}...")

            for idx, centroid in enumerate(centroids_ijk):
                roi = EnergyFraction.extract_roi(
                    sitk_img,
                    centroid,
                    window=int(self.ui.windowEdit.text)
                )
                
                # apply Tukey windowing filter, doc'd in EF class (frequency_filter).
                arr = sitk.GetArrayFromImage(roi).astype(np.float32, copy=False)
                
                arr -= arr.mean(dtype=np.float32)
                alpha = 0.25
                Nz, Ny, Nx = arr.shape
                # scipy tukey filters
                wz = tukey(Nz, alpha)
                wy = tukey(Ny, alpha)
                wx = tukey(Nx, alpha)
                tukey_window = (
                    wz[:, None, None] *
                    wy[None, :, None] *
                    wx[None, None, :]
                )
                tukey_window = tukey_window.astype(np.float32, copy=False)

                arr *= tukey_window 
                roi_windowed = sitk.GetImageFromArray(arr)

                # display ROI
                roiVolumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
                roiVolumeNode.SetName(f"{volName}_ROI_Centroid_{idx}")
                roiVolumeNode.CreateDefaultDisplayNodes()
                roi_windowed.CopyInformation(roi)   # copy spacing/origin/direction from original ROI
                sitkUtils.PushVolumeToSlicer(roi_windowed, roiVolumeNode)
                
                # display magnitude 
                F = np.fft.fftshift(np.fft.fftn(arr))
                mag_log = np.log1p(np.abs(F)).astype(np.float32, copy=False)
                mag_img = sitk.GetImageFromArray(mag_log.astype(np.float32, copy=False))
                mag_img.CopyInformation(roi)
                magNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
                magNode.SetName(f"{volName}_FFTMag_C{idx}")
                magNode.CreateDefaultDisplayNodes()
                sitkUtils.PushVolumeToSlicer(mag_img, magNode)
                
            print("rois and magnitudes displayed")

    # make names nice (because people like putting spaces in file names)
    def _sanitizeName(self, name: str) -> str:
        return "".join(c if c.isalnum() or c in ("_", "-", " ") else "_" for c in name).strip()

    # only select centroid for volume selected
    def _getCentroidNodeForVolume(self, volumeNode, createIfMissing=True):
        if not volumeNode:
            return None

        volId = volumeNode.GetID()
        if volId in self.centroidFiducialNodesByVolumeId:
            node = self.centroidFiducialNodesByVolumeId[volId]
            # If it got deleted from the scene, drop it and recreate if requested
            if node and slicer.mrmlScene.IsNodePresent(node):
                return node
            else:
                self.centroidFiducialNodesByVolumeId.pop(volId, None)

        if not createIfMissing:
            return None

        volName = self._sanitizeName(volumeNode.GetName() or "Volume")
        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")
        node.SetName(f"Centroids_{volName}")
        node.CreateDefaultDisplayNodes()

        # Optional: store association on the node itself (nice for debugging)
        node.SetAttribute("AirwayEnergyFraction.VolumeID", volId)
        node.SetAttribute("AirwayEnergyFraction.VolumeName", volumeNode.GetName() or "")

        self.centroidFiducialNodesByVolumeId[volId] = node
        return node


            
class EnergyFraction:

    def __init__(self, window=-1):
        self.centroids = []
        self.input_volume = None
        self.window = window

    def set_centroids(self, centroids):
        self.centroids = centroids

    def set_input_volume(self, input_volume):
        self.input_volume = input_volume

    def execute(self):
        print()
        print('-'*30)
        print(f"Running energy fraction on {len(self.centroids)} centroids")
        print(f'ROI Size: {2 * self.window + 1} voxels in each direction')
        print()

        cutoffs = [0.05, 0.1, 0.2, 0.4, 0.6]
        header = ['Centroid I', 'Centroid J', 'Centroid K'] + [f'EnergyFrac_{c}' for c in cutoffs]
        rows = []

        for centroid in self.centroids:
            roi = self.extract_roi(self.input_volume, centroid, window=self.window)
            print(f"ROI size: {roi.GetSize()}")

            row = [centroid[0], centroid[1], centroid[2]]

            print(f'\nCentroid {centroid}')
            for cutoff in cutoffs:
                frac, _, _, _ = self.frequency_filter(roi, cutoff=cutoff)
                row.append(frac)
                print(f'  cutoff={cutoff:.2f}: {frac*100:.2f}% high energy')

            rows.append(row)

        df = pd.DataFrame(rows, columns=header)

        out_path = QFileDialog.getSaveFileName(
            None,
            "Save Energy Fraction Results",
            "",
            "CSV files (*.csv)"
        )

        if out_path:
            df.to_csv(out_path, index=False)
            print(f"Saved results to: {out_path}")
        else:
            print("Save cancelled.")
            
            
    # corner freq, order for butterworth highpass, alpha for tukey filter 
    @staticmethod
    def frequency_filter(image, cutoff=0.1, order=2, alpha=0.25, return_mag=False):
        
        if image.GetNumberOfComponentsPerPixel() > 1:
            image = sitk.VectorIndexSelectionCast(image, 0)

        img = sitk.Cast(image, sitk.sitkFloat32)
        arr = sitk.GetArrayFromImage(img).astype(np.float32, copy=False)
        arr -= arr.mean(dtype=np.float32)

        # for 3d fourier dimension match
        # make into an isotropic, maximally even sided cube
        Z, Y, X = arr.shape
        m = min(Z, Y, X)
        if m % 2 != 0:
            m -= 1

        z0 = (Z - m) // 2
        y0 = (Y - m) // 2
        x0 = (X - m) // 2

        arr = arr[z0:z0+m, y0:y0+m, x0:x0+m]
        
        Nz, Ny, Nx = arr.shape

        # scipy tukey filters
        wz = tukey(Nz, alpha)
        wy = tukey(Ny, alpha)
        wx = tukey(Nx, alpha)

        tukey_window = (
            wz[:, None, None] *
            wy[None, :, None] *
            wx[None, None, :]
        ).astype(np.float32, copy=False)

        arr *= tukey_window # apply tukey_window to the ROI image to avoid hard 
        # cropping issues and mitigate human error between centroid placements

        F = np.fft.fftn(arr)
        F = np.fft.fftshift(F)
        
        mag2 = np.abs(F)**2

        z, y, x = np.ogrid[:m, :m, :m]
        c = m // 2
        r = np.sqrt((z-c)**2 + (y-c)**2 + (x-c)**2)
        r_norm = r / r.max()

        H = 1 / (1 + (cutoff / (r_norm + 1e-12))**(2*order))

        E_total = mag2.sum()
        E_high = (mag2 * (H**2)).sum()
        frac = E_high / (E_total + 1e-12)
        return frac, E_total, E_high, arr
    
    @staticmethod
    def extract_roi(img, centroid, window=50):
        x, y, z = centroid
        size = img.GetSize()
        
        x_min = max(x - window, 0)
        x_max = min(x + window, size[0] - 1)
        y_min = max(y - window, 0)
        y_max = min(y + window, size[1] - 1)
        z_min = max(z - window, 0)
        z_max = min(z + window, size[2] - 1)

        roi_size  = [x_max - x_min + 1,
                    y_max - y_min + 1,
                    z_max - z_min + 1]

        roi_index = [x_min, y_min, z_min]
        
        roi = sitk.RegionOfInterest(img, index=roi_index, size=roi_size)
        
        # recreate metadata
        original_origin = img.TransformIndexToPhysicalPoint(roi_index)
        roi.SetOrigin(original_origin)
        roi.SetDirection(img.GetDirection())
        roi.SetSpacing(img.GetSpacing())
        
        return roi
