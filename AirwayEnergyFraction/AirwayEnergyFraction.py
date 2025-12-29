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
        self.centroidFiducialNode = None
        
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
        # Create or reuse centroid fiducial node
        if not self.centroidFiducialNode:
            self.centroidFiducialNode = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLMarkupsFiducialNode"
            )
            self.centroidFiducialNode.SetName("Centroids")
            self.centroidFiducialNode.CreateDefaultDisplayNodes()

        markupsLogic = slicer.modules.markups.logic()
        markupsLogic.SetActiveList(self.centroidFiducialNode)

        # 0 = place exactly one fiducial
        markupsLogic.StartPlaceMode(0)


    def onPrintCentroidButton(self):
        if not self.centroidFiducialNode:
            print("No centroids defined.")
            return

        volumeNode = self._parameterNode.inputVolume
        if not volumeNode:
            print("No active input volume.")
            return

        # Get RAS → IJK matrix
        rasToIjk = vtk.vtkMatrix4x4()
        volumeNode.GetRASToIJKMatrix(rasToIjk)

        n = self.centroidFiducialNode.GetNumberOfControlPoints()
        print(f"Number of centroids: {n}")

        for i in range(n):
            ras = [0.0, 0.0, 0.0]
            self.centroidFiducialNode.GetNthControlPointPosition(i, ras)

            rasH = [ras[0], ras[1], ras[2], 1.0]
            ijkH = [0.0, 0.0, 0.0, 0.0]
            rasToIjk.MultiplyPoint(rasH, ijkH)

            ijk = [int(round(ijkH[0])),
                int(round(ijkH[1])),
                int(round(ijkH[2]))]

            print(f"Centroid {i} IJK: {ijk}")
            
    def onCreateResultsButton(self):

        if not self.centroidFiducialNode:
            print("No centroids defined.")
            return

        volumeNode = self._parameterNode.inputVolume
        if not volumeNode:
            print("No input volume selected.")
            return

        sitk_img = sitkUtils.PullVolumeFromSlicer(volumeNode)


        # ras to ijk
        rasToIjk = vtk.vtkMatrix4x4()
        volumeNode.GetRASToIJKMatrix(rasToIjk)

        centroids_ijk = []
                
        for i in range(self.centroidFiducialNode.GetNumberOfControlPoints()):
            ras = [0.0, 0.0, 0.0]
            self.centroidFiducialNode.GetNthControlPointPosition(i, ras)

            ijkH = [0.0, 0.0, 0.0, 0.0]
            rasToIjk.MultiplyPoint([ras[0], ras[1], ras[2], 1.0], ijkH)

            ijk = [int(round(v)) for v in ijkH[:3]]
            centroids_ijk.append(ijk)

        # analysis
        
        try:
            energyFraction = EnergyFraction(int(self.ui.windowEdit.text))
        except: 
            print('No window defined')
            return
        
        energyFraction.set_centroids(centroids_ijk)
        energyFraction.set_input_volume(sitk_img)
        energyFraction.execute()
        
        # visualize rois
        if self.ui.checkBox.isChecked():
            print('displaying rois...')
        
            for idx, centroid in enumerate(centroids_ijk):

                roi = EnergyFraction.extract_roi(
                    sitk_img,
                    centroid,
                    window=int(self.ui.windowEdit.text)
                )

                roiVolumeNode = slicer.mrmlScene.AddNewNodeByClass(
                    "vtkMRMLScalarVolumeNode"
                )
                roiVolumeNode.SetName(f"ROI_Centroid_{idx}")
                roiVolumeNode.CreateDefaultDisplayNodes()

                sitkUtils.PushVolumeToSlicer(roi, roiVolumeNode)

            print('rois displayed')

            
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
            "Save Energy Fraction R`esults",
            "",
            "CSV files (*.csv)"
        )

        if out_path:
            df.to_csv(out_path, index=False)
            print(f"Saved results to: {out_path}")
        else:
            print("Save cancelled.")
        
    @staticmethod
    def frequency_filter(image, cutoff=0.1, order=2):

        if image.GetNumberOfComponentsPerPixel() > 1:
            image = sitk.VectorIndexSelectionCast(image, 0)

        img = sitk.Cast(image, sitk.sitkFloat32)
        arr = sitk.GetArrayFromImage(img)

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
