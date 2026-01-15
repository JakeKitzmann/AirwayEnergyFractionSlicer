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
        self.pointPairsByVolumeId = {}   # volumeID -> list[tuple[str,str]]
        self._activeVolumeId = None
        self._updatingTable = False

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
        self.ui.executeButton.connect('clicked(bool)', self.onExecuteButton)
        self.ui.inputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onInputVolumeChanged)
        self.ui.pointsTable.itemChanged.connect(self.onPointsTableEdited)

         # initialize active volume + table
        currentVol = self._parameterNode.inputVolume
        self.onInputVolumeChanged(currentVol)


    # initialize table that stores all of the points the user has entered
    def setupPointsTable(self):
        t = self.ui.pointsTable
        t.setRowCount(10)
        t.setColumnCount(2)
        t.setHorizontalHeaderLabels(['Point 1', 'Point 2'])

        t.verticalHeader().setVisible(True)
        t.horizontalHeader().setStretchLastSection(True) # continue if user adds more rows

        # fill with empty items so setText later never crashes
        for r in range(t.rowCount):
            for c in range(t.columnCount):
                if t.item(r, c) is None:
                    t.setItem(r, c, QTableWidgetItem(""))

    # place 2 points to caputure an airway
    def onPointPairButton(self):

        # current volume selected in combo box
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
        else: # if there is one, clear it and use that one
            self._pairFidNode.RemoveAllControlPoints()

        # movable
        self._pairFidNode.SetLocked(False)

        # dont stack observers on repeat call
        if getattr(self, "_pairObserverTag", None) is not None: 
            self._pairFidNode.RemoveObserver(self._pairObserverTag)
        self._pairObserverTag = self._pairFidNode.AddObserver(
            slicer.vtkMRMLMarkupsNode.PointModifiedEvent, self._onPairPointsModified
        )

        # create points for placement
        selectionNode = slicer.mrmlScene.GetNodeByID("vtkMRMLSelectionNodeSingleton")
        selectionNode.SetReferenceActivePlaceNodeID(self._pairFidNode.GetID())

        # 1 means continually place, 0 would be single placement
        interactionNode = slicer.mrmlScene.GetNodeByID("vtkMRMLInteractionNodeSingleton")
        interactionNode.SetPlaceModePersistence(1)     
        interactionNode.SetCurrentInteractionMode(interactionNode.Place)

    # read the table into Point pairs for use in analysis
    def _tableToPairs(self):
        t = self.ui.pointsTable
        pairs = []
        for r in range(t.rowCount):
            s1 = t.item(r, 0).text().strip() if t.item(r, 0) else ""
            s2 = t.item(r, 1).text().strip() if t.item(r, 1) else ""
            if s1 == "" and s2 == "":
                continue
            pairs.append((s1, s2))
        return pairs

    # save the table information into the dict for the active volume
    def _saveTableForActiveVolume(self):
        if self._activeVolumeId is None:
            return
        self.pointPairsByVolumeId[self._activeVolumeId] = self._tableToPairs()

    # every time we edit the table, it should save to dict. This allows user to manually edit the point pairs
    def onPointsTableEdited(self, item):
        if self._updatingTable:
            return
        # Save on every edit
        self._saveTableForActiveVolume()

    # clear the table when no volume is selected
    def _clearTable(self):
        t = self.ui.pointsTable
        self._updatingTable = True
        try:
            for r in range(t.rowCount):
                for c in range(t.columnCount):
                    item = t.item(r, c)
                    if item is None:
                        t.setItem(r, c, QTableWidgetItem(""))
                    else:
                        item.setText("")
        finally:
            self._updatingTable = False

    # populate the table with Point pairs for the active volume
    def _pairsToTable(self, pairs):
        t = self.ui.pointsTable
        self._updatingTable = True
        try:
            # ensure enough rows
            if len(pairs) > t.rowCount:
                t.setRowCount(len(pairs))

            # clear first
            for r in range(t.rowCount):
                for c in range(t.columnCount):
                    item = t.item(r, c)
                    if item is None:
                        t.setItem(r, c, QTableWidgetItem(""))
                    else:
                        item.setText("")

            # populate
            for r, (s1, s2) in enumerate(pairs):
                if t.item(r, 0) is None:
                    t.setItem(r, 0, QTableWidgetItem(""))
                if t.item(r, 1) is None:
                    t.setItem(r, 1, QTableWidgetItem(""))
                t.item(r, 0).setText(s1)
                t.item(r, 1).setText(s2)
        finally:
            self._updatingTable = False

    # when user changes the input volume, save current table and load new one
    def onInputVolumeChanged(self, newNode):
        self._saveTableForActiveVolume()
        self._activeVolumeId = newNode.GetID() if newNode is not None else None
        if self._activeVolumeId is None:
            self._clearTable()
            return

        pairs = self.pointPairsByVolumeId.get(self._activeVolumeId, [])
        self._pairsToTable(pairs)

    # when user places 2 points, save them into the table
    def _onPairPointsModified(self, caller, event):
        fid = self._pairFidNode
        if fid is None:
            return

        # need 2 points to make a pair
        if fid.GetNumberOfDefinedControlPoints() < 2:
            return

        ras0 = [0.0, 0.0, 0.0]
        ras1 = [0.0, 0.0, 0.0]
                
        fid.GetNthControlPointPositionWorld(0, ras0)
        fid.GetNthControlPointPositionWorld(1, ras1)

        ijk0 = self._worldToIJK(self._pairVolumeNode, ras0)
        ijk1 = self._worldToIJK(self._pairVolumeNode, ras1)

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

    # this function allows us to place fiducial points in global (world) coordinates and get the corresponding IJK in the selected volume
    def _worldToIJK(self, volumeNode, world_xyz):

        worldToVolumeRAS = vtk.vtkGeneralTransform()
        slicer.vtkMRMLTransformNode.GetTransformBetweenNodes(
            None,  # from World
            volumeNode.GetParentTransformNode(),  # to volume's parent transform (volume local)
            worldToVolumeRAS
        ) # this creates transform to volume RAS

        ras = worldToVolumeRAS.TransformPoint(world_xyz) # apply transform

        # create matrix for multiplication conversion to ijk
        m = vtk.vtkMatrix4x4()
        volumeNode.GetRASToIJKMatrix(m)

        # multiply conversion matrix
        ijk_h = [0.0, 0.0, 0.0, 1.0]
        m.MultiplyPoint([ras[0], ras[1], ras[2], 1.0], ijk_h)

        # round because voxels are int
        return (int(round(ijk_h[0])), int(round(ijk_h[1])), int(round(ijk_h[2])))


    # find next empty row in table, or append one if none exist
    def _nextEmptyPairRow(self):
        t = self.ui.pointsTable

        # if there's a row with both empty, return that
        for r in range(t.rowCount):
            a = t.item(r, 0).text().strip() if t.item(r, 0) else ""
            b = t.item(r, 1).text().strip() if t.item(r, 1) else ""
            if a == "" and b == "":
                return r
            
        # if none empty, append a row
        r = t.rowCount
        t.setRowCount(r + 1)

        # create row
        t.setItem(r, 0, QTableWidgetItem(""))
        t.setItem(r, 1, QTableWidgetItem(""))
        return r

    # set the strings in a given row (used when placing points)
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

    # get point pair strings from table for analysis
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


    # run the analysis on all point pairs for the active volume
    def onExecuteButton(self):
        self._saveTableForActiveVolume() # save current table before running, ensures user edits are captured

        # get point pairs
        try:
            point_pairs = self.getPointPairsFromTable()
        except ValueError as e:
            slicer.util.errorDisplay(str(e))
            return

        if not point_pairs:
            slicer.util.errorDisplay("No valid point pairs found in the table.")
            return
        
        # init analysis object, pull volume as itk image
        analysis = Analysis()
        sitk_img = sitkUtils.PullVolumeFromSlicer(self._parameterNode.inputVolume) # bring slicer volume to itk image

        print()
        print('-'*30)

        results = [] # store for csv output

        cutoff_pairs = [
            (0.1, self.ui.cutoff1Checkbox.isChecked()),
            (0.2, self.ui.cutoff2Checkbox.isChecked()),
            (0.3, self.ui.cutoff3Checkbox.isChecked()),
            (0.4, self.ui.cutoff4Checkbox.isChecked()),
            (0.6, self.ui.cutoff6Checkbox.isChecked()),
            (0.8, self.ui.cutoff8Checkbox.isChecked()),
        ]

        checkboxList = [c for c, checked in cutoff_pairs if checked]
        print("Selected cutoffs:", checkboxList)

        radius_pairs = [
            (0.5, self.ui.radius05Checkbox.isChecked()),
            (1.0, self.ui.radius1Checkbox.isChecked()),
            (1.5, self.ui.radius15Checkbox.isChecked()),
            (2.0, self.ui.radius2Checkbox.isChecked()),
            (2.5, self.ui.radius25Checkbox.isChecked()),
        ]

        radiusList = [r for r, checked in radius_pairs if checked]
        print("Selected radii:", radiusList)


        # for all point pairs
        for p1, p2 in point_pairs:
            for radius_mult in radiusList: # for different radius multipliers
                 # get the spherical ROI
                roi, mask, bbox, rho, rho_mm, recist = \
                    analysis.extractROI(sitk_img, p1, p2, window=self.ui.taperRoiCheckbox.isChecked(), alpha=0.25, radius_multiplier=radius_mult, spherical=self.ui.sphericalRadioButton.isChecked())

                for cutoff in checkboxList: # for different cycle/mm cutoffs, put here so we don't have to recompute ROI

                    # compute frequency filter and energy fraction of our ROI
                    frac, E_total, E_high, img_cube, arr, arr_filt, mag_log, mag_filt_log = \
                        analysis.frequencyFilter3D(roi, cutoff=cutoff)
                    
                    D2_frac, D2_E_total, D2_E_high, D2_img_sq, D2_arr, D2_arr_filt, D2_mag_log, D2_mag_filt_log = \
                        analysis.frequencyFilter2D(img_cube, cutoff=cutoff)



                    # recreate sitk filtered image and copy metadata
                    img_filt = sitk.GetImageFromArray(arr_filt)
                    img_filt.CopyInformation(img_cube)

                    # for output naming
                    volumeNode = self._parameterNode.inputVolume
                    volume_name = volumeNode.GetName()
                    volume_id = volumeNode.GetID()

                    # push to slicer if desired, big time slowdown here
                    if self.ui.returnImagesCheckbox.isChecked():
                        base = self._ptname(p1)
                        self.pushToSlicer(roi, f'{volume_name}_{base}_cutoff-{cutoff}_radiusmult-{radius_mult}')
                        self.pushToSlicer(sitk.GetImageFromArray(arr_filt), f"{volume_name}_{base}_cutoff-{cutoff}_radiusmult-{radius_mult}_filtered_image")
                        self.pushToSlicer(sitk.GetImageFromArray(mag_log), f"{volume_name}_{base}_cutoff-{cutoff}_radiusmult-{radius_mult}_frequency")
                        self.pushToSlicer(sitk.GetImageFromArray(mag_filt_log), f"{volume_name}_{base}_cutoff-{cutoff}_radiusmult-{radius_mult}_filtered_frequency")

                        img2d = sitk.GetImageFromArray(D2_arr[None, ...].astype(np.float32, copy=False))
                        img2d.CopyInformation(D2_img_sq)
                        self.pushToSlicer(img2d,  f"{volume_name}_{base}_cutoff-{cutoff}_radiusmult-{radius_mult}_2D")

                        self.pushToSlicer(sitk.GetImageFromArray(D2_arr_filt), f"{volume_name}_{base}_cutoff-{cutoff}_radiusmult-{radius_mult}_filtered_image_2D")
                        self.pushToSlicer(sitk.GetImageFromArray(D2_mag_log), f"{volume_name}_{base}_cutoff-{cutoff}_radiusmult-{radius_mult}_frequency_2D")
                        self.pushToSlicer(sitk.GetImageFromArray(D2_mag_filt_log), f"{volume_name}_{base}_cutoff-{cutoff}_radiusmult-{radius_mult}_filtered_frequency_2D")
                        

                    # csv results
                    results.append({
                        'title':                self.ui.customLineEdit.text,
                        'volume_name':          volume_name,
                        'volume_id':            volume_id,
                        "p1_x":                 int(p1.x),
                        "p1_y":                 int(p1.y),
                        "p1_z":                 int(p1.z),
                        "p2_x":                 int(p2.x),
                        "p2_y":                 int(p2.y),
                        "p2_z":                 int(p2.z),
                        'cutoff':               cutoff,
                        'voxel_sphere_radius':  rho,
                        'mm_sphere_radius':     rho_mm,
                        'recist_diameter':      recist,
                        'radius_fraction':      radius_mult,
                        '2D_energy_fraction':    float(D2_frac),
                        "3D_energy_fraction":      float(frac),
                    })

        df = pd.DataFrame(results)

        out_path = QFileDialog.getSaveFileName(
            None,
            "Save Energy Fraction Results",
            "",
            "CSV files (*.csv)"
        )

        if out_path is None:
            print('save cancelled')
            return

        try:
            file_exists = os.path.exists(out_path)

            df.to_csv(
                out_path,
                mode='a' if file_exists else 'w', # append if exists
                header=not file_exists,   # only write header once
                index=False 
            )

            print(f"{'Appended to' if file_exists else 'Saved'} results at: {out_path}") # prettyness
        except Exception as e:
            slicer.util.errorDisplay(f"Failed to save results to {out_path}:\n{str(e)}")

    # push sitk image to slicer with given title
    def pushToSlicer(self, img, title):
        volumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        volumeNode.SetName(title)
        volumeNode.CreateDefaultDisplayNodes()
        sitkUtils.PushVolumeToSlicer(img, volumeNode)

    # sanitize name
    def _ptname(self, p):
        return f"{int(round(p.x))}_{int(round(p.y))}_{int(round(p.z))}"

# used to store ijk coordinates -- i wrote x y z because i'm dumb and bad at my job :) 
class Point:
    def __init__(self, x, y, z):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

# all logic from ipynb notebooks in here
class Analysis:
    def __init__(self):
        pass

    # sitk image, Point(), Point()
    # cosine tapering only available in spherical ROI, can implement later if needed
    @staticmethod
    def extractROI(image_sitk, p1, p2, window=True, alpha=0.25, radius_multiplier=1.0, spherical=False):
        image = sitk.GetArrayFromImage(image_sitk)  # arr order: (z,y,x)

        # find midpoint between p1 and p2
        midpoint = Point((p1.x + p2.x)/2, (p1.y + p2.y)/2, (p1.z + p2.z)/2)

        # endpoint distance in voxels for spherical roi radius
        diam_vxl = radius_multiplier * float(np.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2 + (p2.z - p1.z)**2))

        # endpoint distance in mm for spherical roi radius
        sx, sy, sz = image_sitk.GetSpacing()  # (x,y,z)
        d_mm = np.array([(p2.x - p1.x)*sx, (p2.y - p1.y)*sy, (p2.z - p1.z)*sz], dtype=np.float32)
        diam_mm = radius_multiplier * float(np.linalg.norm(d_mm))

        # radius in voxels and mm of big sphere
        # i'm changing to radius here because it makes more sense in the new context of the larger sphere
        radius_vxl = diam_vxl
        radius_mm  = diam_mm

        recist = radius_mm / radius_multiplier # original RECIST in mm

        # crop to box around sphere, different radii account for anistropic voxels
        rx = int(np.ceil(radius_mm / sx))  # x voxels
        ry = int(np.ceil(radius_mm / sy))  # y voxels
        rz = int(np.ceil(radius_mm / sz))  # z voxels

        shape = image.shape
        zmin = max(int(np.floor(midpoint.z - rz)), 0)
        zmax = min(int(np.ceil (midpoint.z + rz)) + 1, shape[0])
        ymin = max(int(np.floor(midpoint.y - ry)), 0)
        ymax = min(int(np.ceil (midpoint.y + ry)) + 1, shape[1])
        xmin = max(int(np.floor(midpoint.x - rx)), 0)
        xmax = min(int(np.ceil (midpoint.x + rx)) + 1, shape[2])
            
        # crop image to box
        sub_image = image[zmin:zmax, ymin:ymax, xmin:xmax].astype(np.float32, copy=False)

        # image spacing
        sx, sy, sz = image_sitk.GetSpacing()

        # voxel grid coordinates of full image
        zz, yy, xx = np.ogrid[zmin:zmax, ymin:ymax, xmin:xmax]

        # radius grid in mm
        dz = (zz - midpoint.z) * sz
        dy = (yy - midpoint.y) * sy
        dx = (xx - midpoint.x) * sx
        r_mm_grid = np.sqrt(dz*dz + dy*dy + dx*dx).astype(np.float32, copy=False)

        # build soft mask with cosine taper to avoid harsh boundary of sphere within ROI
        r0_mm = radius_mm * (1.0 - alpha)
        soft_mask = np.ones_like(r_mm_grid, dtype=np.float32)
        soft_mask[r_mm_grid > radius_mm] = 0.0 # anything outside radius is zero

        # only taper if wanted
        if window:
            taper = (r_mm_grid > r0_mm) & (r_mm_grid <= radius_mm) # taper to full value as we get closer to center
            soft_mask[taper] = 0.5 * (1.0 + np.cos(np.pi * (r_mm_grid[taper] - r0_mm) / (radius_mm - r0_mm))) # apply cosine taper along radius of sphere

        if spherical:
            sub_masked = sitk.GetImageFromArray(sub_image * soft_mask) # mask
        else:
            sub_masked = sitk.GetImageFromArray(sub_image)

        # recreate metadata
        sub_masked.SetSpacing(image_sitk.GetSpacing())
        sub_masked.SetDirection(image_sitk.GetDirection())
        crop_origin_physical = image_sitk.TransformIndexToPhysicalPoint((xmin, ymin, zmin))
        sub_masked.SetOrigin(crop_origin_physical)

        # return the bounding box of the image to relate to full image if ever needed
        bbox = (zmin, zmax, ymin, ymax, xmin, xmax) 
        return sub_masked, soft_mask, bbox, radius_vxl, radius_mm, recist
    
    @staticmethod
    def frequencyFilter2D(image, cutoff=0.4):
        img = sitk.Cast(image, sitk.sitkFloat32)

        # array view: (Z, Y, X)
        Z, Y, X = sitk.GetArrayViewFromImage(img).shape

        # crop to even-shaped SQUARE for 2D FFT (in-plane only)
        m = min(Y, X)
        if m % 2 != 0:
            m -= 1

        y0 = (Y - m) // 2
        x0 = (X - m) // 2

        zmid = Z // 2

        # ROI: one slice thick at zmid (size order is [x,y,z], index order is [x,y,z])
        img_sq = sitk.RegionOfInterest(img, size=[m, m, 1], index=[x0, y0, zmid])

        # 2D array (m,m)
        arr = sitk.GetArrayFromImage(img_sq)[0].astype(np.float32, copy=False)

        sx, sy, sz = img_sq.GetSpacing()  # (x,y,z)

        # 2D FFT
        F = np.fft.fftshift(np.fft.fft2(arr))
        mag2 = (np.abs(F) ** 2).astype(np.float64, copy=False)

        # build 2D freq grid in cycles/mm
        fy = np.fft.fftfreq(m, d=sy)[:, None]   # (m,1)
        fx = np.fft.fftfreq(m, d=sx)[None, :]   # (1,m)
        r = np.sqrt(fy**2 + fx**2)
        r = np.fft.fftshift(r)

        # ideal HP mask
        f_c = float(cutoff)
        H = (r >= f_c).astype(np.float32)

        E_total = mag2.sum(dtype=np.float64)
        E_high  = (mag2 * (H**2)).sum(dtype=np.float64)
        frac = E_high / (E_total + 1e-12)

        # filter + inverse
        F_filt = F * H
        arr_filt = np.real(np.fft.ifft2(np.fft.ifftshift(F_filt))).astype(np.float32, copy=False)

        # debug magnitude (log)
        mag_log = np.log1p(np.abs(F)).astype(np.float32, copy=False)
        mag_filt_log = np.log1p(np.abs(F_filt)).astype(np.float32, copy=False)

        nyq = min(0.5 / sx, 0.5 / sy)
        print("zmid=", zmid, "spacing(x,y)=", (sx, sy), "nyquist(cyc/mm)=", nyq, "cutoff(cyc/mm)=", f_c)

        return frac, E_total, E_high, img_sq, arr, arr_filt, mag_log, mag_filt_log

    # sitk image, cutoff for ideal HP filter, 
    @staticmethod
    def frequencyFilter3D(image, cutoff=0.4):

        img = sitk.Cast(image, sitk.sitkFloat32)

        # crop to even-shaped cube for FFT
        Z, Y, X = sitk.GetArrayViewFromImage(img).shape  # arr order (z,y,x)
        m = min(Z, Y, X)
        if m % 2 != 0:
            m -= 1

        z0 = (Z - m) // 2
        y0 = (Y - m) // 2
        x0 = (X - m) // 2

        img_cube = sitk.RegionOfInterest(img, size=[m, m, m], index=[x0, y0, z0])

        arr = sitk.GetArrayFromImage(img_cube).astype(np.float32, copy=False)

        # FFT
        F = np.fft.fftshift(np.fft.fftn(arr))
        mag2 = (np.abs(F) ** 2).astype(np.float64, copy=False)

        sx, sy, sz = img_cube.GetSpacing()

        # build freq axes in cycles/mm
        fz = np.fft.fftfreq(m, d=sz)[:, None, None]
        fy = np.fft.fftfreq(m, d=sy)[None, :, None]
        fx = np.fft.fftfreq(m, d=sx)[None, None, :]

        r = np.sqrt(fz**2 + fy**2 + fx**2)
        r = np.fft.fftshift(r)

        # ideal HP filter (cutoff is in cycles/mm)
        f_c = float(cutoff)
        H = (r >= f_c).astype(np.float32)

        E_total = mag2.sum(dtype=np.float64)
        E_high  = (mag2 * (H**2)).sum(dtype=np.float64)
        frac = E_high / (E_total + 1e-12)

        F_filt = F * H
        arr_filt = np.real(np.fft.ifftn(np.fft.ifftshift(F_filt))).astype(np.float32, copy=False)

        mag_log = np.log1p(np.abs(F)).astype(np.float32, copy=False)
        mag_filt_log = np.log1p(np.abs(F_filt)).astype(np.float32, copy=False)

        nyq = min(0.5/sx, 0.5/sy, 0.5/sz)
        print("spacing=", (sx,sy,sz), "nyquist(cyc/mm)=", nyq, "cutoff(cyc/mm)=", f_c)

        return frac, E_total, E_high, img_cube, arr, arr_filt, mag_log, mag_filt_log
