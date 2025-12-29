cmake_minimum_required(VERSION 3.20.6...3.22.6 FATAL_ERROR)

project(AirwayEnergyFraction)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://www.slicer.org/wiki/Documentation/Nightly/Extensions/AirwayEnergyFraction")
set(EXTENSION_CONTRIBUTORS "Jake Kitzmann (The University of Iowa)")
set(EXTENSION_DESCRIPTION "Build high energy frequency fraction of 3D and 2D areas from patient airways")
set(EXTENSION_ICONURL "https://www.example.com/Slicer/Extensions/AirwayEnergyFraction.png")
set(EXTENSION_SCREENSHOTURLS "https://www.example.com/Slicer/Extensions/AirwayEnergyFraction/Screenshots/1.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(AirwayEnergyFraction)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
