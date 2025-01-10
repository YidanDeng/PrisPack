# - Try to find OpenBLAS
# Once done, this will define
#
#  OPENBLAS_FOUND - system has OpenBLAS
#  OPENBLAS_INCLUDE_DIR - the OpenBLAS include directory
#  OPENBLAS_LIBRARIES - link these to use OpenBLAS
#  OPENBLAS_DEFINITIONS - compiler switches required for using OpenBLAS

if (OpenBLAS_INCLUDE_DIR)
    # Already in cache, be silent
    set (OpenBLAS_FIND_QUIETLY TRUE)
endif (OpenBLAS_INCLUDE_DIR)

find_path(OpenBLAS_INCLUDE_DIR
        NAMES cblas.h
        HINTS ${CMAKE_SOURCE_DIR}/external_packages/OpenBLAS/include
)

find_library(OpenBLAS_LIBRARIES
        NAMES openblas
        HINTS ${CMAKE_SOURCE_DIR}/external_packages/OpenBLAS/lib
)

# handle the QUIETLY and REQUIRED arguments and set OPENBLAS_FOUND to TRUE if
# all listed variables are TRUE
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OpenBLAS DEFAULT_MSG
        OpenBLAS_INCLUDE_DIR OpenBLAS_LIBRARIES)

if (OpenBLAS_FOUND)
    message("OpenBLAS found.")
    set (OpenBLAS_LIBRARIES ${OpenBLAS_LIBRARIES})
    message(STATUS "OpenBLAS library path: ${OpenBLAS_LIBRARIES}")
    set (OpenBLAS_INCLUDE_DIRS ${OpenBLAS_INCLUDE_DIR})
    message(STATUS "OpenBLAS include path: ${OpenBLAS_INCLUDE_DIR}")
else (OpenBLAS_FOUND)
    if (OpenBLAS_FIND_REQUIRED)
        message (FATAL_ERROR "Could not find OpenBLAS")
    endif (OpenBLAS_FIND_REQUIRED)
endif (OpenBLAS_FOUND)

mark_as_advanced(OpenBLAS_INCLUDE_DIR OpenBLAS_LIBRARIES)
