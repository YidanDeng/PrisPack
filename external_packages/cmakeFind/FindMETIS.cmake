# Try to locate METIS
# Once done, this will define
#
#  METIS_FOUND - system has METIS
#  METIS_INCLUDE_DIRS - the METIS include directories
#  METIS_LIBRARIES - link these to use METIS

find_path(METIS_INCLUDE_DIR
        NAMES metis.h
        HINTS ${CMAKE_SOURCE_DIR}/external_packages/Metis/include
)

find_library(METIS_LIBRARY
        NAMES metis
        PATHS ${CMAKE_SOURCE_DIR}/external_packages/Metis/lib
        /usr/lib
        /usr/local/lib
        /opt/local/lib
)

find_library(GKLIB_LIBRARY
        NAMES GKlib
        PATHS
        ${CMAKE_SOURCE_DIR}/external_packages/Metis/lib
        /usr/lib
        /usr/local/lib
        /opt/local/lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(METIS DEFAULT_MSG METIS_LIBRARY METIS_INCLUDE_DIR)

if (METIS_FOUND)
    message("Metis found.")
    set(METIS_LIBRARIES ${METIS_LIBRARY})
    set(METIS_INCLUDE_DIRS ${METIS_INCLUDE_DIR})

    message(STATUS "METIS include: ${METIS_INCLUDE_DIRS}")
    message(STATUS "METIS lib: ${METIS_LIBRARIES}")
    message(STATUS "GK lib: ${GKLIB_LIBRARY}")
else ()
    message(WARNING "Could not find METIS")
endif ()

