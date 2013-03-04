PROJECT(OPENCV_HOTSHOTS)

cmake_minimum_required(VERSION 2.8)
find_package(OpenCV REQUIRED)
find_package(Boost COMPONENTS filesystem system REQUIRED)
find_package(GLUT)
find_package(OpenGL)

INCLUDE_DIRECTORIES(
${OPENCV_INCLUDE_DIR}
${CMAKE_CURRENT_SOURCE_DIR}
${Boost_INCLUDE_DIRS}
${GLUT_INCLUDE_DIR}
)
SET( LIBRARY_OUTPUT_PATH ${OPENCV_HOTSHOTS_BINARY_DIR}/../lib CACHE PATH
     "Output directory for the OpenCV_hotshots library" )
SET( EXECUTABLE_OUTPUT_PATH 
     ${OPENCV_HOTSHOTS_BINARY_DIR}/../bin CACHE PATH
     "Output directory for the OpenCV_hotshots executables" )

SUBDIRS(opencv_hotshots/ft)
