cmake_minimum_required(VERSION 3.22)
project(agx_camera_native C)

find_library(log-lib log)

set(LIBJPEG_ROOT ${CMAKE_SOURCE_DIR}/libjpeg-turbo)
set(LIBJPEG_INCLUDE_DIR ${LIBJPEG_ROOT})
set(LIBJPEG_LIB_DIR ${LIBJPEG_ROOT}/${ANDROID_ABI})

add_library(libjpeg SHARED IMPORTED)
set_target_properties(libjpeg PROPERTIES
    IMPORTED_LOCATION ${LIBJPEG_LIB_DIR}/libjpeg.so
    INTERFACE_INCLUDE_DIRECTORIES ${LIBJPEG_INCLUDE_DIR}
)

add_library(jpeg_encoder SHARED jpeg_encoder.c)
target_include_directories(jpeg_encoder PRIVATE ${LIBJPEG_INCLUDE_DIR})
target_link_libraries(jpeg_encoder ${log-lib} libjpeg)
