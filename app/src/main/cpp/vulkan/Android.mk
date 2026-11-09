LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)
LOCAL_MODULE := vulkan_probe
LOCAL_SRC_FILES := vulkan_probe.cpp
LOCAL_CPPFLAGS := -std=c++17 -Wall -Wextra -Wno-unused-parameter
LOCAL_LDLIBS := -llog -landroid -lvulkan
include $(BUILD_SHARED_LIBRARY)

