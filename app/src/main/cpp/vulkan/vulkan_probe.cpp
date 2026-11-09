#include <jni.h>

#include <android/hardware_buffer.h>
#include <android/log.h>
#include <android/native_window_jni.h>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_android.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include "demosaic_spv.h"
#include "s2_tiled_spv.h"
#include "agx_spv.h"
#include "agx_encoder_spv.h"

#define LOG_TAG "VulkanProbe"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace {

struct State {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t queueFamily = 0;
    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    VkQueryPool queryPool = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    VkSampler sampler = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    VkShaderModule shader = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorSetLayout s2DescriptorSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout s2PipelineLayout = VK_NULL_HANDLE;
    VkDescriptorPool s2DescriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet s2DescriptorSet = VK_NULL_HANDLE;
    VkShaderModule s2Shader = VK_NULL_HANDLE;
    VkPipeline s2Pipeline = VK_NULL_HANDLE;
    VkDescriptorSetLayout agxDescriptorSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout agxPipelineLayout = VK_NULL_HANDLE;
    VkDescriptorPool agxDescriptorPool = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> agxSets;
    VkShaderModule agxShader = VK_NULL_HANDLE;
    VkPipeline agxPipeline = VK_NULL_HANDLE;
    VkImage inputImage = VK_NULL_HANDLE;
    VkDeviceMemory inputMemory = VK_NULL_HANDLE;
    VkImageView inputView = VK_NULL_HANDLE;
    VkImage outputImage = VK_NULL_HANDLE;
    VkDeviceMemory outputMemory = VK_NULL_HANDLE;
    VkImageView outputView = VK_NULL_HANDLE;
    VkImage outputFlagImage = VK_NULL_HANDLE;
    VkDeviceMemory outputFlagMemory = VK_NULL_HANDLE;
    VkImageView outputFlagView = VK_NULL_HANDLE;
    VkImage outputSigmaImage = VK_NULL_HANDLE;
    VkDeviceMemory outputSigmaMemory = VK_NULL_HANDLE;
    VkImageView outputSigmaView = VK_NULL_HANDLE;
    VkImage outputAgxImage = VK_NULL_HANDLE;
    VkDeviceMemory outputAgxMemory = VK_NULL_HANDLE;
    VkImageView outputAgxView = VK_NULL_HANDLE;
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    std::vector<VkImage> swapImages;
    std::vector<VkImageView> swapViews;
    VkSemaphore imageAvailable = VK_NULL_HANDLE;
    VkSemaphore renderFinished = VK_NULL_HANDLE;
    uint32_t swapWidth = 0;
    uint32_t swapHeight = 0;
    bool useSwapchain = false;
    AHardwareBuffer* ahb = nullptr;
    VkFormat inputFormat = VK_FORMAT_UNDEFINED;
    VkFormat outputFormat = VK_FORMAT_UNDEFINED;
    VkFormat outputFlagFormat = VK_FORMAT_UNDEFINED;
    VkFormat outputSigmaFormat = VK_FORMAT_UNDEFINED;
    VkFormat outputAgxFormat = VK_FORMAT_UNDEFINED;
    uint64_t externalFormat = 0;
    uint32_t timestampValidBits = 0;
    float timestampPeriod = 0.0f;
};

bool hasExtension(const std::vector<VkExtensionProperties>& props, const char* name) {
    for (const auto& p : props) {
        if (std::strcmp(p.extensionName, name) == 0) return true;
    }
    return false;
}

uint32_t findMemoryType(const VkPhysicalDeviceMemoryProperties& memProps,
                        uint32_t typeBits,
                        VkMemoryPropertyFlags required) {
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
        if ((typeBits & (1u << i)) != 0 &&
            (memProps.memoryTypes[i].propertyFlags & required) == required) {
            return i;
        }
    }
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
        if ((typeBits & (1u << i)) != 0) return i;
    }
    return UINT32_MAX;
}

bool createInstance(State& s, std::vector<std::string>& log, bool useSurface = false) {
    std::vector<VkExtensionProperties> available;
    uint32_t count = 0;
    if (vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr) != VK_SUCCESS) {
        log.push_back("vkEnumerateInstanceExtensionProperties failed");
        return false;
    }
    available.resize(count);
    vkEnumerateInstanceExtensionProperties(nullptr, &count, available.data());

    std::vector<const char*> enabled;
    if (hasExtension(available, VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME)) {
        enabled.push_back(VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME);
    }
    if (useSurface) {
        if (!hasExtension(available, VK_KHR_SURFACE_EXTENSION_NAME) ||
            !hasExtension(available, VK_KHR_ANDROID_SURFACE_EXTENSION_NAME)) {
            log.push_back("surface instance extensions missing");
            return false;
        }
        enabled.push_back(VK_KHR_SURFACE_EXTENSION_NAME);
        enabled.push_back(VK_KHR_ANDROID_SURFACE_EXTENSION_NAME);
    }

    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "AgXCam Vulkan probe";
    app.applicationVersion = 1;
    app.pEngineName = "vulkan-probe";
    app.engineVersion = 1;
    app.apiVersion = VK_API_VERSION_1_1;

    VkInstanceCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ci.pApplicationInfo = &app;
    ci.enabledExtensionCount = static_cast<uint32_t>(enabled.size());
    ci.ppEnabledExtensionNames = enabled.data();

    VkResult r = vkCreateInstance(&ci, nullptr, &s.instance);
    if (r != VK_SUCCESS) {
        log.push_back("vkCreateInstance failed: " + std::to_string(r));
        return false;
    }
    return true;
}

bool createDevice(State& s, std::vector<std::string>& log, bool useSurface = false) {
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(s.instance, &count, nullptr);
    if (count == 0) {
        log.push_back("no Vulkan physical devices");
        return false;
    }
    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(s.instance, &count, devices.data());

    for (VkPhysicalDevice dev : devices) {
        uint32_t extCount = 0;
        vkEnumerateDeviceExtensionProperties(dev, nullptr, &extCount, nullptr);
        std::vector<VkExtensionProperties> exts(extCount);
        vkEnumerateDeviceExtensionProperties(dev, nullptr, &extCount, exts.data());
        if (!hasExtension(exts, VK_ANDROID_EXTERNAL_MEMORY_ANDROID_HARDWARE_BUFFER_EXTENSION_NAME)) {
            continue;
        }
        if (useSurface && !hasExtension(exts, VK_KHR_SWAPCHAIN_EXTENSION_NAME)) {
            continue;
        }

        uint32_t qCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &qCount, nullptr);
        std::vector<VkQueueFamilyProperties> queues(qCount);
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &qCount, queues.data());
        uint32_t qf = UINT32_MAX;
        for (uint32_t i = 0; i < qCount; ++i) {
            if ((queues[i].queueFlags & VK_QUEUE_COMPUTE_BIT) == 0) continue;
            if (useSurface) {
                VkBool32 present = VK_FALSE;
                vkGetPhysicalDeviceSurfaceSupportKHR(dev, i, s.surface, &present);
                if (!present) continue;
            }
            qf = i;
            break;
        }
        if (qf == UINT32_MAX) continue;

        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(dev, &props);

        std::vector<const char*> enabled = {
            VK_ANDROID_EXTERNAL_MEMORY_ANDROID_HARDWARE_BUFFER_EXTENSION_NAME
        };
        if (hasExtension(exts, VK_EXT_QUEUE_FAMILY_FOREIGN_EXTENSION_NAME)) {
            enabled.push_back(VK_EXT_QUEUE_FAMILY_FOREIGN_EXTENSION_NAME);
        }
        if (useSurface) {
            enabled.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
        }

        float priority = 1.0f;
        VkDeviceQueueCreateInfo qci{};
        qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qci.queueFamilyIndex = qf;
        qci.queueCount = 1;
        qci.pQueuePriorities = &priority;

        VkDeviceCreateInfo dci{};
        dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos = &qci;
        dci.enabledExtensionCount = static_cast<uint32_t>(enabled.size());
        dci.ppEnabledExtensionNames = enabled.data();

        VkResult r = vkCreateDevice(dev, &dci, nullptr, &s.device);
        if (r != VK_SUCCESS) {
            log.push_back("vkCreateDevice failed: " + std::to_string(r));
            continue;
        }

        s.physical = dev;
        s.queueFamily = qf;
        s.timestampValidBits = queues[qf].timestampValidBits;
        s.timestampPeriod = props.limits.timestampPeriod;
        vkGetDeviceQueue(s.device, qf, 0, &s.queue);
        std::ostringstream devLine;
        devLine << "device=" << props.deviceName
                << " api=" << VK_API_VERSION_MAJOR(props.apiVersion) << "."
                << VK_API_VERSION_MINOR(props.apiVersion)
                << " queueFamily=" << qf
                << " timestampBits=" << s.timestampValidBits
                << " timestampPeriod=" << s.timestampPeriod;
        log.push_back(devLine.str());
        return true;
    }

    log.push_back("no Vulkan device with AHB import support");
    return false;
}

bool allocateAhb(State& s, int width, int height, std::vector<std::string>& log) {
    AHardwareBuffer_Desc desc{};
    desc.width = static_cast<uint32_t>(width);
    desc.height = static_cast<uint32_t>(height);
    desc.layers = 1;
    desc.format = AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM;
    desc.usage = AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN |
                 AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE;
    int r = AHardwareBuffer_allocate(&desc, &s.ahb);
    if (r != 0 || s.ahb == nullptr) {
        log.push_back("AHardwareBuffer_allocate failed: " + std::to_string(r));
        return false;
    }

    void* data = nullptr;
    int32_t bytesPerPixel = 0;
    int32_t bytesPerStride = 0;
    r = AHardwareBuffer_lockAndGetInfo(
        s.ahb, AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN, -1, nullptr,
        &data, &bytesPerPixel, &bytesPerStride);
    if (r == 0 && data != nullptr && bytesPerPixel == 4 &&
        bytesPerStride >= width * 4) {
        for (int y = 0; y < height; ++y) {
            auto* row = reinterpret_cast<uint32_t*>(
                static_cast<uint8_t*>(data) + static_cast<size_t>(y) * bytesPerStride);
            for (int x = 0; x < width; ++x) {
                uint32_t v = static_cast<uint32_t>((x * 17 + y * 31) & 0xFFFFu);
                row[x] = (v & 0xFFu) | ((v & 0xFF00u) << 8) | 0xFF000000u;
            }
        }
        AHardwareBuffer_unlock(s.ahb, nullptr);
        log.push_back("ahb=RGBA8 " + std::to_string(width) + "x" + std::to_string(height) +
                      " strideBytes=" + std::to_string(bytesPerStride) +
                      " bytesPerPixel=" + std::to_string(bytesPerPixel));
    } else {
        log.push_back("AHardwareBuffer_lockAndGetInfo failed: " + std::to_string(r));
    }
    return true;
}

bool createImportedImage(State& s, int width, int height, std::vector<std::string>& log) {
    auto getProps = reinterpret_cast<PFN_vkGetAndroidHardwareBufferPropertiesANDROID>(
        vkGetDeviceProcAddr(s.device, "vkGetAndroidHardwareBufferPropertiesANDROID"));
    if (getProps == nullptr) {
        log.push_back("vkGetAndroidHardwareBufferPropertiesANDROID missing");
        return false;
    }

    VkAndroidHardwareBufferFormatPropertiesANDROID fmt{};
    fmt.sType = VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_FORMAT_PROPERTIES_ANDROID;
    VkAndroidHardwareBufferPropertiesANDROID props{};
    props.sType = VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_PROPERTIES_ANDROID;
    props.pNext = &fmt;
    VkResult r = getProps(s.device, s.ahb, &props);
    if (r != VK_SUCCESS) {
        log.push_back("vkGetAndroidHardwareBufferPropertiesANDROID failed: " +
                      std::to_string(r));
        return false;
    }

    s.inputFormat = fmt.format;
    s.externalFormat = fmt.externalFormat;
    {
        std::ostringstream line;
        line << "ahbFormat=" << fmt.format
             << " externalFormat=" << fmt.externalFormat
             << " allocationSize=" << props.allocationSize
             << " memoryTypeBits=0x" << std::hex << props.memoryTypeBits;
        log.push_back(line.str());
    }
    if (s.inputFormat == VK_FORMAT_UNDEFINED) {
        log.push_back("AHB external format is undefined");
        return false;
    }

    VkExternalMemoryImageCreateInfo external{};
    external.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
    external.handleTypes =
        VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID;

    VkImageCreateInfo image{};
    image.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    image.pNext = &external;
    image.imageType = VK_IMAGE_TYPE_2D;
    image.format = s.inputFormat;
    image.extent = {static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1};
    image.mipLevels = 1;
    image.arrayLayers = 1;
    image.samples = VK_SAMPLE_COUNT_1_BIT;
    image.tiling = VK_IMAGE_TILING_OPTIMAL;
    image.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
    image.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    image.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    r = vkCreateImage(s.device, &image, nullptr, &s.inputImage);
    if (r != VK_SUCCESS) {
        log.push_back("vkCreateImage(input) failed: " + std::to_string(r));
        return false;
    }

    VkMemoryRequirements req{};
    vkGetImageMemoryRequirements(s.device, s.inputImage, &req);
    VkPhysicalDeviceMemoryProperties memProps{};
    vkGetPhysicalDeviceMemoryProperties(s.physical, &memProps);
    uint32_t memType = findMemoryType(memProps, props.memoryTypeBits,
                                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (memType == UINT32_MAX) {
        log.push_back("no memory type for imported AHB");
        return false;
    }

    VkImportAndroidHardwareBufferInfoANDROID importInfo{};
    importInfo.sType = VK_STRUCTURE_TYPE_IMPORT_ANDROID_HARDWARE_BUFFER_INFO_ANDROID;
    importInfo.buffer = s.ahb;

    VkMemoryDedicatedAllocateInfo dedicated{};
    dedicated.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
    dedicated.pNext = &importInfo;
    dedicated.image = s.inputImage;

    VkMemoryAllocateInfo alloc{};
    alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.pNext = &dedicated;
    alloc.allocationSize = req.size;
    alloc.memoryTypeIndex = memType;

    r = vkAllocateMemory(s.device, &alloc, nullptr, &s.inputMemory);
    if (r != VK_SUCCESS) {
        log.push_back("vkAllocateMemory(import) failed: " + std::to_string(r));
        return false;
    }
    r = vkBindImageMemory(s.device, s.inputImage, s.inputMemory, 0);
    if (r != VK_SUCCESS) {
        log.push_back("vkBindImageMemory(import) failed: " + std::to_string(r));
        return false;
    }

    VkImageViewCreateInfo view{};
    view.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view.image = s.inputImage;
    view.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view.format = s.inputFormat;
    view.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    r = vkCreateImageView(s.device, &view, nullptr, &s.inputView);
    if (r != VK_SUCCESS) {
        log.push_back("vkCreateImageView(input) failed: " + std::to_string(r));
        return false;
    }
    log.push_back("imported AHB as VkImage");
    return true;
}

bool createStorageImage(State& s, VkFormat format, int width, int height,
                        VkImage& imageOut, VkDeviceMemory& memoryOut,
                        VkImageView& viewOut, const char* label,
                        std::vector<std::string>& log) {
    VkImageCreateInfo image{};
    image.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    image.imageType = VK_IMAGE_TYPE_2D;
    image.format = format;
    image.extent = {static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1};
    image.mipLevels = 1;
    image.arrayLayers = 1;
    image.samples = VK_SAMPLE_COUNT_1_BIT;
    image.tiling = VK_IMAGE_TILING_OPTIMAL;
    image.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    image.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    image.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VkResult r = vkCreateImage(s.device, &image, nullptr, &imageOut);
    if (r != VK_SUCCESS) {
        log.push_back("vkCreateImage(" + std::string(label) + ") failed: " +
                      std::to_string(r));
        return false;
    }

    VkMemoryRequirements req{};
    vkGetImageMemoryRequirements(s.device, imageOut, &req);
    VkPhysicalDeviceMemoryProperties memProps{};
    vkGetPhysicalDeviceMemoryProperties(s.physical, &memProps);
    uint32_t memType = findMemoryType(memProps, req.memoryTypeBits,
                                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (memType == UINT32_MAX) {
        log.push_back("no memory type for " + std::string(label));
        return false;
    }

    VkMemoryAllocateInfo alloc{};
    alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.allocationSize = req.size;
    alloc.memoryTypeIndex = memType;
    r = vkAllocateMemory(s.device, &alloc, nullptr, &memoryOut);
    if (r != VK_SUCCESS) {
        log.push_back("vkAllocateMemory(" + std::string(label) + ") failed: " +
                      std::to_string(r));
        return false;
    }
    r = vkBindImageMemory(s.device, imageOut, memoryOut, 0);
    if (r != VK_SUCCESS) {
        log.push_back("vkBindImageMemory(" + std::string(label) + ") failed: " +
                      std::to_string(r));
        return false;
    }

    VkImageViewCreateInfo view{};
    view.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view.image = imageOut;
    view.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view.format = format;
    view.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    r = vkCreateImageView(s.device, &view, nullptr, &viewOut);
    if (r != VK_SUCCESS) {
        log.push_back("vkCreateImageView(" + std::string(label) + ") failed: " +
                      std::to_string(r));
        return false;
    }
    return true;
}

bool createSurface(State& s, ANativeWindow* window, std::vector<std::string>& log) {
    auto createAndroidSurface =
        reinterpret_cast<PFN_vkCreateAndroidSurfaceKHR>(
            vkGetInstanceProcAddr(s.instance, "vkCreateAndroidSurfaceKHR"));
    if (createAndroidSurface == nullptr) {
        log.push_back("vkCreateAndroidSurfaceKHR missing");
        return false;
    }
    VkAndroidSurfaceCreateInfoKHR info{};
    info.sType = VK_STRUCTURE_TYPE_ANDROID_SURFACE_CREATE_INFO_KHR;
    info.window = window;
    VkResult r = createAndroidSurface(s.instance, &info, nullptr, &s.surface);
    if (r != VK_SUCCESS) {
        log.push_back("vkCreateAndroidSurfaceKHR failed: " + std::to_string(r));
        return false;
    }
    return true;
}

bool createSwapchain(State& s, std::vector<std::string>& log) {
    VkSurfaceCapabilitiesKHR caps{};
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(s.physical, s.surface, &caps);
    uint32_t formatCount = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(s.physical, s.surface, &formatCount, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(formatCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(s.physical, s.surface, &formatCount, formats.data());
    VkSurfaceFormatKHR chosen = formats[0];
    for (const auto& f : formats) {
        if (f.format == VK_FORMAT_R8G8B8A8_UNORM) {
            chosen = f;
            break;
        }
    }

    VkSwapchainCreateInfoKHR info{};
    info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    info.surface = s.surface;
    info.minImageCount = caps.minImageCount + 1;
    info.imageFormat = chosen.format;
    info.imageColorSpace = chosen.colorSpace;
    info.imageExtent = caps.currentExtent;
    info.imageArrayLayers = 1;
    info.imageUsage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    info.preTransform = caps.currentTransform;
    info.compositeAlpha = (caps.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR)
        ? VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR : VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR;
    info.presentMode = VK_PRESENT_MODE_FIFO_KHR;
    info.clipped = VK_TRUE;
    VkResult r = vkCreateSwapchainKHR(s.device, &info, nullptr, &s.swapchain);
    if (r != VK_SUCCESS) {
        log.push_back("vkCreateSwapchainKHR(encoder) failed: " + std::to_string(r));
        return false;
    }

    uint32_t imageCount = 0;
    vkGetSwapchainImagesKHR(s.device, s.swapchain, &imageCount, nullptr);
    s.swapImages.resize(imageCount);
    vkGetSwapchainImagesKHR(s.device, s.swapchain, &imageCount, s.swapImages.data());
    s.swapViews.resize(imageCount);
    for (uint32_t i = 0; i < imageCount; ++i) {
        VkImageViewCreateInfo view{};
        view.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        view.image = s.swapImages[i];
        view.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view.format = chosen.format;
        view.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        r = vkCreateImageView(s.device, &view, nullptr, &s.swapViews[i]);
        if (r != VK_SUCCESS) {
            log.push_back("vkCreateImageView(swap) failed: " + std::to_string(r));
            return false;
        }
    }
    s.swapWidth = caps.currentExtent.width;
    s.swapHeight = caps.currentExtent.height;
    log.push_back("encoder swapchain images=" + std::to_string(imageCount) +
                  " format=" + std::to_string(chosen.format) +
                  " extent=" + std::to_string(s.swapWidth) + "x" +
                  std::to_string(s.swapHeight));
    return true;
}

bool createOutputImages(State& s, int width, int height,
                        std::vector<std::string>& log) {
    s.outputFormat = VK_FORMAT_R32G32B32A32_SFLOAT;
    s.outputFlagFormat = VK_FORMAT_R32_SFLOAT;
    s.outputSigmaFormat = VK_FORMAT_R32_SFLOAT;
    if (!createStorageImage(s, s.outputFormat, width, height,
                            s.outputImage, s.outputMemory, s.outputView,
                            "demosaic", log)) {
        return false;
    }
    if (!createStorageImage(s, s.outputFlagFormat, width, height,
                            s.outputFlagImage, s.outputFlagMemory, s.outputFlagView,
                            "demosaic flag", log)) {
        return false;
    }
    if (!createStorageImage(s, s.outputSigmaFormat, (width + 1) / 2, (height + 1) / 2,
                            s.outputSigmaImage, s.outputSigmaMemory, s.outputSigmaView,
                            "sigma", log)) {
        return false;
    }
    if (s.useSwapchain) {
        return true;
    }
    s.outputAgxFormat = VK_FORMAT_R8G8B8A8_UNORM;
    return createStorageImage(s, s.outputAgxFormat, width, height,
                              s.outputAgxImage, s.outputAgxMemory, s.outputAgxView,
                              "agx", log);
}

bool createPipeline(State& s, std::vector<std::string>& log) {
    VkSamplerCreateInfo sampler{};
    sampler.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sampler.magFilter = VK_FILTER_NEAREST;
    sampler.minFilter = VK_FILTER_NEAREST;
    sampler.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sampler.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler.maxLod = 0.0f;
    VkResult r = vkCreateSampler(s.device, &sampler, nullptr, &s.sampler);
    if (r != VK_SUCCESS) {
        log.push_back("vkCreateSampler failed: " + std::to_string(r));
        return false;
    }

    VkDescriptorSetLayoutBinding bindings[6]{};
    for (uint32_t i = 0; i < 3; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    for (uint32_t i = 3; i < 6; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo layout{};
    layout.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout.bindingCount = 6;
    layout.pBindings = bindings;
    r = vkCreateDescriptorSetLayout(s.device, &layout, nullptr, &s.descriptorSetLayout);
    if (r != VK_SUCCESS) {
        log.push_back("vkCreateDescriptorSetLayout failed: " + std::to_string(r));
        return false;
    }

    VkPipelineLayoutCreateInfo pipelineLayout{};
    pipelineLayout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayout.setLayoutCount = 1;
    pipelineLayout.pSetLayouts = &s.descriptorSetLayout;
    r = vkCreatePipelineLayout(s.device, &pipelineLayout, nullptr, &s.pipelineLayout);
    if (r != VK_SUCCESS) {
        log.push_back("vkCreatePipelineLayout failed: " + std::to_string(r));
        return false;
    }

    VkShaderModuleCreateInfo shaderInfo{};
    shaderInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shaderInfo.codeSize = demosaic_spv_count * sizeof(uint32_t);
    shaderInfo.pCode = demosaic_spv;
    r = vkCreateShaderModule(s.device, &shaderInfo, nullptr, &s.shader);
    if (r != VK_SUCCESS) {
        log.push_back("vkCreateShaderModule failed: " + std::to_string(r));
        return false;
    }

    VkPipelineShaderStageCreateInfo stage{};
    stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = s.shader;
    stage.pName = "main";

    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage = stage;
    pipelineInfo.layout = s.pipelineLayout;
    r = vkCreateComputePipelines(s.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &s.pipeline);
    if (r != VK_SUCCESS) {
        log.push_back("vkCreateComputePipelines failed: " + std::to_string(r));
        return false;
    }

    VkDescriptorPoolSize sizes[2]{};
    sizes[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    sizes[0].descriptorCount = 3;
    sizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    sizes[1].descriptorCount = 3;
    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = 2;
    poolInfo.pPoolSizes = sizes;
    r = vkCreateDescriptorPool(s.device, &poolInfo, nullptr, &s.descriptorPool);
    if (r != VK_SUCCESS) {
        log.push_back("vkCreateDescriptorPool failed: " + std::to_string(r));
        return false;
    }

    VkDescriptorSetAllocateInfo alloc{};
    alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc.descriptorPool = s.descriptorPool;
    alloc.descriptorSetCount = 1;
    alloc.pSetLayouts = &s.descriptorSetLayout;
    r = vkAllocateDescriptorSets(s.device, &alloc, &s.descriptorSet);
    if (r != VK_SUCCESS) {
        log.push_back("vkAllocateDescriptorSets failed: " + std::to_string(r));
        return false;
    }

    VkDescriptorImageInfo inputInfo{};
    inputInfo.sampler = s.sampler;
    inputInfo.imageView = s.inputView;
    inputInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkDescriptorImageInfo outputInfo{};
    outputInfo.imageView = s.outputView;
    outputInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    VkDescriptorImageInfo flagInfo{};
    flagInfo.imageView = s.outputFlagView;
    flagInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    VkDescriptorImageInfo sigmaInfo{};
    sigmaInfo.imageView = s.outputSigmaView;
    sigmaInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkWriteDescriptorSet writes[6]{};
    for (uint32_t i = 0; i < 3; ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = s.descriptorSet;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[i].pImageInfo = &inputInfo;
    }
    writes[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[3].dstSet = s.descriptorSet;
    writes[3].dstBinding = 3;
    writes[3].descriptorCount = 1;
    writes[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[3].pImageInfo = &outputInfo;
    writes[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[4].dstSet = s.descriptorSet;
    writes[4].dstBinding = 4;
    writes[4].descriptorCount = 1;
    writes[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[4].pImageInfo = &flagInfo;
    writes[5].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[5].dstSet = s.descriptorSet;
    writes[5].dstBinding = 5;
    writes[5].descriptorCount = 1;
    writes[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[5].pImageInfo = &sigmaInfo;
    vkUpdateDescriptorSets(s.device, 6, writes, 0, nullptr);
    return true;
}

bool createS2Pipeline(State& s, std::vector<std::string>& log) {
    VkDescriptorSetLayoutBinding bindings[2]{};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo layout{};
    layout.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout.bindingCount = 2;
    layout.pBindings = bindings;
    VkResult r = vkCreateDescriptorSetLayout(
        s.device, &layout, nullptr, &s.s2DescriptorSetLayout);
    if (r != VK_SUCCESS) {
        log.push_back("vkCreateDescriptorSetLayout(S2) failed: " + std::to_string(r));
        return false;
    }

    VkPipelineLayoutCreateInfo pipelineLayout{};
    pipelineLayout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayout.setLayoutCount = 1;
    pipelineLayout.pSetLayouts = &s.s2DescriptorSetLayout;
    r = vkCreatePipelineLayout(s.device, &pipelineLayout, nullptr, &s.s2PipelineLayout);
    if (r != VK_SUCCESS) {
        log.push_back("vkCreatePipelineLayout(S2) failed: " + std::to_string(r));
        return false;
    }

    VkShaderModuleCreateInfo shaderInfo{};
    shaderInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shaderInfo.codeSize = s2_tiled_spv_count * sizeof(uint32_t);
    shaderInfo.pCode = s2_tiled_spv;
    r = vkCreateShaderModule(s.device, &shaderInfo, nullptr, &s.s2Shader);
    if (r != VK_SUCCESS) {
        log.push_back("vkCreateShaderModule(S2) failed: " + std::to_string(r));
        return false;
    }

    VkPipelineShaderStageCreateInfo stage{};
    stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = s.s2Shader;
    stage.pName = "main";

    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage = stage;
    pipelineInfo.layout = s.s2PipelineLayout;
    r = vkCreateComputePipelines(
        s.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &s.s2Pipeline);
    if (r != VK_SUCCESS) {
        log.push_back("vkCreateComputePipelines(S2) failed: " + std::to_string(r));
        return false;
    }

    VkDescriptorPoolSize sizes[2]{};
    sizes[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    sizes[0].descriptorCount = 1;
    sizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    sizes[1].descriptorCount = 1;
    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = 2;
    poolInfo.pPoolSizes = sizes;
    r = vkCreateDescriptorPool(s.device, &poolInfo, nullptr, &s.s2DescriptorPool);
    if (r != VK_SUCCESS) {
        log.push_back("vkCreateDescriptorPool(S2) failed: " + std::to_string(r));
        return false;
    }

    VkDescriptorSetAllocateInfo alloc{};
    alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc.descriptorPool = s.s2DescriptorPool;
    alloc.descriptorSetCount = 1;
    alloc.pSetLayouts = &s.s2DescriptorSetLayout;
    r = vkAllocateDescriptorSets(s.device, &alloc, &s.s2DescriptorSet);
    if (r != VK_SUCCESS) {
        log.push_back("vkAllocateDescriptorSets(S2) failed: " + std::to_string(r));
        return false;
    }

    VkDescriptorImageInfo inputInfo{};
    inputInfo.sampler = s.sampler;
    inputInfo.imageView = s.inputView;
    inputInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkDescriptorImageInfo sigmaInfo{};
    sigmaInfo.imageView = s.outputSigmaView;
    sigmaInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkWriteDescriptorSet writes[2]{};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = s.s2DescriptorSet;
    writes[0].dstBinding = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[0].pImageInfo = &inputInfo;
    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = s.s2DescriptorSet;
    writes[1].dstBinding = 1;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[1].pImageInfo = &sigmaInfo;
    vkUpdateDescriptorSets(s.device, 2, writes, 0, nullptr);
    return true;
}

bool createAgxPipeline(State& s, std::vector<std::string>& log) {
    VkDescriptorSetLayoutBinding bindings[2]{};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo layout{};
    layout.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout.bindingCount = 2;
    layout.pBindings = bindings;
    VkResult r = vkCreateDescriptorSetLayout(
        s.device, &layout, nullptr, &s.agxDescriptorSetLayout);
    if (r != VK_SUCCESS) {
        log.push_back("vkCreateDescriptorSetLayout(AgX) failed: " + std::to_string(r));
        return false;
    }

    VkPipelineLayoutCreateInfo pipelineLayout{};
    pipelineLayout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayout.setLayoutCount = 1;
    pipelineLayout.pSetLayouts = &s.agxDescriptorSetLayout;
    r = vkCreatePipelineLayout(s.device, &pipelineLayout, nullptr, &s.agxPipelineLayout);
    if (r != VK_SUCCESS) {
        log.push_back("vkCreatePipelineLayout(AgX) failed: " + std::to_string(r));
        return false;
    }

    VkShaderModuleCreateInfo shaderInfo{};
    shaderInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    if (s.useSwapchain) {
        shaderInfo.codeSize = agx_encoder_spv_count * sizeof(uint32_t);
        shaderInfo.pCode = agx_encoder_spv;
    } else {
        shaderInfo.codeSize = agx_spv_count * sizeof(uint32_t);
        shaderInfo.pCode = agx_spv;
    }
    r = vkCreateShaderModule(s.device, &shaderInfo, nullptr, &s.agxShader);
    if (r != VK_SUCCESS) {
        log.push_back("vkCreateShaderModule(AgX) failed: " + std::to_string(r));
        return false;
    }

    VkPipelineShaderStageCreateInfo stage{};
    stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = s.agxShader;
    stage.pName = "main";

    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage = stage;
    pipelineInfo.layout = s.agxPipelineLayout;
    r = vkCreateComputePipelines(
        s.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &s.agxPipeline);
    if (r != VK_SUCCESS) {
        log.push_back("vkCreateComputePipelines(AgX) failed: " + std::to_string(r));
        return false;
    }

    const uint32_t setCount = s.useSwapchain
        ? static_cast<uint32_t>(s.swapViews.size()) : 1u;
    VkDescriptorPoolSize sizes[2]{};
    sizes[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    sizes[0].descriptorCount = setCount;
    sizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    sizes[1].descriptorCount = setCount;
    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = setCount;
    poolInfo.poolSizeCount = 2;
    poolInfo.pPoolSizes = sizes;
    r = vkCreateDescriptorPool(s.device, &poolInfo, nullptr, &s.agxDescriptorPool);
    if (r != VK_SUCCESS) {
        log.push_back("vkCreateDescriptorPool(AgX) failed: " + std::to_string(r));
        return false;
    }

    std::vector<VkDescriptorSetLayout> layouts(setCount, s.agxDescriptorSetLayout);
    VkDescriptorSetAllocateInfo alloc{};
    alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc.descriptorPool = s.agxDescriptorPool;
    alloc.descriptorSetCount = setCount;
    alloc.pSetLayouts = layouts.data();
    s.agxSets.resize(setCount);
    r = vkAllocateDescriptorSets(s.device, &alloc, s.agxSets.data());
    if (r != VK_SUCCESS) {
        log.push_back("vkAllocateDescriptorSets(AgX) failed: " + std::to_string(r));
        return false;
    }

    for (uint32_t i = 0; i < setCount; ++i) {
        VkDescriptorImageInfo inputInfo{};
        inputInfo.sampler = s.sampler;
        inputInfo.imageView = s.outputView;
        inputInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkDescriptorImageInfo outputInfo{};
        outputInfo.imageView = s.useSwapchain ? s.swapViews[i] : s.outputAgxView;
        outputInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkWriteDescriptorSet writes[2]{};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = s.agxSets[i];
        writes[0].dstBinding = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[0].pImageInfo = &inputInfo;
        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet = s.agxSets[i];
        writes[1].dstBinding = 1;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[1].pImageInfo = &outputInfo;
        vkUpdateDescriptorSets(s.device, 2, writes, 0, nullptr);
    }
    return true;
}

bool createCommands(State& s, int width, int height, std::vector<std::string>& log) {
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = s.queueFamily;
    VkResult r = vkCreateCommandPool(s.device, &poolInfo, nullptr, &s.commandPool);
    if (r != VK_SUCCESS) {
        log.push_back("vkCreateCommandPool failed: " + std::to_string(r));
        return false;
    }

    VkCommandBufferAllocateInfo alloc{};
    alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc.commandPool = s.commandPool;
    alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc.commandBufferCount = 1;
    r = vkAllocateCommandBuffers(s.device, &alloc, &s.commandBuffer);
    if (r != VK_SUCCESS) {
        log.push_back("vkAllocateCommandBuffers failed: " + std::to_string(r));
        return false;
    }

    VkQueryPoolCreateInfo queryInfo{};
    queryInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    queryInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
    queryInfo.queryCount = 4;
    r = vkCreateQueryPool(s.device, &queryInfo, nullptr, &s.queryPool);
    if (r != VK_SUCCESS) {
        log.push_back("vkCreateQueryPool failed: " + std::to_string(r));
        return false;
    }

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    r = vkCreateFence(s.device, &fenceInfo, nullptr, &s.fence);
    if (r != VK_SUCCESS) {
        log.push_back("vkCreateFence failed: " + std::to_string(r));
        return false;
    }
    if (s.useSwapchain) {
        VkSemaphoreCreateInfo semInfo{};
        semInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        vkCreateSemaphore(s.device, &semInfo, nullptr, &s.imageAvailable);
        vkCreateSemaphore(s.device, &semInfo, nullptr, &s.renderFinished);
    }

    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    r = vkBeginCommandBuffer(s.commandBuffer, &begin);
    if (r != VK_SUCCESS) {
        log.push_back("vkBeginCommandBuffer failed: " + std::to_string(r));
        return false;
    }

    VkImageMemoryBarrier barriers[5]{};
    barriers[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barriers[0].srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
    barriers[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    barriers[0].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barriers[0].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barriers[0].image = s.inputImage;
    barriers[0].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    barriers[1].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barriers[1].srcAccessMask = 0;
    barriers[1].dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barriers[1].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barriers[1].newLayout = VK_IMAGE_LAYOUT_GENERAL;
    barriers[1].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barriers[1].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barriers[1].image = s.outputImage;
    barriers[1].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    barriers[2].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barriers[2].srcAccessMask = 0;
    barriers[2].dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barriers[2].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barriers[2].newLayout = VK_IMAGE_LAYOUT_GENERAL;
    barriers[2].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barriers[2].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barriers[2].image = s.outputFlagImage;
    barriers[2].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    barriers[3].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barriers[3].srcAccessMask = 0;
    barriers[3].dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barriers[3].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barriers[3].newLayout = VK_IMAGE_LAYOUT_GENERAL;
    barriers[3].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barriers[3].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barriers[3].image = s.outputSigmaImage;
    barriers[3].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    barriers[4].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barriers[4].srcAccessMask = 0;
    barriers[4].dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barriers[4].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barriers[4].newLayout = VK_IMAGE_LAYOUT_GENERAL;
    barriers[4].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barriers[4].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barriers[4].image = s.outputAgxImage;
    barriers[4].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    const uint32_t barrierCount = s.useSwapchain ? 4u : 5u;
    vkCmdPipelineBarrier(
        s.commandBuffer,
        VK_PIPELINE_STAGE_HOST_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 0, nullptr, 0, nullptr, barrierCount, barriers);
    vkEndCommandBuffer(s.commandBuffer);

    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &s.commandBuffer;
    r = vkQueueSubmit(s.queue, 1, &submit, s.fence);
    if (r != VK_SUCCESS) {
        log.push_back("vkQueueSubmit(setup) failed: " + std::to_string(r));
        return false;
    }
    vkWaitForFences(s.device, 1, &s.fence, VK_TRUE, UINT64_MAX);
    vkResetFences(s.device, 1, &s.fence);

    if (s.useSwapchain) {
        return true;
    }

    begin.flags = 0;
    r = vkBeginCommandBuffer(s.commandBuffer, &begin);
    if (r != VK_SUCCESS) {
        log.push_back("vkBeginCommandBuffer(dispatch) failed: " + std::to_string(r));
        return false;
    }
    vkCmdBindPipeline(s.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, s.pipeline);
    vkCmdBindDescriptorSets(
        s.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, s.pipelineLayout,
        0, 1, &s.descriptorSet, 0, nullptr);
    vkCmdResetQueryPool(s.commandBuffer, s.queryPool, 0, 4);
    if (s.timestampValidBits > 0) {
        vkCmdWriteTimestamp(s.commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, s.queryPool, 0);
    }
    vkCmdDispatch(s.commandBuffer, (width + 7) / 8, (height + 7) / 8, 1);
    if (s.timestampValidBits > 0) {
        vkCmdWriteTimestamp(s.commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, s.queryPool, 1);
    }
    vkCmdBindPipeline(s.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, s.s2Pipeline);
    vkCmdBindDescriptorSets(
        s.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, s.s2PipelineLayout,
        0, 1, &s.s2DescriptorSet, 0, nullptr);
    vkCmdDispatch(
        s.commandBuffer,
        (((width + 1) / 2) + 15) / 16,
        (((height + 1) / 2) + 15) / 16,
        1);
    if (s.timestampValidBits > 0) {
        vkCmdWriteTimestamp(s.commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, s.queryPool, 2);
    }

    VkImageMemoryBarrier agxBarrier{};
    agxBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    agxBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    agxBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    agxBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    agxBarrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    agxBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    agxBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    agxBarrier.image = s.outputImage;
    agxBarrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(
        s.commandBuffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &agxBarrier);

    vkCmdBindPipeline(s.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, s.agxPipeline);
    vkCmdBindDescriptorSets(
        s.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, s.agxPipelineLayout,
        0, 1, &s.agxSets[0], 0, nullptr);
    vkCmdDispatch(s.commandBuffer, (width + 7) / 8, (height + 7) / 8, 1);
    if (s.timestampValidBits > 0) {
        vkCmdWriteTimestamp(s.commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, s.queryPool, 3);
    }
    vkEndCommandBuffer(s.commandBuffer);
    return true;
}

bool runSwapBenchmark(State& s, int width, int height, int frames,
                      std::vector<std::string>& log) {
    const int warmup = 3;
    std::vector<double> cpuMs;
    std::vector<double> demosaicGpuMs;
    std::vector<double> s2GpuMs;
    std::vector<double> agxGpuMs;
    std::vector<double> totalGpuMs;
    bool firstFrame = true;

    for (int i = 0; i < warmup + frames; ++i) {
        auto t0 = std::chrono::steady_clock::now();
        uint32_t imageIndex = 0;
        VkResult r = vkAcquireNextImageKHR(
            s.device, s.swapchain, UINT64_MAX, s.imageAvailable,
            VK_NULL_HANDLE, &imageIndex);
        if (r != VK_SUCCESS && r != VK_SUBOPTIMAL_KHR) {
            log.push_back("swapchain acquire failed: " + std::to_string(r));
            return false;
        }

        vkResetCommandBuffer(s.commandBuffer, 0);
        VkCommandBufferBeginInfo begin{};
        begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        vkBeginCommandBuffer(s.commandBuffer, &begin);

        VkImageMemoryBarrier barriers[2]{};
        barriers[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barriers[0].srcAccessMask = firstFrame ? 0 : VK_ACCESS_SHADER_READ_BIT;
        barriers[0].dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barriers[0].oldLayout = firstFrame
            ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barriers[0].newLayout = VK_IMAGE_LAYOUT_GENERAL;
        barriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[0].image = s.outputImage;
        barriers[0].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

        barriers[1].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barriers[1].srcAccessMask = 0;
        barriers[1].dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barriers[1].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barriers[1].newLayout = VK_IMAGE_LAYOUT_GENERAL;
        barriers[1].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[1].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[1].image = s.swapImages[imageIndex];
        barriers[1].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdPipelineBarrier(
            s.commandBuffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 0, nullptr, 0, nullptr, 2, barriers);

        vkCmdResetQueryPool(s.commandBuffer, s.queryPool, 0, 4);
        if (s.timestampValidBits > 0) {
            vkCmdWriteTimestamp(
                s.commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, s.queryPool, 0);
        }
        vkCmdBindPipeline(s.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, s.pipeline);
        vkCmdBindDescriptorSets(
            s.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, s.pipelineLayout,
            0, 1, &s.descriptorSet, 0, nullptr);
        vkCmdDispatch(s.commandBuffer, (width + 7) / 8, (height + 7) / 8, 1);
        if (s.timestampValidBits > 0) {
            vkCmdWriteTimestamp(
                s.commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, s.queryPool, 1);
        }

        vkCmdBindPipeline(s.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, s.s2Pipeline);
        vkCmdBindDescriptorSets(
            s.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, s.s2PipelineLayout,
            0, 1, &s.s2DescriptorSet, 0, nullptr);
        vkCmdDispatch(
            s.commandBuffer,
            (((width + 1) / 2) + 15) / 16,
            (((height + 1) / 2) + 15) / 16,
            1);
        if (s.timestampValidBits > 0) {
            vkCmdWriteTimestamp(
                s.commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, s.queryPool, 2);
        }

        VkImageMemoryBarrier colorBarrier{};
        colorBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        colorBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        colorBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        colorBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        colorBarrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        colorBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        colorBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        colorBarrier.image = s.outputImage;
        colorBarrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdPipelineBarrier(
            s.commandBuffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &colorBarrier);

        vkCmdBindPipeline(s.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, s.agxPipeline);
        vkCmdBindDescriptorSets(
            s.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, s.agxPipelineLayout,
            0, 1, &s.agxSets[imageIndex], 0, nullptr);
        vkCmdDispatch(
            s.commandBuffer,
            (static_cast<int>(s.swapWidth) + 7) / 8,
            (static_cast<int>(s.swapHeight) + 7) / 8,
            1);
        if (s.timestampValidBits > 0) {
            vkCmdWriteTimestamp(
                s.commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, s.queryPool, 3);
        }

        VkImageMemoryBarrier presentBarrier{};
        presentBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        presentBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        presentBarrier.dstAccessMask = 0;
        presentBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        presentBarrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        presentBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        presentBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        presentBarrier.image = s.swapImages[imageIndex];
        presentBarrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdPipelineBarrier(
            s.commandBuffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
            0, 0, nullptr, 0, nullptr, 1, &presentBarrier);
        vkEndCommandBuffer(s.commandBuffer);

        VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        VkSubmitInfo submit{};
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.waitSemaphoreCount = 1;
        submit.pWaitSemaphores = &s.imageAvailable;
        submit.pWaitDstStageMask = &waitStage;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &s.commandBuffer;
        submit.signalSemaphoreCount = 1;
        submit.pSignalSemaphores = &s.renderFinished;
        r = vkQueueSubmit(s.queue, 1, &submit, s.fence);
        if (r != VK_SUCCESS) {
            log.push_back("swapchain submit failed: " + std::to_string(r));
            return false;
        }
        vkWaitForFences(s.device, 1, &s.fence, VK_TRUE, UINT64_MAX);
        vkResetFences(s.device, 1, &s.fence);

        if (i >= warmup && s.timestampValidBits > 0) {
            uint64_t ts[4] = {0, 0, 0, 0};
            r = vkGetQueryPoolResults(
                s.device, s.queryPool, 0, 4, sizeof(ts), ts, sizeof(uint64_t),
                VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
            if (r == VK_SUCCESS) {
                double demosaic = static_cast<double>(ts[1] - ts[0]) *
                                  s.timestampPeriod / 1.0e6;
                double s2 = static_cast<double>(ts[2] - ts[1]) *
                            s.timestampPeriod / 1.0e6;
                double agx = static_cast<double>(ts[3] - ts[2]) *
                             s.timestampPeriod / 1.0e6;
                demosaicGpuMs.push_back(demosaic);
                s2GpuMs.push_back(s2);
                agxGpuMs.push_back(agx);
                totalGpuMs.push_back(demosaic + s2 + agx);
            }
        }

        VkPresentInfoKHR present{};
        present.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        present.waitSemaphoreCount = 1;
        present.pWaitSemaphores = &s.renderFinished;
        present.swapchainCount = 1;
        present.pSwapchains = &s.swapchain;
        present.pImageIndices = &imageIndex;
        r = vkQueuePresentKHR(s.queue, &present);
        if (r != VK_SUCCESS && r != VK_SUBOPTIMAL_KHR) {
            log.push_back("swapchain present failed: " + std::to_string(r));
            return false;
        }
        auto t1 = std::chrono::steady_clock::now();
        if (i >= warmup) {
            cpuMs.push_back(
                std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        firstFrame = false;
    }

    auto stats = [](const std::vector<double>& v, double& mean, double& mn, double& mx) {
        mean = mn = mx = 0.0;
        if (v.empty()) return;
        double sum = 0.0;
        mn = v[0];
        mx = v[0];
        for (double x : v) {
            sum += x;
            mn = std::min(mn, x);
            mx = std::max(mx, x);
        }
        mean = sum / static_cast<double>(v.size());
    };

    double cpuMean = 0.0, cpuMin = 0.0, cpuMax = 0.0;
    double demosaicMean = 0.0, demosaicMin = 0.0, demosaicMax = 0.0;
    double s2Mean = 0.0, s2Min = 0.0, s2Max = 0.0;
    double agxMean = 0.0, agxMin = 0.0, agxMax = 0.0;
    double totalMean = 0.0, totalMin = 0.0, totalMax = 0.0;
    stats(cpuMs, cpuMean, cpuMin, cpuMax);
    stats(demosaicGpuMs, demosaicMean, demosaicMin, demosaicMax);
    stats(s2GpuMs, s2Mean, s2Min, s2Max);
    stats(agxGpuMs, agxMean, agxMin, agxMax);
    stats(totalGpuMs, totalMean, totalMin, totalMax);

    std::ostringstream line;
    line << std::fixed << std::setprecision(3)
         << "frames=" << frames
         << " cpuMean=" << cpuMean << "ms"
         << " cpuMin=" << cpuMin << "ms"
         << " cpuMax=" << cpuMax << "ms";
    if (!totalGpuMs.empty()) {
        line << " demosaicGpuMean=" << demosaicMean << "ms"
             << " s2GpuMean=" << s2Mean << "ms"
             << " agxGpuMean=" << agxMean << "ms"
             << " gpuMean=" << totalMean << "ms"
             << " gpuMin=" << totalMin << "ms"
             << " gpuMax=" << totalMax << "ms";
    } else {
        line << " gpu=unavailable";
    }
    line << " swapchain=1";
    log.push_back(line.str());
    return true;
}

bool runBenchmark(State& s, int width, int height, int frames,
                  std::vector<std::string>& log) {
    if (s.useSwapchain) {
        return runSwapBenchmark(s, width, height, frames, log);
    }
    const int warmup = 3;
    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &s.commandBuffer;

    for (int i = 0; i < warmup; ++i) {
        VkResult r = vkQueueSubmit(s.queue, 1, &submit, s.fence);
        if (r != VK_SUCCESS) {
            log.push_back("warmup submit failed: " + std::to_string(r));
            return false;
        }
        vkWaitForFences(s.device, 1, &s.fence, VK_TRUE, UINT64_MAX);
        vkResetFences(s.device, 1, &s.fence);
    }

    std::vector<double> cpuMs;
    std::vector<double> demosaicGpuMs;
    std::vector<double> s2GpuMs;
    std::vector<double> agxGpuMs;
    std::vector<double> totalGpuMs;
    cpuMs.reserve(frames);
    demosaicGpuMs.reserve(frames);
    s2GpuMs.reserve(frames);
    agxGpuMs.reserve(frames);
    totalGpuMs.reserve(frames);
    for (int i = 0; i < frames; ++i) {
        auto t0 = std::chrono::steady_clock::now();
        VkResult r = vkQueueSubmit(s.queue, 1, &submit, s.fence);
        if (r != VK_SUCCESS) {
            log.push_back("submit failed: " + std::to_string(r));
            return false;
        }
        vkWaitForFences(s.device, 1, &s.fence, VK_TRUE, UINT64_MAX);
        auto t1 = std::chrono::steady_clock::now();
        vkResetFences(s.device, 1, &s.fence);
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        cpuMs.push_back(ms);

        if (s.timestampValidBits > 0) {
            uint64_t ts[4] = {0, 0, 0, 0};
            r = vkGetQueryPoolResults(
                s.device, s.queryPool, 0, 4, sizeof(ts), ts, sizeof(uint64_t),
                VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
            if (r == VK_SUCCESS) {
                double demosaic = static_cast<double>(ts[1] - ts[0]) *
                                  s.timestampPeriod / 1.0e6;
                double s2 = static_cast<double>(ts[2] - ts[1]) *
                            s.timestampPeriod / 1.0e6;
                double agx = static_cast<double>(ts[3] - ts[2]) *
                             s.timestampPeriod / 1.0e6;
                demosaicGpuMs.push_back(demosaic);
                s2GpuMs.push_back(s2);
                agxGpuMs.push_back(agx);
                totalGpuMs.push_back(demosaic + s2 + agx);
            }
        }
    }

    auto stats = [](const std::vector<double>& v, double& mean, double& mn, double& mx) {
        mean = mn = mx = 0.0;
        if (v.empty()) return;
        double sum = 0.0;
        mn = v[0];
        mx = v[0];
        for (double x : v) {
            sum += x;
            mn = std::min(mn, x);
            mx = std::max(mx, x);
        }
        mean = sum / static_cast<double>(v.size());
    };

    double cpuMean = 0.0, cpuMin = 0.0, cpuMax = 0.0;
    double demosaicMean = 0.0, demosaicMin = 0.0, demosaicMax = 0.0;
    double s2Mean = 0.0, s2Min = 0.0, s2Max = 0.0;
    double agxMean = 0.0, agxMin = 0.0, agxMax = 0.0;
    double totalMean = 0.0, totalMin = 0.0, totalMax = 0.0;
    stats(cpuMs, cpuMean, cpuMin, cpuMax);
    stats(demosaicGpuMs, demosaicMean, demosaicMin, demosaicMax);
    stats(s2GpuMs, s2Mean, s2Min, s2Max);
    stats(agxGpuMs, agxMean, agxMin, agxMax);
    stats(totalGpuMs, totalMean, totalMin, totalMax);

    std::ostringstream line;
    line << std::fixed << std::setprecision(3)
         << "frames=" << frames
         << " cpuMean=" << cpuMean << "ms"
         << " cpuMin=" << cpuMin << "ms"
         << " cpuMax=" << cpuMax << "ms";
    if (!totalGpuMs.empty()) {
        line << " demosaicGpuMean=" << demosaicMean << "ms"
             << " s2GpuMean=" << s2Mean << "ms"
             << " agxGpuMean=" << agxMean << "ms"
             << " gpuMean=" << totalMean << "ms"
             << " gpuMin=" << totalMin << "ms"
             << " gpuMax=" << totalMax << "ms";
    } else {
        line << " gpu=unavailable";
    }
    log.push_back(line.str());
    return true;
}

void destroy(State& s) {
    if (s.device != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(s.device);
        for (VkImageView view : s.swapViews) {
            if (view != VK_NULL_HANDLE) vkDestroyImageView(s.device, view, nullptr);
        }
        if (s.swapchain != VK_NULL_HANDLE) {
            vkDestroySwapchainKHR(s.device, s.swapchain, nullptr);
        }
        if (s.imageAvailable != VK_NULL_HANDLE) {
            vkDestroySemaphore(s.device, s.imageAvailable, nullptr);
        }
        if (s.renderFinished != VK_NULL_HANDLE) {
            vkDestroySemaphore(s.device, s.renderFinished, nullptr);
        }
        if (s.fence != VK_NULL_HANDLE) vkDestroyFence(s.device, s.fence, nullptr);
        if (s.queryPool != VK_NULL_HANDLE) vkDestroyQueryPool(s.device, s.queryPool, nullptr);
        if (s.commandPool != VK_NULL_HANDLE) vkDestroyCommandPool(s.device, s.commandPool, nullptr);
        if (s.agxPipeline != VK_NULL_HANDLE) vkDestroyPipeline(s.device, s.agxPipeline, nullptr);
        if (s.agxShader != VK_NULL_HANDLE) vkDestroyShaderModule(s.device, s.agxShader, nullptr);
        if (s.agxPipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(s.device, s.agxPipelineLayout, nullptr);
        if (s.agxDescriptorPool != VK_NULL_HANDLE) vkDestroyDescriptorPool(s.device, s.agxDescriptorPool, nullptr);
        if (s.agxDescriptorSetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(s.device, s.agxDescriptorSetLayout, nullptr);
        if (s.s2Pipeline != VK_NULL_HANDLE) vkDestroyPipeline(s.device, s.s2Pipeline, nullptr);
        if (s.s2Shader != VK_NULL_HANDLE) vkDestroyShaderModule(s.device, s.s2Shader, nullptr);
        if (s.s2PipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(s.device, s.s2PipelineLayout, nullptr);
        if (s.s2DescriptorPool != VK_NULL_HANDLE) vkDestroyDescriptorPool(s.device, s.s2DescriptorPool, nullptr);
        if (s.s2DescriptorSetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(s.device, s.s2DescriptorSetLayout, nullptr);
        if (s.pipeline != VK_NULL_HANDLE) vkDestroyPipeline(s.device, s.pipeline, nullptr);
        if (s.shader != VK_NULL_HANDLE) vkDestroyShaderModule(s.device, s.shader, nullptr);
        if (s.pipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(s.device, s.pipelineLayout, nullptr);
        if (s.descriptorPool != VK_NULL_HANDLE) vkDestroyDescriptorPool(s.device, s.descriptorPool, nullptr);
        if (s.descriptorSetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(s.device, s.descriptorSetLayout, nullptr);
        if (s.sampler != VK_NULL_HANDLE) vkDestroySampler(s.device, s.sampler, nullptr);
        if (s.outputAgxView != VK_NULL_HANDLE) vkDestroyImageView(s.device, s.outputAgxView, nullptr);
        if (s.outputAgxImage != VK_NULL_HANDLE) vkDestroyImage(s.device, s.outputAgxImage, nullptr);
        if (s.outputAgxMemory != VK_NULL_HANDLE) vkFreeMemory(s.device, s.outputAgxMemory, nullptr);
        if (s.outputSigmaView != VK_NULL_HANDLE) vkDestroyImageView(s.device, s.outputSigmaView, nullptr);
        if (s.outputSigmaImage != VK_NULL_HANDLE) vkDestroyImage(s.device, s.outputSigmaImage, nullptr);
        if (s.outputSigmaMemory != VK_NULL_HANDLE) vkFreeMemory(s.device, s.outputSigmaMemory, nullptr);
        if (s.outputFlagView != VK_NULL_HANDLE) vkDestroyImageView(s.device, s.outputFlagView, nullptr);
        if (s.outputFlagImage != VK_NULL_HANDLE) vkDestroyImage(s.device, s.outputFlagImage, nullptr);
        if (s.outputFlagMemory != VK_NULL_HANDLE) vkFreeMemory(s.device, s.outputFlagMemory, nullptr);
        if (s.outputView != VK_NULL_HANDLE) vkDestroyImageView(s.device, s.outputView, nullptr);
        if (s.outputImage != VK_NULL_HANDLE) vkDestroyImage(s.device, s.outputImage, nullptr);
        if (s.outputMemory != VK_NULL_HANDLE) vkFreeMemory(s.device, s.outputMemory, nullptr);
        if (s.inputView != VK_NULL_HANDLE) vkDestroyImageView(s.device, s.inputView, nullptr);
        if (s.inputImage != VK_NULL_HANDLE) vkDestroyImage(s.device, s.inputImage, nullptr);
        if (s.inputMemory != VK_NULL_HANDLE) vkFreeMemory(s.device, s.inputMemory, nullptr);
        vkDestroyDevice(s.device, nullptr);
    }
    if (s.surface != VK_NULL_HANDLE && s.instance != VK_NULL_HANDLE) {
        vkDestroySurfaceKHR(s.instance, s.surface, nullptr);
    }
    if (s.instance != VK_NULL_HANDLE) vkDestroyInstance(s.instance, nullptr);
    if (s.ahb != nullptr) AHardwareBuffer_release(s.ahb);
    s = State{};
}

std::string probeSurface(JNIEnv* env, jobject surface) {
    std::ostringstream out;
    ANativeWindow* window = ANativeWindow_fromSurface(env, surface);
    if (window == nullptr) {
        return "encoder surface probe: ANativeWindow_fromSurface failed\n";
    }

    std::vector<VkExtensionProperties> available;
    uint32_t count = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr);
    available.resize(count);
    vkEnumerateInstanceExtensionProperties(nullptr, &count, available.data());
    const bool hasSurface = hasExtension(available, VK_KHR_SURFACE_EXTENSION_NAME);
    const bool hasAndroidSurface =
        hasExtension(available, VK_KHR_ANDROID_SURFACE_EXTENSION_NAME);
    out << "encoder surface: VK_KHR_surface=" << hasSurface
        << " VK_KHR_android_surface=" << hasAndroidSurface << "\n";
    if (!hasSurface || !hasAndroidSurface) {
        ANativeWindow_release(window);
        return out.str();
    }

    std::vector<const char*> enabled = {
        VK_KHR_SURFACE_EXTENSION_NAME,
        VK_KHR_ANDROID_SURFACE_EXTENSION_NAME
    };
    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "AgXCam encoder surface probe";
    app.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ci.pApplicationInfo = &app;
    ci.enabledExtensionCount = static_cast<uint32_t>(enabled.size());
    ci.ppEnabledExtensionNames = enabled.data();

    VkInstance instance = VK_NULL_HANDLE;
    VkResult r = vkCreateInstance(&ci, nullptr, &instance);
    if (r != VK_SUCCESS) {
        out << "vkCreateInstance(surface) failed: " << r << "\n";
        ANativeWindow_release(window);
        return out.str();
    }

    auto createAndroidSurface =
        reinterpret_cast<PFN_vkCreateAndroidSurfaceKHR>(
            vkGetInstanceProcAddr(instance, "vkCreateAndroidSurfaceKHR"));
    if (createAndroidSurface == nullptr) {
        out << "vkCreateAndroidSurfaceKHR missing\n";
        vkDestroyInstance(instance, nullptr);
        ANativeWindow_release(window);
        return out.str();
    }

    VkAndroidSurfaceCreateInfoKHR surfaceInfo{};
    surfaceInfo.sType = VK_STRUCTURE_TYPE_ANDROID_SURFACE_CREATE_INFO_KHR;
    surfaceInfo.window = window;
    VkSurfaceKHR vkSurface = VK_NULL_HANDLE;
    r = createAndroidSurface(instance, &surfaceInfo, nullptr, &vkSurface);
    if (r != VK_SUCCESS) {
        out << "vkCreateAndroidSurfaceKHR failed: " << r << "\n";
        vkDestroyInstance(instance, nullptr);
        ANativeWindow_release(window);
        return out.str();
    }

    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());
    for (VkPhysicalDevice dev : devices) {
        VkSurfaceCapabilitiesKHR caps{};
        r = vkGetPhysicalDeviceSurfaceCapabilitiesKHR(dev, vkSurface, &caps);
        if (r != VK_SUCCESS) continue;
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(dev, &props);
        out << "surface device=" << props.deviceName
            << " extent=" << caps.currentExtent.width << "x" << caps.currentExtent.height
            << " usage=0x" << std::hex << caps.supportedUsageFlags << std::dec
            << " min=" << caps.minImageExtent.width << "x" << caps.minImageExtent.height
            << " max=" << caps.maxImageExtent.width << "x" << caps.maxImageExtent.height
            << "\n";

        uint32_t formatCount = 0;
        vkGetPhysicalDeviceSurfaceFormatsKHR(dev, vkSurface, &formatCount, nullptr);
        std::vector<VkSurfaceFormatKHR> formats(formatCount);
        vkGetPhysicalDeviceSurfaceFormatsKHR(dev, vkSurface, &formatCount, formats.data());
        out << "surface formats=" << formatCount;
        for (uint32_t i = 0; i < std::min<uint32_t>(formatCount, 6); ++i) {
            out << " [" << formats[i].format << "," << formats[i].colorSpace << "]";
        }
        out << "\n";

        uint32_t modeCount = 0;
        vkGetPhysicalDeviceSurfacePresentModesKHR(dev, vkSurface, &modeCount, nullptr);
        std::vector<VkPresentModeKHR> modes(modeCount);
        vkGetPhysicalDeviceSurfacePresentModesKHR(dev, vkSurface, &modeCount, modes.data());
        out << "surface presentModes=" << modeCount;
        for (uint32_t i = 0; i < std::min<uint32_t>(modeCount, 6); ++i) {
            out << " " << modes[i];
        }
        out << "\n";
        break;
    }

    vkDestroySurfaceKHR(instance, vkSurface, nullptr);
    vkDestroyInstance(instance, nullptr);
    ANativeWindow_release(window);
    return out.str();
}

std::string runEncoderProbe(JNIEnv* env, jobject surface, int width, int height,
                            int frames, bool timestamps) {
    ANativeWindow* window = ANativeWindow_fromSurface(env, surface);
    if (window == nullptr) {
        return "encoder pipeline: ANativeWindow_fromSurface failed\n";
    }

    State s;
    s.useSwapchain = true;
    std::vector<std::string> log;
    bool ok = createInstance(s, log, true);
    if (ok) ok = createSurface(s, window, log);
    if (ok) ok = createDevice(s, log, true);
    if (ok && !timestamps) s.timestampValidBits = 0;
    if (ok) ok = createSwapchain(s, log);
    if (ok) ok = allocateAhb(s, width, height, log);
    if (ok) ok = createImportedImage(s, width, height, log);
    if (ok) ok = createOutputImages(s, width, height, log);
    if (ok) ok = createPipeline(s, log);
    if (ok) ok = createS2Pipeline(s, log);
    if (ok) ok = createAgxPipeline(s, log);
    if (ok) ok = createCommands(s, width, height, log);
    if (ok) ok = runBenchmark(s, width, height, frames, log);

    std::ostringstream out;
    out << "encoder pipeline " << width << "x" << height << "\n";
    for (const auto& line : log) out << line << "\n";
    if (!ok) out << "encoder pipeline failed\n";
    destroy(s);
    ANativeWindow_release(window);
    return out.str();
}

std::string runProbe(int width, int height, int frames) {
    State s;
    std::vector<std::string> log;
    bool ok = true;

    if (!createInstance(s, log)) ok = false;
    if (ok && !createDevice(s, log)) ok = false;
    if (ok && !allocateAhb(s, width, height, log)) ok = false;
    if (ok && !createImportedImage(s, width, height, log)) ok = false;
    if (ok && !createOutputImages(s, width, height, log)) ok = false;
    if (ok && !createPipeline(s, log)) ok = false;
    if (ok && !createS2Pipeline(s, log)) ok = false;
    if (ok && !createAgxPipeline(s, log)) ok = false;
    if (ok && !createCommands(s, width, height, log)) ok = false;
    if (ok && !runBenchmark(s, width, height, frames, log)) ok = false;

    std::ostringstream out;
    out << "vulkan probe " << width << "x" << height << "\n";
    for (const auto& line : log) out << line << "\n";
    if (!ok) out << "probe failed\n";
    destroy(s);
    return out.str();
}

}  // namespace

extern "C" JNIEXPORT jstring JNICALL
Java_com_agx_camera_vulkan_VulkanProbe_runProbe(
    JNIEnv* env, jobject /*thiz*/, jint width, jint height, jint frames) {
    std::string result = runProbe(width, height, frames);
    LOGI("%s", result.c_str());
    return env->NewStringUTF(result.c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_agx_camera_vulkan_VulkanProbe_probeSurface(
    JNIEnv* env, jobject /*thiz*/, jobject surface) {
    std::string result = probeSurface(env, surface);
    LOGI("%s", result.c_str());
    return env->NewStringUTF(result.c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_agx_camera_vulkan_VulkanProbe_runEncoderProbe(
    JNIEnv* env, jobject /*thiz*/, jobject surface, jint frames, jboolean timestamps) {
    std::string result = runEncoderProbe(
        env, surface, 4096, 3072, frames, timestamps == JNI_TRUE);
    LOGI("%s", result.c_str());
    return env->NewStringUTF(result.c_str());
}
