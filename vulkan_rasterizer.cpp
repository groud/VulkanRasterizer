#include "vulkan_rasterizer.h"

//#include <vulkan/vulkan.h>

#include <iostream>
#include <vector>
#include <set>
#include <cstring>
#include <algorithm>
#include <fstream>
#include <unordered_map>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#include <chrono>

#define WIDTH 800
#define HEIGHT 600

#define MODEL_PATH "models/chalet.obj"
#define TEXTURE_PATH "models/chalet.jpg"

static std::vector<char> readFile(const std::string& filename) {
  std::ifstream file(filename, std::ios::ate | std::ios::binary);

  if (!file.is_open()) {
    throw std::runtime_error("failed to open file!");
  }

  size_t fileSize = (size_t) file.tellg();
  std::vector<char> buffer(fileSize);

  file.seekg(0);
  file.read(buffer.data(), fileSize);

  file.close();

  return buffer;
}

const char* validation_layers[1] = {"VK_LAYER_LUNARG_standard_validation"};
const std::vector<const char*> deviceExtensions = {
  VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

VkDebugReportCallbackEXT callback;
VkResult VulkanRasterizer::_create_debug_report_callback_EXT(VkInstance instance,
    const VkDebugReportCallbackCreateInfoEXT* pCreateInfo,
    const VkAllocationCallbacks* pAllocator,
    VkDebugReportCallbackEXT* pCallback) {
    auto func = (PFN_vkCreateDebugReportCallbackEXT) vkGetInstanceProcAddr(instance, "vkCreateDebugReportCallbackEXT");
    if (func != nullptr) {
        return func(instance, pCreateInfo, pAllocator, pCallback);
    } else {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}

void VulkanRasterizer::_destroy_debug_report_callback_EXT(VkInstance instance, VkDebugReportCallbackEXT callback, const VkAllocationCallbacks* pAllocator) {
    auto func = (PFN_vkDestroyDebugReportCallbackEXT) vkGetInstanceProcAddr(instance, "vkDestroyDebugReportCallbackEXT");
    if (func != nullptr) {
        func(instance, callback, pAllocator);
    }
}

std::vector<const char*> VulkanRasterizer::_get_required_extensions(bool enable_validation) {
    uint32_t glfwExtensionCount = 0;
    const char** glfwExtensions;
    glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

    std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

    if (enable_validation) {
        extensions.push_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
    }

    return extensions;
}

VKAPI_ATTR VkBool32 VKAPI_CALL VulkanRasterizer::_debug_callback(VkDebugReportFlagsEXT flags, VkDebugReportObjectTypeEXT objType, uint64_t obj, size_t location, int32_t code, const char* layerPrefix, const char* msg, void* userData) {

    std::cerr << "validation layer: " << msg << std::endl;

    return VK_FALSE;
}


bool _check_validation_layer_support(std::vector<const char*> validationLayers) {
  uint32_t layerCount;
  vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
  std::vector<VkLayerProperties> availableLayers(layerCount);
  vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

  for (const char* layerName : validationLayers) {
      bool layerFound = false;

      for (const auto& layerProperties : availableLayers) {
          if (strcmp(layerName, layerProperties.layerName) == 0) {
              layerFound = true;
              break;
          }
      }

      if (!layerFound) {
          return false;
      }
  }

  return true;
}

int VulkanRasterizer::_create_instance() {
  // App infos
  VkApplicationInfo appInfo = {};
  appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  appInfo.pApplicationName = "Basic rasterizer";
  appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.pEngineName = "No Engine";
  appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.apiVersion = VK_API_VERSION_1_0;

  // Create info
  VkInstanceCreateInfo create_info = {};
  create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  create_info.pApplicationInfo = &appInfo;

  // Get the required extensions
  std::vector<const char*> extensions = _get_required_extensions(enable_validation);
  create_info.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
  create_info.ppEnabledExtensionNames = extensions.data();

  // Enable the validation layers
  if (enable_validation) {
    const std::vector<const char*> validationLayers = {
      "VK_LAYER_LUNARG_standard_validation"
    };
    create_info.ppEnabledLayerNames = validation_layers;
    create_info.enabledLayerCount = 1;
  } else {
    create_info.enabledLayerCount = 0;
  }

  // Create the instance
  VkResult err = vkCreateInstance(&create_info, nullptr, &instance);
  if (err != VK_SUCCESS) {
    throw std::runtime_error("failed to create instance!");
  } else {
    std::cout << "Instance created!" << std::endl;
  }

  return 0;
}

int VulkanRasterizer::_enable_debug() {
  // Debug layer
  if (enable_validation) {
    VkDebugReportCallbackCreateInfoEXT callbackinfo = {};
    callbackinfo.sType = VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT;
    callbackinfo.flags = VK_DEBUG_REPORT_ERROR_BIT_EXT | VK_DEBUG_REPORT_WARNING_BIT_EXT;
    callbackinfo.pfnCallback = _debug_callback;

    if (_create_debug_report_callback_EXT(instance, &callbackinfo, nullptr, &callback) != VK_SUCCESS) {
      throw std::runtime_error("failed to set up debug callback!");
    }
  }

  return 0;
}

bool VulkanRasterizer::_check_device_extension_support(VkPhysicalDevice device, const std::vector<const char*> &deviceExtensions) {
    uint32_t extensionCount;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

    std::vector<VkExtensionProperties> availableExtensions(extensionCount);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

    std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

    for (const auto& extension : availableExtensions) {
        requiredExtensions.erase(extension.extensionName);
    }

    return requiredExtensions.empty();
}

int VulkanRasterizer::_get_physical_device_score(VkPhysicalDevice p_physical_device) {
  VkPhysicalDeviceProperties deviceProperties;
  VkPhysicalDeviceFeatures deviceFeatures;
  vkGetPhysicalDeviceProperties(p_physical_device, &deviceProperties);
  vkGetPhysicalDeviceFeatures(p_physical_device, &deviceFeatures);

  int score = 0;
  // Device type
  if(deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
    score = 1;
  } else if (deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) {
    score = 0;
  } else {
    return -1;
  }

  // Geometry shader
  if(!deviceFeatures.geometryShader)
    return -1;

  if (!_pick_queue_families(p_physical_device).is_complete())
    return -1;

  // Required extensions
  if (!_check_device_extension_support(p_physical_device, deviceExtensions))
    return -1;

  // Surface support
  SwapChainSupportDetails swap_chain_support = _query_swap_chain_support(p_physical_device);
  if (swap_chain_support.formats.empty() || swap_chain_support.presentModes.empty())
    return -1;

  // Anisotropy support
  if (!deviceFeatures.samplerAnisotropy)
    return -1;

  return score;
}

struct VulkanRasterizer::QueueFamilyIndices VulkanRasterizer::_pick_queue_families(VkPhysicalDevice p_physical_device) {
  struct QueueFamilyIndices indices;
  // Family count and types
  uint32_t queueFamilyCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(p_physical_device, &queueFamilyCount, nullptr);
  std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
  vkGetPhysicalDeviceQueueFamilyProperties(p_physical_device, &queueFamilyCount, queueFamilies.data());

  int i = 0;
  for (const auto& queueFamily : queueFamilies) {
    if (queueFamily.queueCount > 0 && queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
      indices.graphics_family = i;
    }

    VkBool32 presentSupport = false;
    vkGetPhysicalDeviceSurfaceSupportKHR(p_physical_device, i, surface, &presentSupport);
    if (queueFamily.queueCount > 0 && presentSupport) {
      indices.present_family = i;
    }

    if(indices.is_complete()) {
      break;
    }
    i++;
  }
  return indices;
}


int VulkanRasterizer::_pick_physical_device() {
  // Pick physical device
  physical_device = VK_NULL_HANDLE;
  uint32_t deviceCount = 0;
  vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
  if (deviceCount == 0) {
    throw std::runtime_error("failed to find GPUs with Vulkan support!");
  }
  std::vector<VkPhysicalDevice> devices(deviceCount);
  vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

  // Take the best available GPU
  int max_score = -1;
  for (const auto& device : devices) {
    int score = _get_physical_device_score(device);
    if (score > max_score) {
      physical_device = device;
    }
  }

  if (physical_device == VK_NULL_HANDLE) {
    throw std::runtime_error("failed to find a suitable GPU!");
  }

  return 0;
}

int VulkanRasterizer::_create_logical_device() {
  struct QueueFamilyIndices queue_families = _pick_queue_families(physical_device);

  // Queue creation info
  float queuePriority = 1.0f;

  std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
  std::set<int> uniqueQueueFamilies = {queue_families.graphics_family, queue_families.present_family};

  for (int queue_family : uniqueQueueFamilies) {
    VkDeviceQueueCreateInfo queueCreateInfo = {};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = queue_family;
    queueCreateInfo.queueCount = 1;
    queueCreateInfo.pQueuePriorities = &queuePriority;
    queueCreateInfos.push_back(queueCreateInfo);
  }

  VkDeviceQueueCreateInfo queueCreateInfo = {};
  queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queueCreateInfo.queueFamilyIndex = queue_families.graphics_family;
  queueCreateInfo.queueCount = 1;
  queueCreateInfo.pQueuePriorities = &queuePriority;

  // List of required device features
  VkPhysicalDeviceFeatures deviceFeatures = {};
  deviceFeatures.samplerAnisotropy = VK_TRUE;

  // Logical device creation info
  VkDeviceCreateInfo create_info = {};
  create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  create_info.pQueueCreateInfos = queueCreateInfos.data();
  create_info.queueCreateInfoCount = queueCreateInfos.size();
  create_info.pEnabledFeatures = &deviceFeatures;
  create_info.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
  create_info.ppEnabledExtensionNames = deviceExtensions.data();
  if (enable_validation) {
    create_info.ppEnabledLayerNames = validation_layers;
    create_info.enabledLayerCount = 1;
  } else {
    create_info.enabledLayerCount = 0;
  }

  if (vkCreateDevice(physical_device, &create_info, nullptr, &device) != VK_SUCCESS) {
    throw std::runtime_error("failed to create logical device!");
  }

  vkGetDeviceQueue(device, queue_families.graphics_family, 0, &graphics_queue);
  vkGetDeviceQueue(device, queue_families.present_family, 0, &present_queue);

  return 0;
}

int VulkanRasterizer::_create_surface() {
  // >>>>>>>>>>>>>>>>>
    if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
        throw std::runtime_error("failed to create window surface!");
    }
  // <<<<<<<<<<<<<<<<<
  return 0;
};

struct VulkanRasterizer::SwapChainSupportDetails VulkanRasterizer::_query_swap_chain_support(VkPhysicalDevice p_physical_device) {
    SwapChainSupportDetails details;

    // Retreive the surface capabilities
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(p_physical_device, surface, &details.capabilities);

    // Retreive supported formats
    uint32_t formatCount;
    vkGetPhysicalDeviceSurfaceFormatsKHR(p_physical_device, surface, &formatCount, nullptr);
    if (formatCount != 0) {
      details.formats.resize(formatCount);
      vkGetPhysicalDeviceSurfaceFormatsKHR(p_physical_device, surface, &formatCount, details.formats.data());
    }

    // Retreive the presentation modes
    uint32_t presentModeCount;
    vkGetPhysicalDeviceSurfacePresentModesKHR(p_physical_device, surface, &presentModeCount, nullptr);

    if (presentModeCount != 0) {
      details.presentModes.resize(presentModeCount);
      vkGetPhysicalDeviceSurfacePresentModesKHR(p_physical_device, surface, &presentModeCount, details.presentModes.data());
    }

    return details;
}

VkSurfaceFormatKHR VulkanRasterizer::_pick_swap_surface_format(const std::vector<VkSurfaceFormatKHR>& availableFormats) {
  // Choose the best format we would like when unspecified
  if (availableFormats.size() == 1 && availableFormats[0].format == VK_FORMAT_UNDEFINED) {
    return {VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR};
  }

  // Check if our best format is in the available ones
  for (const auto& availableFormat : availableFormats) {
    if (availableFormat.format == VK_FORMAT_B8G8R8A8_UNORM && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
      return availableFormat;
    }
  }

  // TODO: In corner casse we might choose another one, but it should be good in most cases

  return availableFormats[0];
}

VkPresentModeKHR VulkanRasterizer::_pick_swap_present_mode(const std::vector<VkPresentModeKHR> availablePresentModes) {
  // Choose MAILBOX mode if possible, or IMMEDIATE if not (because FIFO is badly supported by some drivers).
  // On last resort choose FIFO
  VkPresentModeKHR bestMode = VK_PRESENT_MODE_FIFO_KHR;

  for (const auto& availablePresentMode : availablePresentModes) {
    if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
      return availablePresentMode;
    } else if (availablePresentMode == VK_PRESENT_MODE_IMMEDIATE_KHR) {
      bestMode = availablePresentMode;
    }
  }

  return bestMode;
}

VkExtent2D VulkanRasterizer::_pick_swap_extend(const VkSurfaceCapabilitiesKHR& capabilities) {
  // Retreive the swap chain resolution
  if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
    // Match the window resolution
    return capabilities.currentExtent;
  } else {
    // Choose the best fitting window resolution
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    VkExtent2D actualExtent = {
      std::max(capabilities.minImageExtent.width, std::min(capabilities.maxImageExtent.width, static_cast<uint32_t>(width))),
      std::max(capabilities.minImageExtent.height, std::min(capabilities.maxImageExtent.height, static_cast<uint32_t>(height)))
    };
    return actualExtent;
  }
}

int VulkanRasterizer::_create_swap_chain() {
    SwapChainSupportDetails swap_chain_support = _query_swap_chain_support(physical_device);

    VkSurfaceFormatKHR surfaceFormat = _pick_swap_surface_format(swap_chain_support.formats);
    VkPresentModeKHR present_mode = _pick_swap_present_mode(swap_chain_support.presentModes);
    VkExtent2D extent = _pick_swap_extend(swap_chain_support.capabilities);

    uint32_t imageCount = swap_chain_support.capabilities.minImageCount + 1;
    if (swap_chain_support.capabilities.maxImageCount > 0 && imageCount > swap_chain_support.capabilities.maxImageCount) {
      imageCount = swap_chain_support.capabilities.maxImageCount;
    }

    VkSwapchainCreateInfoKHR create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    create_info.surface = surface;
    create_info.minImageCount = imageCount;
    create_info.imageFormat = surfaceFormat.format;
    create_info.imageColorSpace = surfaceFormat.colorSpace;
    create_info.imageExtent = extent;
    create_info.imageArrayLayers = 1;
    create_info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    QueueFamilyIndices indices = _pick_queue_families(physical_device);
    uint32_t queueFamilyIndices[] = {(uint32_t) indices.graphics_family, (uint32_t) indices.present_family};

    if (indices.graphics_family != indices.present_family) {
        create_info.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        create_info.queueFamilyIndexCount = 2;
        create_info.pQueueFamilyIndices = queueFamilyIndices;
    } else {
        create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        create_info.queueFamilyIndexCount = 0; // Optional
        create_info.pQueueFamilyIndices = nullptr; // Optional
    }

    create_info.preTransform = swap_chain_support.capabilities.currentTransform;
    create_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    create_info.presentMode = present_mode;
    create_info.clipped = VK_TRUE;

    create_info.oldSwapchain = VK_NULL_HANDLE;

    if (vkCreateSwapchainKHR(device, &create_info, nullptr, &swap_chain) != VK_SUCCESS) {
        throw std::runtime_error("failed to create swap chain!");
    }

    vkGetSwapchainImagesKHR(device, swap_chain, &imageCount, nullptr);
    swap_chain_images.resize(imageCount);
    vkGetSwapchainImagesKHR(device, swap_chain, &imageCount, swap_chain_images.data());

    swap_chain_image_format = surfaceFormat.format;
    swap_chain_extent = extent;

    return 0;
}

int VulkanRasterizer::_recreate_swap_chain() {
  int width = 0, height = 0;
  while (width == 0 || height == 0) {
    glfwGetFramebufferSize(window, &width, &height);
    glfwWaitEvents();
  }

  vkDeviceWaitIdle(device);

  _cleanup_swap_chain();

  _create_swap_chain();
  _create_image_views();
  _create_render_pass();
  _create_graphic_pipeline();
  _create_depth_resources();
  _create_frame_buffers();
  _create_command_buffers();

  return 0;
}

VkImageView VulkanRasterizer::_create_image_view(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags, uint32_t mip_levels) {
  // Create an image view from an image
  VkImageViewCreateInfo create_info = {};
  create_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  create_info.image = image;
  create_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
  create_info.format = format;
  create_info.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
  create_info.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
  create_info.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
  create_info.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
  create_info.subresourceRange.aspectMask = aspectFlags;
  create_info.subresourceRange.baseMipLevel = 0;
  create_info.subresourceRange.levelCount = mip_levels;
  create_info.subresourceRange.baseArrayLayer = 0;
  create_info.subresourceRange.layerCount = 1;

  VkImageView image_view;
  if (vkCreateImageView(device, &create_info, nullptr, &image_view) != VK_SUCCESS) {
    throw std::runtime_error("failed to create image views!");
  }

  return image_view;
}

int VulkanRasterizer::_create_image_views() {
  // Create the swapchain image views
  swap_chain_image_views.resize(swap_chain_images.size());

  for (size_t i = 0; i < swap_chain_images.size(); i++) {
    swap_chain_image_views[i] = _create_image_view(swap_chain_images[i], swap_chain_image_format, VK_IMAGE_ASPECT_COLOR_BIT, 1);
  }

  return 0;
}

VkShaderModule VulkanRasterizer::_create_shader_module(const std::vector<char>& code) {
  VkShaderModuleCreateInfo createInfo = {};
  createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  createInfo.codeSize = code.size();
  createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

  VkShaderModule shaderModule;
  if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
    throw std::runtime_error("failed to create shader module!");
  }

  return shaderModule;
}

int VulkanRasterizer::_create_render_pass() {
  // Attachment description
  VkAttachmentDescription colorAttachment = {};
  colorAttachment.format = swap_chain_image_format;
  colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
  colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

  VkAttachmentDescription depthAttachment = {};
  depthAttachment.format = _find_depth_format();
  depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
  depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

  std::array<VkAttachmentDescription, 2> attachments = {colorAttachment, depthAttachment};

  // Subpasses
  VkAttachmentReference colorAttachmentRef = {};
  colorAttachmentRef.attachment = 0;
  colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

  VkAttachmentReference depthAttachmentRef = {};
  depthAttachmentRef.attachment = 1;
  depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

  VkSubpassDescription subpass = {};
  subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  subpass.colorAttachmentCount = 1;
  subpass.pColorAttachments = &colorAttachmentRef;
  subpass.pDepthStencilAttachment = &depthAttachmentRef;

  // Subpass dependency
  VkSubpassDependency dependency = {};
  dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
  dependency.dstSubpass = 0;
  dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  dependency.srcAccessMask = 0;
  dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

  // Create render pass
  VkRenderPassCreateInfo renderPassInfo = {};
  renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
  renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
  renderPassInfo.pAttachments = attachments.data();
  renderPassInfo.subpassCount = 1;
  renderPassInfo.pSubpasses = &subpass;
  renderPassInfo.dependencyCount = 1;
  renderPassInfo.pDependencies = &dependency;

  if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &render_pass) != VK_SUCCESS) {
    throw std::runtime_error("failed to create render pass!");
  }

  return 0;
}

int VulkanRasterizer::_create_graphic_pipeline() {
  // Shaders
  auto vert_shader_code = readFile("shaders/vert.spv");
  auto frag_shader_code = readFile("shaders/frag.spv");
  VkShaderModule vertShaderModule = _create_shader_module(vert_shader_code);
  VkShaderModule fragShaderModule = _create_shader_module(frag_shader_code);

  VkPipelineShaderStageCreateInfo vertShaderStageInfo = {};
  vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
  vertShaderStageInfo.module = vertShaderModule;
  vertShaderStageInfo.pName = "main"; // Entry point in the shader

  VkPipelineShaderStageCreateInfo fragShaderStageInfo = {};
  fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  fragShaderStageInfo.module = fragShaderModule;
  fragShaderStageInfo.pName = "main";

  VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

  // Pipeline
  VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};
  vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

  auto bindingDescription = Vertex::getBindingDescription();
  auto attributeDescriptions = Vertex::getAttributeDescriptions();

  vertexInputInfo.vertexBindingDescriptionCount = 1;
  vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
  vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
  vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();


  // How do we use vertex (here=triangle list)
  VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
  inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  inputAssembly.primitiveRestartEnable = VK_FALSE;

  // Viewport and scissors
  VkViewport viewport = {};
  viewport.x = 0.0f;
  viewport.y = 0.0f;
  viewport.width = (float) swap_chain_extent.width;
  viewport.height = (float) swap_chain_extent.height;
  viewport.minDepth = 0.0f;
  viewport.maxDepth = 1.0f;

  VkRect2D scissor = {};
  scissor.offset = {0, 0};
  scissor.extent = swap_chain_extent;

  VkPipelineViewportStateCreateInfo viewportState = {};
  viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
  viewportState.viewportCount = 1;
  viewportState.pViewports = &viewport;
  viewportState.scissorCount = 1;
  viewportState.pScissors = &scissor;

  // Rasterizer
  VkPipelineRasterizationStateCreateInfo rasterizer = {};
  rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  rasterizer.depthClampEnable = VK_FALSE;
  rasterizer.rasterizerDiscardEnable = VK_FALSE;
  rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
  rasterizer.lineWidth = 1.0f;
  rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
  rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
  rasterizer.depthBiasEnable = VK_FALSE;
  rasterizer.depthBiasConstantFactor = 0.0f; // Optional
  rasterizer.depthBiasClamp = 0.0f; // Optional
  rasterizer.depthBiasSlopeFactor = 0.0f; // Optional

  // Multisampling (for anti aliasing)
  VkPipelineMultisampleStateCreateInfo multisampling = {};
  multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  multisampling.sampleShadingEnable = VK_FALSE;
  multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
  multisampling.minSampleShading = 1.0f; // Optional
  multisampling.pSampleMask = nullptr; // Optional
  multisampling.alphaToCoverageEnable = VK_FALSE; // Optional
  multisampling.alphaToOneEnable = VK_FALSE; // Optional

  // Color blending
  VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
  colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
  colorBlendAttachment.blendEnable = VK_FALSE; // No blending !
  colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE; // Optional
  colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO; // Optional
  colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD; // Optional
  colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE; // Optional
  colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO; // Optional
  colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD; // Optional

  VkPipelineColorBlendStateCreateInfo colorBlending = {};
  colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  colorBlending.logicOpEnable = VK_FALSE;
  colorBlending.logicOp = VK_LOGIC_OP_COPY; // Optional
  colorBlending.attachmentCount = 1;
  colorBlending.pAttachments = &colorBlendAttachment;
  colorBlending.blendConstants[0] = 0.0f; // Optional
  colorBlending.blendConstants[1] = 0.0f; // Optional
  colorBlending.blendConstants[2] = 0.0f; // Optional
  colorBlending.blendConstants[3] = 0.0f; // Optional

  // Depth stencil
  VkPipelineDepthStencilStateCreateInfo depthStencil = {};
  depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
  depthStencil.depthTestEnable = VK_TRUE;
  depthStencil.depthWriteEnable = VK_TRUE;
  depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
  depthStencil.depthBoundsTestEnable = VK_FALSE;
  depthStencil.minDepthBounds = 0.0f; // Optional
  depthStencil.maxDepthBounds = 1.0f; // Optional
  depthStencil.stencilTestEnable = VK_FALSE;
  depthStencil.front = {}; // Optional
  depthStencil.back = {}; // Optional

  // Pipeline layout
  VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
  pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipelineLayoutInfo.setLayoutCount = 1;
  pipelineLayoutInfo.pSetLayouts = &descriptor_set_layout;
  pipelineLayoutInfo.pushConstantRangeCount = 0; // Optional
  pipelineLayoutInfo.pPushConstantRanges = nullptr; // Optional

  if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipeline_layout) != VK_SUCCESS) {
    throw std::runtime_error("failed to create pipeline layout!");
  }

  // Create the pipeline
  VkGraphicsPipelineCreateInfo pipelineInfo = {};
  pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
  pipelineInfo.stageCount = 2;
  pipelineInfo.pStages = shaderStages;
  pipelineInfo.pVertexInputState = &vertexInputInfo;
  pipelineInfo.pInputAssemblyState = &inputAssembly;
  pipelineInfo.pViewportState = &viewportState;
  pipelineInfo.pRasterizationState = &rasterizer;
  pipelineInfo.pMultisampleState = &multisampling;
  pipelineInfo.pDepthStencilState = &depthStencil;
  pipelineInfo.pColorBlendState = &colorBlending;
  pipelineInfo.pDynamicState = nullptr; // Optional
  pipelineInfo.layout = pipeline_layout;
  pipelineInfo.renderPass = render_pass;
  pipelineInfo.subpass = 0;
  pipelineInfo.basePipelineHandle = VK_NULL_HANDLE; // Optional
  pipelineInfo.basePipelineIndex = -1; // Optional

  if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS) {
    throw std::runtime_error("failed to create graphics pipeline!");
  }

  vkDestroyShaderModule(device, fragShaderModule, nullptr);
  vkDestroyShaderModule(device, vertShaderModule, nullptr);

  return 0;
}


int VulkanRasterizer::_create_frame_buffers() {
  // Create the frame buffers
  swap_chain_framebuffers.resize(swap_chain_image_views.size());
  for (size_t i = 0; i < swap_chain_image_views.size(); i++) {
    std::array<VkImageView, 2> attachments = {
      swap_chain_image_views[i],
      depth_image_view
    };

    VkFramebufferCreateInfo framebufferInfo = {};
    framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    framebufferInfo.renderPass = render_pass;
    framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
    framebufferInfo.pAttachments = attachments.data();
    framebufferInfo.width = swap_chain_extent.width;
    framebufferInfo.height = swap_chain_extent.height;
    framebufferInfo.layers = 1;

    if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swap_chain_framebuffers[i]) != VK_SUCCESS) {
        throw std::runtime_error("failed to create framebuffer!");
    }
  }

  return 0;
}

int VulkanRasterizer::_create_command_pool() {
  QueueFamilyIndices queueFamilyIndices = _pick_queue_families(physical_device);

  VkCommandPoolCreateInfo poolInfo = {};
  poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  poolInfo.queueFamilyIndex = queueFamilyIndices.graphics_family;
  poolInfo.flags = 0; // Optional

  if (vkCreateCommandPool(device, &poolInfo, nullptr, &command_pool) != VK_SUCCESS) {
    throw std::runtime_error("failed to create command pool!");
  }

  return 0;
}

uint32_t VulkanRasterizer::_find_memory_type(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
  VkPhysicalDeviceMemoryProperties memProperties;
  vkGetPhysicalDeviceMemoryProperties(physical_device, &memProperties);

  for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
    if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
      return i;
    }
  }

  throw std::runtime_error("failed to find suitable memory type!");
}


int VulkanRasterizer::_create_buffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& buffer_memory) {
  // Create a vertex buffer (without any memory binded)
  VkBufferCreateInfo bufferInfo = {};
  bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufferInfo.size = size;
  bufferInfo.usage = usage;
  bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
    throw std::runtime_error("failed to create vertex buffer!");
  }

  // Allocate the required memory
  VkMemoryRequirements memRequirements;
  vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

  VkMemoryAllocateInfo allocInfo = {};
  allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.allocationSize = memRequirements.size;
  allocInfo.memoryTypeIndex = _find_memory_type(memRequirements.memoryTypeBits, properties);

  if (vkAllocateMemory(device, &allocInfo, nullptr, &buffer_memory) != VK_SUCCESS) {
    throw std::runtime_error("failed to allocate vertex buffer memory!");
  }

  // Bind the buffer to the memory
  vkBindBufferMemory(device, buffer, buffer_memory, 0);
  return 0;
}

VkCommandBuffer VulkanRasterizer::_begin_single_time_commands() {
  // Allocate a new command buffer
  VkCommandBufferAllocateInfo allocInfo = {};
  allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandPool = command_pool;
  allocInfo.commandBufferCount = 1;

  VkCommandBuffer commandBuffer;
  vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

  VkCommandBufferBeginInfo beginInfo = {};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

  // Start the command buffer
  vkBeginCommandBuffer(commandBuffer, &beginInfo);

  return commandBuffer;
}

void VulkanRasterizer::_end_single_time_commands(VkCommandBuffer command_buffer) {
  // End command buffer
  vkEndCommandBuffer(command_buffer);

  // Execute the command buffer
  VkSubmitInfo submitInfo = {};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &command_buffer;

  vkQueueSubmit(graphics_queue, 1, &submitInfo, VK_NULL_HANDLE);
  vkQueueWaitIdle(graphics_queue);

  // Free the command buffer
  vkFreeCommandBuffers(device, command_pool, 1, &command_buffer);
}


void VulkanRasterizer::_copy_buffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
  // Copy a buffer into another
  VkCommandBuffer command_buffer = _begin_single_time_commands();

  VkBufferCopy copyRegion = {};
  copyRegion.size = size;
  vkCmdCopyBuffer(command_buffer, srcBuffer, dstBuffer, 1, &copyRegion);

  _end_single_time_commands(command_buffer);
}

void VulkanRasterizer::_copy_buffer_to_image(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height) {
  // Copy a buffer into an image
  VkCommandBuffer command_buffer = _begin_single_time_commands();

  VkBufferImageCopy region = {};
  region.bufferOffset = 0;
  region.bufferRowLength = 0;
  region.bufferImageHeight = 0;
  region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  region.imageSubresource.mipLevel = 0;
  region.imageSubresource.baseArrayLayer = 0;
  region.imageSubresource.layerCount = 1;
  region.imageOffset = {0, 0, 0};
  region.imageExtent = {
    width,
    height,
    1
  };

  vkCmdCopyBufferToImage(command_buffer, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
  // The 4th parameters assumes the image has already been transitioned to an optimal layout for receiving data

  _end_single_time_commands(command_buffer);
}
int VulkanRasterizer::_transition_image_layout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout, uint32_t mip_levels) {
  // Transition an image from one layout to another
  VkCommandBuffer command_buffer = _begin_single_time_commands();

  VkImageMemoryBarrier barrier = {};
  barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  barrier.oldLayout = oldLayout;
  barrier.newLayout = newLayout;
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.image = image;
  if (newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    if (_has_stencil_component(format)) {
      barrier.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
    }
  } else {
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  }
  barrier.subresourceRange.baseMipLevel = 0;
  barrier.subresourceRange.levelCount = mip_levels;
  barrier.subresourceRange.baseArrayLayer = 0;
  barrier.subresourceRange.layerCount = 1;

  VkPipelineStageFlags sourceStage;
  VkPipelineStageFlags destinationStage;

  if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

    sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
  } else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
  } else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    destinationStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
  } else {
    throw std::invalid_argument("unsupported layout transition!");
  }


  vkCmdPipelineBarrier(
      command_buffer,
      sourceStage, destinationStage,
      0,
      0, nullptr,
      0, nullptr,
      1, &barrier);

  _end_single_time_commands(command_buffer);
  return 0;
}

int VulkanRasterizer::_create_image(uint32_t width, uint32_t height, uint32_t mip_levels, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& image_memory) {
  // Create an image
  VkImageCreateInfo imageInfo = {};
  imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  imageInfo.imageType = VK_IMAGE_TYPE_2D;
  imageInfo.extent.width = width;
  imageInfo.extent.height = height;
  imageInfo.extent.depth = 1;
  imageInfo.mipLevels = mip_levels;
  imageInfo.arrayLayers = 1;
  imageInfo.format = format;
  imageInfo.tiling = tiling;
  imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  imageInfo.usage = usage;
  imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
  imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  if (vkCreateImage(device, &imageInfo, nullptr, &image) != VK_SUCCESS) {
    throw std::runtime_error("failed to create image!");
  }

  // Allocate memory for the image
  VkMemoryRequirements memRequirements;
  vkGetImageMemoryRequirements(device, image, &memRequirements);

  VkMemoryAllocateInfo allocInfo = {};
  allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.allocationSize = memRequirements.size;
  allocInfo.memoryTypeIndex = _find_memory_type(memRequirements.memoryTypeBits, properties);

  if (vkAllocateMemory(device, &allocInfo, nullptr, &image_memory) != VK_SUCCESS) {
    throw std::runtime_error("failed to allocate image memory!");
  }

  vkBindImageMemory(device, image, image_memory, 0);

  return 0;
}

int VulkanRasterizer::_create_mipmaps(VkImage image, VkFormat format, int32_t texWidth, int32_t texHeight, uint32_t mipLevels) {

  // Check if image format supports linear blitting
  VkFormatProperties formatProperties;
  vkGetPhysicalDeviceFormatProperties(physical_device, format, &formatProperties);
  if (!(formatProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT)) {
    throw std::runtime_error("texture image format does not support linear blitting!");
  }


    // Generate the mipmaps
    VkCommandBuffer commandBuffer = _begin_single_time_commands();

    VkImageMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.image = image;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    barrier.subresourceRange.levelCount = 1;

    int32_t mipWidth = texWidth;
    int32_t mipHeight = texHeight;

    for (uint32_t i = 1; i < mipLevels; i++) {
      barrier.subresourceRange.baseMipLevel = i - 1;
      barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
      barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
      barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
      barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

      vkCmdPipelineBarrier(commandBuffer,
          VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
          0, nullptr,
          0, nullptr,
          1, &barrier);

      VkImageBlit blit = {};
      blit.srcOffsets[0] = { 0, 0, 0 };
      blit.srcOffsets[1] = { mipWidth, mipHeight, 1 };
      blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      blit.srcSubresource.mipLevel = i - 1;
      blit.srcSubresource.baseArrayLayer = 0;
      blit.srcSubresource.layerCount = 1;
      blit.dstOffsets[0] = { 0, 0, 0 };
      blit.dstOffsets[1] = { mipWidth > 1 ? mipWidth / 2 : 1, mipHeight > 1 ? mipHeight / 2 : 1, 1 };
      blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      blit.dstSubresource.mipLevel = i;
      blit.dstSubresource.baseArrayLayer = 0;
      blit.dstSubresource.layerCount = 1;

      vkCmdBlitImage(commandBuffer,
          image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
          image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
          1, &blit,
          VK_FILTER_LINEAR);

      barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
      barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
      barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

      vkCmdPipelineBarrier(commandBuffer,
          VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
          0, nullptr,
          0, nullptr,
          1, &barrier);

      if (mipWidth > 1) mipWidth /= 2;
      if (mipHeight > 1) mipHeight /= 2;
    }

    barrier.subresourceRange.baseMipLevel = mipLevels - 1;
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(commandBuffer,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
        0, nullptr,
        0, nullptr,
        1, &barrier);

    _end_single_time_commands(commandBuffer);

    return 0;
}

int VulkanRasterizer::_create_texture_image() {
  // Get image from file
  int texWidth, texHeight, texChannels;
  stbi_uc* pixels = stbi_load(TEXTURE_PATH, &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
  VkDeviceSize imageSize = texWidth * texHeight * 4;

  if (!pixels) {
    throw std::runtime_error("failed to load texture image!");
  }
  mip_levels = static_cast<uint32_t>(std::floor(std::log2(std::max(texWidth, texHeight)))) + 1;

  // Create staging buffer
  VkBuffer staging_buffer;
  VkDeviceMemory staging_buffer_memory;
  _create_buffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, staging_buffer, staging_buffer_memory);

  void* data;
  vkMapMemory(device, staging_buffer_memory, 0, imageSize, 0, &data);
  memcpy(data, pixels, static_cast<size_t>(imageSize));
  vkUnmapMemory(device, staging_buffer_memory);

  stbi_image_free(pixels);

  // Create image
  _create_image(texWidth, texHeight, mip_levels, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, texture_image, texture_image_memory);

  // Transition the image to a layout optimal for receiving data
  _transition_image_layout(texture_image, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, mip_levels);

  // Copy the buffer to the image
  _copy_buffer_to_image(staging_buffer, texture_image, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));

  // Transition the image to a layout optimal for
  //_transition_image_layout(texture_image, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, mip_levels);
  _create_mipmaps(texture_image, VK_FORMAT_R8G8B8A8_UNORM, texWidth, texHeight, mip_levels);

  vkDestroyBuffer(device, staging_buffer, nullptr);
  vkFreeMemory(device, staging_buffer_memory, nullptr);
  return 0;
}

int VulkanRasterizer::_create_texture_image_view() {
  // Create an image view
  texture_image_view= _create_image_view(texture_image, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT, mip_levels);

  return 0;
}

int VulkanRasterizer::_create_texture_sampler() {
  // Create a sampler to interpretate the texture texels
  VkSamplerCreateInfo samplerInfo = {};
  samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  samplerInfo.magFilter = VK_FILTER_LINEAR;
  samplerInfo.minFilter = VK_FILTER_LINEAR;
  samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  samplerInfo.anisotropyEnable = VK_TRUE;
  samplerInfo.maxAnisotropy = 16;
  samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
  samplerInfo.unnormalizedCoordinates = VK_FALSE;
  samplerInfo.compareEnable = VK_FALSE;
  samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
  samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
  samplerInfo.mipLodBias = 0.0f;
  samplerInfo.minLod = 0.0f;
//  samplerInfo.minLod = static_cast<float>(mip_levels) / 2;
  samplerInfo.maxLod = static_cast<float>(mip_levels);

  if (vkCreateSampler(device, &samplerInfo, nullptr, &texture_sampler) != VK_SUCCESS) {
    throw std::runtime_error("failed to create texture sampler!");
  }
  return 0;
}

int VulkanRasterizer::_load_model() {
  tinyobj::attrib_t attrib;
  std::vector<tinyobj::shape_t> shapes;
  std::vector<tinyobj::material_t> materials;
  std::string err;
  std::unordered_map<Vertex, uint32_t> uniqueVertices = {};

  if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &err, MODEL_PATH)) {
    throw std::runtime_error(err);
  }

  for (const auto& shape : shapes) {
    for (const auto& index : shape.mesh.indices) {
      Vertex vertex = {};

      vertex.pos = {
        attrib.vertices[3 * index.vertex_index + 0],
        attrib.vertices[3 * index.vertex_index + 1],
        attrib.vertices[3 * index.vertex_index + 2]
      };

      vertex.texCoord = {
        attrib.texcoords[2 * index.texcoord_index + 0],
        1.0f - attrib.texcoords[2 * index.texcoord_index + 1]
      };

      vertex.color = {1.0f, 1.0f, 1.0f};

      if (uniqueVertices.count(vertex) == 0) {
        uniqueVertices[vertex] = static_cast<uint32_t>(vertices.size());
        vertices.push_back(vertex);
      }

      indices.push_back(uniqueVertices[vertex]);
    }
  }

  // Vertex deduplication

  return 0;
}

int VulkanRasterizer::_create_vertex_buffer() {
  VkDeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();

  // Create a staging buffer (in host memory)
  VkBuffer staging_buffer;
  VkDeviceMemory staging_buffer_memory;
  _create_buffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, staging_buffer, staging_buffer_memory);

  // Fills the staging buffer
  void* data;
  vkMapMemory(device, staging_buffer_memory, 0, bufferSize, 0, &data);
  memcpy(data, vertices.data(), (size_t) bufferSize);
  vkUnmapMemory(device, staging_buffer_memory);

  _create_buffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertex_buffer, vertex_buffer_memory);

  // Copy the staging buffer into the vertex one (from host memory to GPU one)
  _copy_buffer(staging_buffer, vertex_buffer, bufferSize);

  // Free and destroy the staging buffer
  vkDestroyBuffer(device, staging_buffer, nullptr);
  vkFreeMemory(device, staging_buffer_memory, nullptr);
  return 0;
}

int VulkanRasterizer::_create_index_buffer() {
  VkDeviceSize bufferSize = sizeof(indices[0]) * indices.size();

  // Create a staging buffer (in host memory)
  VkBuffer staging_buffer;
  VkDeviceMemory staging_buffer_memory;
  _create_buffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, staging_buffer, staging_buffer_memory);

  // Fills the staging buffer
  void* data;
  vkMapMemory(device, staging_buffer_memory, 0, bufferSize, 0, &data);
  memcpy(data, indices.data(), (size_t) bufferSize);
  vkUnmapMemory(device, staging_buffer_memory);

  _create_buffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, index_buffer, index_buffer_memory);

  // Copy the staging buffer into the vertex one (from host memory to GPU one)
  _copy_buffer(staging_buffer, index_buffer, bufferSize);

  // Free and destroy the staging buffer
  vkDestroyBuffer(device, staging_buffer, nullptr);
  vkFreeMemory(device, staging_buffer_memory, nullptr);
  return 0;
}

int VulkanRasterizer::_create_uniform_buffers() {
  // Create a buffer for each image in the swap chain
  VkDeviceSize bufferSize = sizeof(UniformBufferObject);

  uniform_buffers.resize(swap_chain_images.size());
  uniform_buffers_memory.resize(swap_chain_images.size());

  for (size_t i = 0; i < swap_chain_images.size(); i++) {
    _create_buffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, uniform_buffers[i], uniform_buffers_memory[i]);
  }

  return 0;
}

int VulkanRasterizer::_create_descriptor_pool() {
  // Create descriptor pools
  std::array<VkDescriptorPoolSize, 2> poolSizes = {};
  poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  poolSizes[0].descriptorCount = static_cast<uint32_t>(swap_chain_images.size());
  poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  poolSizes[1].descriptorCount = static_cast<uint32_t>(swap_chain_images.size());

  VkDescriptorPoolCreateInfo poolInfo = {};
  poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
  poolInfo.pPoolSizes = poolSizes.data();
  poolInfo.maxSets = static_cast<uint32_t>(swap_chain_images.size());

  if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptor_pool) != VK_SUCCESS) {
    throw std::runtime_error("failed to create descriptor pool!");
  }

  return 0;
}

int VulkanRasterizer::_create_descriptor_set_layout() {
  // Create a descuiptor set for uniforms
  VkDescriptorSetLayoutBinding uboLayoutBinding = {};
  uboLayoutBinding.binding = 0;
  uboLayoutBinding.descriptorCount = 1;
  uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT; // Only for the vertex shader
  uboLayoutBinding.pImmutableSamplers = nullptr; // Optional

  VkDescriptorSetLayoutBinding samplerLayoutBinding = {};
  samplerLayoutBinding.binding = 1;
  samplerLayoutBinding.descriptorCount = 1;
  samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT; // Only for fragment shader
  samplerLayoutBinding.pImmutableSamplers = nullptr;

  std::array<VkDescriptorSetLayoutBinding, 2> bindings = {uboLayoutBinding, samplerLayoutBinding};

  VkDescriptorSetLayoutCreateInfo layoutInfo = {};
  layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
  layoutInfo.pBindings = bindings.data();

  if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptor_set_layout) != VK_SUCCESS) {
    throw std::runtime_error("failed to create descriptor set layout!");
  }

  return 0;
}

int VulkanRasterizer::_create_descriptor_sets() {
  // Create a descriptor set for each uniform buffer
  std::vector<VkDescriptorSetLayout> layouts(swap_chain_images.size(), descriptor_set_layout);
  VkDescriptorSetAllocateInfo allocInfo = {};
  allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  allocInfo.descriptorPool = descriptor_pool;
  allocInfo.descriptorSetCount = static_cast<uint32_t>(swap_chain_images.size());
  allocInfo.pSetLayouts = layouts.data();

  descriptor_sets.resize(swap_chain_images.size());
  if (vkAllocateDescriptorSets(device, &allocInfo, &descriptor_sets[0]) != VK_SUCCESS) {
    throw std::runtime_error("failed to allocate descriptor sets!");
  }

  for (size_t i = 0; i < swap_chain_images.size(); i++) {
    VkDescriptorBufferInfo bufferInfo = {};
    bufferInfo.buffer = uniform_buffers[i];
    bufferInfo.offset = 0;
    bufferInfo.range = sizeof(UniformBufferObject);

    VkDescriptorImageInfo imageInfo = {};
    imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    imageInfo.imageView = texture_image_view;
    imageInfo.sampler = texture_sampler;

    std::array<VkWriteDescriptorSet, 2> descriptorWrites = {};

    descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[0].dstSet = descriptor_sets[i];
    descriptorWrites[0].dstBinding = 0;
    descriptorWrites[0].dstArrayElement = 0;
    descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    descriptorWrites[0].descriptorCount = 1;
    descriptorWrites[0].pBufferInfo = &bufferInfo;

    descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[1].dstSet = descriptor_sets[i];
    descriptorWrites[1].dstBinding = 1;
    descriptorWrites[1].dstArrayElement = 0;
    descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    descriptorWrites[1].descriptorCount = 1;
    descriptorWrites[1].pImageInfo = &imageInfo;

    vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
  }

  return 0;
}

VkFormat VulkanRasterizer::_find_supported_format(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features) {
  // Get acceptable image formats
  for (VkFormat format : candidates) {
    VkFormatProperties props;
    vkGetPhysicalDeviceFormatProperties(physical_device, format, &props);

    if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features) {
      return format;
    } else if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features) {
      return format;
    }
  }

  throw std::runtime_error("failed to find supported format!");
}

VkFormat VulkanRasterizer::_find_depth_format() {
  return _find_supported_format(
      {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT},
      VK_IMAGE_TILING_OPTIMAL,
      VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT
      );
}

bool VulkanRasterizer::_has_stencil_component(VkFormat format) {
    return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT;
}

int VulkanRasterizer::_create_depth_resources() {
  VkFormat depth_format = _find_depth_format();

  _create_image(swap_chain_extent.width, swap_chain_extent.height, 1, depth_format, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, depth_image, depth_image_memory);

  depth_image_view = _create_image_view(depth_image, depth_format, VK_IMAGE_ASPECT_DEPTH_BIT, 1);

  _transition_image_layout(depth_image, depth_format, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, 1);

  return 0;
}

int VulkanRasterizer::_create_command_buffers() {
  command_buffers.resize(swap_chain_framebuffers.size());

  // Allocate the command buffers
  VkCommandBufferAllocateInfo allocInfo = {};
  allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocInfo.commandPool = command_pool;
  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandBufferCount = (uint32_t) command_buffers.size();

  if (vkAllocateCommandBuffers(device, &allocInfo, command_buffers.data()) != VK_SUCCESS) {
    throw std::runtime_error("failed to allocate command buffers!");
  }

  for (size_t i = 0; i < command_buffers.size(); i++) {
    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
    beginInfo.pInheritanceInfo = nullptr; // Optional

    if (vkBeginCommandBuffer(command_buffers[i], &beginInfo) != VK_SUCCESS) {
      throw std::runtime_error("failed to begin recording command buffer!");
    }

    // Fills the render pass infos
    VkRenderPassBeginInfo renderPassInfo = {};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassInfo.renderPass = render_pass;
    renderPassInfo.framebuffer = swap_chain_framebuffers[i];
    renderPassInfo.renderArea.offset = {0, 0};
    renderPassInfo.renderArea.extent = swap_chain_extent;
    std::array<VkClearValue, 2> clearValues = {};
    clearValues[0].color = {0.0f, 0.0f, 0.0f, 1.0f};
    clearValues[1].depthStencil = {1.0f, 0};
    renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
    renderPassInfo.pClearValues = clearValues.data();

    // Fills in the command buffer
    vkCmdBeginRenderPass(command_buffers[i], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(command_buffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

    VkBuffer vertex_buffers[] = {vertex_buffer};
    VkDeviceSize offsets[] = {0};
    vkCmdBindVertexBuffers(command_buffers[i], 0, 1, vertex_buffers, offsets);
    vkCmdBindIndexBuffer(command_buffers[i], index_buffer, 0, VK_INDEX_TYPE_UINT32);
    vkCmdBindDescriptorSets(command_buffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 0, 1, &descriptor_sets[i], 0, nullptr);
    vkCmdDrawIndexed(command_buffers[i], static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);
    vkCmdEndRenderPass(command_buffers[i]);

    if (vkEndCommandBuffer(command_buffers[i]) != VK_SUCCESS) {
      throw std::runtime_error("failed to record command buffer!");
    }
  }
  return 0;
}

int VulkanRasterizer::_create_semaphores_and_fences() {
  image_available_semaphores.resize(MAX_FRAMES_IN_FLIGHT);
  render_finished_semaphores.resize(MAX_FRAMES_IN_FLIGHT);
  in_flight_fences.resize(MAX_FRAMES_IN_FLIGHT);

  VkSemaphoreCreateInfo semaphoreInfo = {};
  semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

  VkFenceCreateInfo fenceInfo = {};
  fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

  for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
    if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &image_available_semaphores[i]) != VK_SUCCESS ||
        vkCreateSemaphore(device, &semaphoreInfo, nullptr, &render_finished_semaphores[i]) != VK_SUCCESS ||
        vkCreateFence(device, &fenceInfo, nullptr, &in_flight_fences[i]) != VK_SUCCESS) {
      throw std::runtime_error("failed to create synchronization objects for a frame!");
    }
  }
  return 0;

}

int VulkanRasterizer::_update_uniform_buffer(uint32_t currentImage) {
  // Compute the elapsed time
  static auto startTime = std::chrono::high_resolution_clock::now();

  auto currentTime = std::chrono::high_resolution_clock::now();
  float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

  // Build the MVP matrix
  UniformBufferObject ubo = {};
  ubo.model = glm::rotate(glm::mat4(1.0f), time * glm::radians(10.0f), glm::vec3(0.0f, 0.0f, 1.0f));
  ubo.view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
  ubo.proj = glm::perspective(glm::radians(45.0f), swap_chain_extent.width / (float) swap_chain_extent.height, 0.1f, 10.0f);
  ubo.proj[1][1] *= -1; // GLM was originally designed for OpenGL, where the Y coordinate of the clip coordinates is inverted.

  // Copy the uniforms into the buffer
  void* data;
  vkMapMemory(device, uniform_buffers_memory[currentImage], 0, sizeof(ubo), 0, &data);
  memcpy(data, &ubo, sizeof(ubo));
  vkUnmapMemory(device, uniform_buffers_memory[currentImage]);

  return 0;
}

int VulkanRasterizer::_draw_frame() {
  vkWaitForFences(device, 1, &in_flight_fences[currentFrame], VK_TRUE, std::numeric_limits<uint64_t>::max());

  uint32_t imageIndex;
  VkResult result = vkAcquireNextImageKHR(device, swap_chain, std::numeric_limits<uint64_t>::max(), image_available_semaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);
  if (result == VK_ERROR_OUT_OF_DATE_KHR) {
    _recreate_swap_chain();
    return 0;
  } else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
    throw std::runtime_error("failed to acquire swap chain image!");
  }

  vkResetFences(device, 1, &in_flight_fences[currentFrame]);

  _update_uniform_buffer(imageIndex);

  VkSubmitInfo submitInfo = {};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.waitSemaphoreCount = 1;
  VkSemaphore waitSemaphores[] = {image_available_semaphores[currentFrame]};
  VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
  submitInfo.pWaitSemaphores = waitSemaphores;
  submitInfo.pWaitDstStageMask = waitStages;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &command_buffers[imageIndex];
  VkSemaphore signalSemaphores[] = {render_finished_semaphores[currentFrame]};
  submitInfo.signalSemaphoreCount = 1;
  submitInfo.pSignalSemaphores = signalSemaphores;

  if (vkQueueSubmit(graphics_queue, 1, &submitInfo, in_flight_fences[currentFrame]) != VK_SUCCESS) {
    throw std::runtime_error("failed to submit draw command buffer!");
  }

  VkPresentInfoKHR presentInfo = {};
  presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
  presentInfo.waitSemaphoreCount = 1;
  presentInfo.pWaitSemaphores = signalSemaphores;
  VkSwapchainKHR swap_chains[] = {swap_chain};
  presentInfo.swapchainCount = 1;
  presentInfo.pSwapchains = swap_chains;
  presentInfo.pImageIndices = &imageIndex;
  presentInfo.pResults = nullptr; // Optional

  result = vkQueuePresentKHR(present_queue, &presentInfo);
  if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebuffer_resized) {
    framebuffer_resized = false;
    _recreate_swap_chain();
  } else if (result != VK_SUCCESS) {
    throw std::runtime_error("failed to present swap chain image!");
  }

  currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
  return 0;
}


static void framebufferResizeCallback(GLFWwindow* window, int width, int height) {
  auto app = reinterpret_cast<VulkanRasterizer*>(glfwGetWindowUserPointer(window));
  app->notify_should_resize();
}

int VulkanRasterizer::init() {
  // >>>>>>>>>>>>>>>>>
  glfwInit();
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
  glfwSetWindowUserPointer(window, this);
  glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
  // <<<<<<<<<<<<<<<<<

  _create_instance();
  _enable_debug();
  _create_surface();
  _pick_physical_device();
  _create_logical_device();
  _create_swap_chain();
  _create_image_views();
  _create_render_pass();

  _create_descriptor_set_layout();

  _create_graphic_pipeline();
  _create_command_pool();
  _create_depth_resources();
  _create_frame_buffers();

  _create_texture_image();
  _create_texture_image_view();
  _create_texture_sampler();

  _load_model();
  _create_vertex_buffer();
  _create_index_buffer();
  _create_uniform_buffers();

  _create_descriptor_pool();
  _create_descriptor_sets();

  _create_command_buffers();
  _create_semaphores_and_fences();

  return 0;
};


int VulkanRasterizer::process() {
  while(!glfwWindowShouldClose(window)) {
    glfwPollEvents();
    _draw_frame();
  }

  vkDeviceWaitIdle(device);
  return 0;
};

int VulkanRasterizer::_cleanup_swap_chain() {

  vkDestroyImageView(device, depth_image_view, nullptr);
  vkDestroyImage(device, depth_image, nullptr);
  vkFreeMemory(device, depth_image_memory, nullptr);

  for (auto framebuffer : swap_chain_framebuffers) {
    vkDestroyFramebuffer(device, framebuffer, nullptr);
  }

  vkFreeCommandBuffers(device, command_pool, static_cast<uint32_t>(command_buffers.size()), command_buffers.data());

  vkDestroyPipeline(device, graphicsPipeline, nullptr);
  vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
  vkDestroyRenderPass(device, render_pass, nullptr);

  for (auto imageView : swap_chain_image_views) {
    vkDestroyImageView(device, imageView, nullptr);
  }
  vkDestroySwapchainKHR(device, swap_chain, nullptr);
  return 0;
}

int VulkanRasterizer::cleanup() {
  _cleanup_swap_chain();

  vkDestroySampler(device, texture_sampler, nullptr);
  vkDestroyImageView(device, texture_image_view, nullptr);
  vkDestroyImage(device, texture_image, nullptr);
  vkFreeMemory(device, texture_image_memory, nullptr);

  vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
  vkDestroyDescriptorSetLayout(device, descriptor_set_layout, nullptr);
  for (size_t i = 0; i < swap_chain_images.size(); i++) {
    vkDestroyBuffer(device, uniform_buffers[i], nullptr);
    vkFreeMemory(device, uniform_buffers_memory[i], nullptr);
  }
  vkDestroyBuffer(device, vertex_buffer, nullptr);
  vkFreeMemory(device, vertex_buffer_memory, nullptr);
  vkDestroyBuffer(device, index_buffer, nullptr);
  vkFreeMemory(device, index_buffer_memory, nullptr);

  for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
    vkDestroySemaphore(device, render_finished_semaphores[i], nullptr);
    vkDestroySemaphore(device, image_available_semaphores[i], nullptr);
    vkDestroyFence(device, in_flight_fences[i], nullptr);
  }

  vkDestroyCommandPool(device, command_pool, nullptr);

  vkDestroyDevice(device, nullptr);
  if (enable_validation) {
    _destroy_debug_report_callback_EXT(instance, callback, nullptr);
  }
  vkDestroySurfaceKHR(instance, surface, nullptr);
  vkDestroyInstance(instance, nullptr);

  // >>>>>>>>>>>>>>>>>
  glfwDestroyWindow(window);
  glfwTerminate();
  // <<<<<<<<<<<<<<<<<

  return 0;
};

int main(const int argc, const char *argv[]) {
  VulkanRasterizer *vulkan_rasterizer = new VulkanRasterizer(true);
  vulkan_rasterizer->init();
  vulkan_rasterizer->process();
  vulkan_rasterizer->cleanup();

  delete(vulkan_rasterizer);

	return 0;
}
