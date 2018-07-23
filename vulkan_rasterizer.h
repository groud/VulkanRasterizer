#ifndef VULKAN_RASTERIZER_H
#define VULKAN_RASTERIZER_H

#include <vector>
#include <array>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLM_LANG_STL11_FORCED
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>
#include <glm/glm.hpp>


class VulkanRasterizer {

  public:
    struct Vertex {
      glm::vec3 pos;
      glm::vec3 color;
      glm::vec2 texCoord;

      static VkVertexInputBindingDescription getBindingDescription() {
        VkVertexInputBindingDescription bindingDescription = {};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(Vertex);
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        return bindingDescription;
      }

      static std::array<VkVertexInputAttributeDescription, 3> getAttributeDescriptions() {
        std::array<VkVertexInputAttributeDescription, 3> attributeDescriptions = {};
        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[0].offset = offsetof(Vertex, pos);

        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[1].offset = offsetof(Vertex, color);

        attributeDescriptions[2].binding = 0;
        attributeDescriptions[2].location = 2;
        attributeDescriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
        attributeDescriptions[2].offset = offsetof(Vertex, texCoord);

        return attributeDescriptions;
      }

      bool operator==(const Vertex& other) const {
        return pos == other.pos && color == other.color && texCoord == other.texCoord;
      }
    };

  private:
    struct QueueFamilyIndices {
      int graphics_family = -1;
      int present_family = -1;

      bool is_complete() {
        return graphics_family >= 0 && present_family >= 0;
      }
    };

    struct SwapChainSupportDetails {
      VkSurfaceCapabilitiesKHR capabilities;
      std::vector<VkSurfaceFormatKHR> formats;
      std::vector<VkPresentModeKHR> presentModes;
    };

    struct UniformBufferObject {
      glm::mat4 model;
      glm::mat4 view;
      glm::mat4 proj;
    };

    const int MAX_FRAMES_IN_FLIGHT = 2;

    bool enable_validation;

    GLFWwindow * window;

    VkInstance instance = NULL;
    VkPhysicalDevice physical_device = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkSurfaceKHR surface;

    VkSwapchainKHR swap_chain;
    std::vector<VkImage> swap_chain_images;
    VkFormat swap_chain_image_format;
    VkExtent2D swap_chain_extent;
    std::vector<VkImageView> swap_chain_image_views;
    std::vector<VkFramebuffer> swap_chain_framebuffers;

    VkQueue graphics_queue;
    VkQueue present_queue;

    VkRenderPass render_pass;

    VkDescriptorSetLayout descriptor_set_layout;

    VkPipelineLayout pipeline_layout;

    VkPipeline graphicsPipeline;

    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;

    VkBuffer vertex_buffer;
    VkDeviceMemory vertex_buffer_memory;
    VkBuffer index_buffer;
    VkDeviceMemory index_buffer_memory;
    std::vector<VkBuffer> uniform_buffers;
    std::vector<VkDeviceMemory> uniform_buffers_memory;
    VkDescriptorPool descriptor_pool;
    std::vector<VkDescriptorSet> descriptor_sets;

    uint32_t mip_levels;
    VkImage texture_image;
    VkDeviceMemory texture_image_memory;
    VkImageView texture_image_view;
    VkSampler texture_sampler;

    VkImage depth_image;
    VkDeviceMemory depth_image_memory;
    VkImageView depth_image_view;

    VkCommandPool command_pool;
    std::vector<VkCommandBuffer> command_buffers;

    std::vector<VkSemaphore> image_available_semaphores;
    std::vector<VkSemaphore> render_finished_semaphores;
    std::vector<VkFence> in_flight_fences;
    size_t currentFrame = 0;
    bool framebuffer_resized = false;

    // Validation layer
    bool _check_validation_layer_support(std::vector<const char*> validationLayers);

    // Debug callback
    VkResult _create_debug_report_callback_EXT(VkInstance instance, const VkDebugReportCallbackCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugReportCallbackEXT* pCallback);
    void _destroy_debug_report_callback_EXT(VkInstance instance, VkDebugReportCallbackEXT callback, const VkAllocationCallbacks* pAllocator);
    static VKAPI_ATTR VkBool32 VKAPI_CALL _debug_callback(VkDebugReportFlagsEXT flags, VkDebugReportObjectTypeEXT objType, uint64_t obj, size_t location, int32_t code, const char* layerPrefix, const char* msg, void* userData);

    std::vector<const char*> _get_required_extensions(bool enable_validation);

    // Initialization
    int _create_instance();
    int _enable_debug();

    // Create surface
    int _create_surface();

    // Physical device selection
    struct QueueFamilyIndices _pick_queue_families(VkPhysicalDevice p_physical_device);
    bool _check_device_extension_support(VkPhysicalDevice p_physical_device, const std::vector<const char*> &deviceExtensions);
    int _get_physical_device_score(VkPhysicalDevice p_physical_device);
    int _pick_physical_device();

    // Create logical device
    int _create_logical_device();

    // Create swap chain
    struct SwapChainSupportDetails _query_swap_chain_support(VkPhysicalDevice p_physical_device);
    VkSurfaceFormatKHR _pick_swap_surface_format(const std::vector<VkSurfaceFormatKHR>& availableFormats);
    VkPresentModeKHR _pick_swap_present_mode(const std::vector<VkPresentModeKHR> availablePresentModes);
    VkExtent2D _pick_swap_extend(const VkSurfaceCapabilitiesKHR& capabilities);
    int _create_swap_chain();
    int _recreate_swap_chain();
    int _cleanup_swap_chain();

    // Create image views
    int _create_image_views();

    // Create graphic pipeline
    int _create_render_pass();
    VkShaderModule _create_shader_module(const std::vector<char>& code);
    int _create_descriptor_set_layout();
    int _create_graphic_pipeline();

    // Helpers
    VkCommandBuffer _begin_single_time_commands();
    void _end_single_time_commands(VkCommandBuffer command_buffer);
    VkImageView _create_image_view(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags, uint32_t mip_levels);

    // Buffers operations
    uint32_t _find_memory_type(uint32_t typeFilter, VkMemoryPropertyFlags properties);
    int _create_buffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& buffer_memory);
    void _copy_buffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size);
    void _copy_buffer_to_image(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height);
    int _transition_image_layout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout, uint32_t mip_levels);
    int _create_image(uint32_t width, uint32_t height, uint32_t mip_levels, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory);
    int _create_mipmaps(VkImage image, VkFormat format, int32_t texWidth, int32_t texHeight, uint32_t mipLevels);

    int _create_texture_image();
    int _create_texture_image_view();
    int _create_texture_sampler();
    int _load_model();
    int _create_vertex_buffer();
    int _create_index_buffer();
    int _create_uniform_buffers();
    int _create_descriptor_pool();
    int _create_descriptor_sets();

    // Depth buffer
    VkFormat _find_supported_format(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features);
    VkFormat _find_depth_format();
    bool _has_stencil_component(VkFormat format);
    int _create_depth_resources();

    // Commands
    int _create_frame_buffers();
    int _create_command_pool();
    int _create_command_buffers();

    int _create_semaphores_and_fences();

    // Drawing
    int _update_uniform_buffer(uint32_t currentImage);
    int _draw_frame();

  public:
    void notify_should_resize() {framebuffer_resized = true;};

    int init();
    int process();
    int cleanup();

    VulkanRasterizer(bool p_enable_validation) : enable_validation(p_enable_validation) {};
};
namespace std {
  template<> struct hash<VulkanRasterizer::Vertex> {
    size_t operator()(VulkanRasterizer::Vertex const& vertex) const {
      return ((hash<glm::vec3>()(vertex.pos) ^
            (hash<glm::vec3>()(vertex.color) << 1)) >> 1) ^
        (hash<glm::vec2>()(vertex.texCoord) << 1);
    }
  };
}
#endif
