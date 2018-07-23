VULKAN_SDK_PATH=/home/gilles/Vulkan/vulkansdk-linux-x86_64-1.1.77.0/1.1.77.0/x86_64

CFLAGS=-std=c++11 -O3 -I$(VULKAN_SDK_PATH)/include -Iincludes -g
LDFLAGS=-L $(VULKAN_SDK_PATH)/lib `pkg-config --static --libs glfw3` -lvulkan

all: VulkanRasterizer

VulkanRasterizer: vulkan_rasterizer.cpp
	g++ $(CFLAGS) -o VulkanRasterizer vulkan_rasterizer.cpp $(LDFLAGS)

ClangTidy: vulkan_rasterizer.cpp
	clang vulkan_rasterizer.cpp $(CFLAGS) -o VulkanRasterizer $(LDFLAGS)

.PHONY: test clean

test: VulkanRasterizer
	LD_LIBRARY_PATH=$(VULKAN_SDK_PATH)/lib VK_LAYER_PATH=$(VULKAN_SDK_PATH)/etc/explicit_layer.d ./VulkanRasterizer

shader:
	cd shaders; \
	$(VULKAN_SDK_PATH)/bin/glslangValidator -V shader.vert;\
	$(VULKAN_SDK_PATH)/bin/glslangValidator -V shader.frag

clean:
	rm -f VulkanRasterizer
