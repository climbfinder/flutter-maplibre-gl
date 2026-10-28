#pragma once

// Climbfinder fork: the custom layer interface of MapLibre Native android-v13.5.0
// (include/mbgl/style/layers/custom_layer_host.hpp and the parameter structs it
// pulls in), vendored because the AAR ships no headers. The host is handed to
// libmaplibre.so as a raw pointer and called through its vtable, so the order
// of the virtual functions and the layout of CustomLayerRenderParameters must
// stay exactly those of the MapLibre version in build.gradle.

#include <array>

namespace mbgl {

namespace gfx {
class Context;
} // namespace gfx

namespace style {

struct CustomLayerInitParameters {
    virtual ~CustomLayerInitParameters() = default;
};

struct CustomLayerRenderParameters {
    double width;
    double height;
    double latitude;
    double longitude;
    double zoom;
    double bearing;
    double pitch;
    double fieldOfView;
    std::array<double, 16> projectionMatrix;
    std::array<double, 16> nearClippedProjectionMatrix;
};

class CustomLayerHost {
public:
    virtual ~CustomLayerHost() = default;
    virtual void initialize(const CustomLayerInitParameters&) = 0;
    virtual void preRender(const mbgl::gfx::Context&, const CustomLayerRenderParameters&) {}
    virtual void render(const CustomLayerRenderParameters&) = 0;
    virtual void contextLost() = 0;
    virtual void deinitialize() = 0;
};

} // namespace style
} // namespace mbgl
