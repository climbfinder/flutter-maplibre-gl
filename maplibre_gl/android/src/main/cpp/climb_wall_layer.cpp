// Climbfinder fork: the 3D climb wall as a MapLibre custom layer, in OpenGL ES 3.
// The iOS twin is ClimbWallStyleLayer.swift; keep the shaders of both in step.
//
// The app sends one mesh (see its ClimbWallMesh for the vertex layout) in
// meters around an origin, and tables for what depends on the zoom. Every
// frame only the matrix and those uniforms change: no rebuild while the camera
// moves. The road's textures, center dashes, ink edges and the side stripes
// are drawn per pixel in the fragment shader, like the web's WebGL layer.

#include <GLES3/gl3.h>
#include <android/log.h>
#include <jni.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <mutex>
#include <vector>

#include <mbgl/style/layers/custom_layer_host.hpp>

namespace {

const char* kLogTag = "ClimbWallLayer";
constexpr double kEarthCircumferenceM = 40075016.686;
constexpr double kTileSize = 512.0;
constexpr int kFloatsPerVertex = 14;
constexpr GLsizei kStride = kFloatsPerVertex * sizeof(float);

const GLchar* kVertexShader = R"(#version 300 es
layout(location = 0) in vec2 a_center;
layout(location = 1) in vec2 a_normal;
layout(location = 2) in vec4 a_shape;    // across, elevation, fraction, isTop
layout(location = 3) in vec3 a_road;     // u, kind, surface
layout(location = 4) in vec3 a_color;
uniform mat4 u_matrix;
uniform float u_halfWidthM;
uniform float u_exaggeration;
uniform float u_riseM;
out vec3 v_color;
out float v_across;
out float v_u;
flat out float v_kind;
flat out float v_surface;
out vec3 v_world;
void main() {
    vec2 p = a_center + a_normal * a_shape.x * u_halfWidthM;
    float z = a_shape.w * (a_shape.y * u_exaggeration + 1.0 + u_riseM * a_shape.z);
    gl_Position = u_matrix * vec4(p, z, 1.0);
    v_color = a_color;
    v_across = a_shape.x;
    v_u = a_road.x;
    v_kind = a_road.y;
    v_surface = a_road.z;
    v_world = vec3(p, z);
}
)";

const GLchar* kFragmentShader = R"(#version 300 es
precision highp float;
precision highp int;
uniform float u_halfWidthM;
uniform float u_metersPerPx;
uniform float u_graphIntervalM;
uniform float u_totalM;
uniform vec3 u_surfaceColors[4];
in vec3 v_color;
in float v_across;
in float v_u;
flat in float v_kind;
flat in float v_surface;
in vec3 v_world;
out vec4 fragColor;

// How much of this pixel lies past `edge` along `x`, which changes by `w` per
// pixel: a soft step one pixel wide, so the details painted on the wall don't
// stair-step.
float coverage(float x, float edge, float w) {
    return clamp((x - edge) / max(w, 1e-6) + 0.5, 0.0, 1.0);
}

// The road's top lit by a low sun from the north-west (mercator y points
// south): stretches that climb towards it brighten, those that turn away
// darken, flat road stays as it was.
vec3 lit(vec3 c, vec3 world) {
    vec3 n = cross(dFdx(world), dFdy(world));
    float len = length(n);
    if (len < 1e-12) return c;
    n /= len;
    if (n.z < 0.0) n = -n;
    vec3 sun = normalize(vec3(-0.6, -0.6, 1.0));
    float shade = clamp(dot(n, sun) / sun.z, 0.7, 1.2);
    return min(c * mix(1.0, shade, 0.6), vec3(1.0));
}

float hash(int x, int y, int salt) {
    uint h = uint(x) * 374761393u + uint(y) * 668265263u + uint(salt) * 1442695041u;
    h = (h ^ (h >> 13u)) * 1274126177u;
    h ^= h >> 16u;
    return float(h & 0xffffffu) / 16777216.0;
}

vec3 road(int surface, float x, float y, float hw, float mpp) {
    vec3 c = u_surfaceColors[surface];
    if (surface == 3) {
        float cell = max(hw * 0.5, mpp * 4.0);
        float row = floor(y / cell);
        float shift = mod(row, 2.0) * 0.5;
        float fx = fract(x / cell + shift);
        float fy = fract(y / cell);
        float joint = max(0.07, 0.7 * mpp / cell);
        if (fx < joint || fx > 1.0 - joint || fy < joint || fy > 1.0 - joint) return c * 0.72;
        return c * (0.94 + 0.1 * hash(int(floor(x / cell + shift)), int(row), 7));
    }
    if (surface == 1) {
        float cell = max(hw * 0.12, mpp * 1.4);
        float h = hash(int(floor(x / cell)), int(floor(y / cell)), 0);
        if (h < 0.22) return c * 0.86;
        if (h > 0.9) return min(c * 1.06, vec3(1.0));
        return c;
    }
    if (surface == 2) {
        float cell = hw * 1.3;
        int kx = int(floor(x / cell));
        int ky = int(floor(y / cell));
        float shade = 1.0;
        for (int dx = -1; dx <= 1; dx++) {
            for (int dy = -1; dy <= 1; dy++) {
                int k = kx + dx;
                int r = ky + dy;
                if (hash(k, r, 0) <= 0.55) continue;
                vec2 center = vec2(float(k) + 0.5 + (hash(k, r, 3) - 0.5) * 0.6, float(r) + 0.5 + (hash(k, r, 9) - 0.5) * 0.6) * cell;
                float radius = (0.24 + 0.28 * hash(k, r, 5)) * hw;
                vec2 d = vec2(x, y) - center;
                float angle = atan(d.y, d.x);
                float wobble = 1.0 + 0.18 * sin(angle * 3.0 + hash(k, r, 11) * 6.2832) + 0.1 * sin(angle * 5.0 + hash(k, r, 17) * 6.2832);
                float t = length(d) / (radius * wobble);
                if (t < 0.8) shade = min(shade, 0.45);
                else if (t < 1.1) shade = min(shade, 0.88);
            }
        }
        return c * shade;
    }
    return c;
}

void main() {
    float hw = u_halfWidthM;
    float mpp = u_metersPerPx;
    int kind = int(v_kind + 0.5);
    if (kind == 0) {
        vec3 c = v_color;
        if (u_graphIntervalM > 0.0) {
            float k = floor(v_u / u_graphIntervalM + 0.5);
            float stripe = k * u_graphIntervalM;
            if (k > 0.0 && stripe < u_totalM) c = mix(c, vec3(1.0), 0.6 * coverage(-abs(v_u - stripe), -mpp, fwidth(v_u)));
        }
        fragColor = vec4(c, 1.0);
        return;
    }
    float a = abs(v_across);
    float wa = fwidth(a);
    int surface = int(v_surface + 0.5);
    vec3 c = road(surface, v_u, v_across * hw + hw, hw, mpp);
    bool isDashed = kind == 1 && (surface == 0 || surface == 2) && hw * 0.22 / mpp >= 0.5;
    if (isDashed) {
        // Measured along the road itself: the phase jumps at every dash.
        float wu = fwidth(v_u);
        float phase = mod(v_u, 4.0 * hw);
        float along = max(coverage(min(phase, 2.0 * hw - phase), 0.0, wu), coverage(phase, 4.0 * hw, wu));
        c = mix(c, vec3(0.969), along * (1.0 - coverage(a, 0.11, wa)));
    }
    if (hw * 0.22 / mpp >= 0.5) c = mix(c, vec3(0.078), coverage(a, 0.78, wa));
    fragColor = vec4(lit(c, v_world), 1.0);
}
)";

// Where the shadow's strips overlap (inside a bend) each pixel must be shaded
// once, by the darkest strip. Their interpolated depths differ by rounding,
// so the depth comes from the ground plane under the pixel instead, the same
// for every strip, nudged nearer by the quantized darkness: a first pass
// keeps the darkest per pixel in the depth buffer, a second draws only that.
// Every level stays behind the ground: nudged in front of it, the shadow hid
// the wall's foot, so the wall floated and its low start vanished.
const GLchar* kShadowFragmentShader = R"(#version 300 es
precision highp float;
uniform vec3 u_groundDepth;
uniform float u_depthStep;
uniform float u_depthNudge;
in float v_across;
out vec4 fragColor;
void main() {
    float t = clamp((abs(v_across) / 5.0 - 0.2) / 0.8, 0.0, 1.0);
    float alpha = 0.4 * (1.0 - t * t * (3.0 - 2.0 * t));
    float level = floor(alpha * 64.0 + 0.5);
    gl_FragDepth = dot(u_groundDepth, vec3(gl_FragCoord.xy, 1.0)) + (65.0 - level) * u_depthStep - u_depthNudge;
    fragColor = vec4(0.0, 0.0, 0.0, alpha);
}
)";

const GLchar* kLabelVertexShader = R"(#version 300 es
layout(location = 0) in vec4 a_quad;  // clip x, y, then u, v
out vec2 v_uv;
void main() {
    gl_Position = vec4(a_quad.xy, 0.0, 1.0);
    v_uv = a_quad.zw;
}
)";

const GLchar* kLabelFragmentShader = R"(#version 300 es
precision mediump float;
uniform sampler2D u_image;
in vec2 v_uv;
out vec4 fragColor;
void main() {
    fragColor = texture(u_image, v_uv);
}
)";

constexpr int kFloatsPerTop = 7;
constexpr int kLabelMeta = 11;
constexpr double kMinWallPx = 8.0;
constexpr double kOcclusionDepthMargin = 1.004;

/// One label image from the app, standing on the wall's top `distance` meters
/// along it; see the app's ClimbWallLabelSprite.
struct Label {
    std::vector<uint8_t> rgba;
    int pixelWidth = 0;
    int pixelHeight = 0;
    double pixelRatio = 1, anchorX = 0, anchorY = 0, pivotY = 0, halfWidth = 0, bodyHeight = 0;
    double x = 0, y = 0, elevation = 0, fraction = 0, distance = 0;
};

struct Placed {
    size_t index;
    // Two triangles, x y u v per corner, in clip space.
    float base[24];
    float body[24];
};

struct ScreenPoint {
    bool ok = false;
    double x = 0, y = 0, depth = 0;
};

/// Where this frame's labels go: the app's ClimbWallLabelsPainter (the web's
/// ClimbWallLabels) ported to run inside the map's own frame. A label hidden
/// behind the wall is left out; one a nearer label covers shrinks. The iOS
/// twin is ClimbWallLabelLayout in ClimbWallStyleLayer.swift.
class LabelLayout {
public:
    LabelLayout(const double* matrix, double width, double height, double halfWidthM, double exaggeration,
                double riseM, double metersPerPx, double maxElevation, const std::vector<float>& tops)
        : m_(matrix), width_(width), height_(height), halfWidthM_(halfWidthM), exaggeration_(exaggeration),
          riseM_(riseM), metersPerPx_(metersPerPx), maxElevation_(maxElevation), tops_(tops) {}

    std::vector<Placed> place(const std::vector<Label>& labels) {
        std::vector<Placed> placed;
        const double wallTop = maxElevation_ * exaggeration_ + 1 + riseM_;
        if (wallTop / metersPerPx_ < kMinWallPx) return placed;
        struct Item { size_t index; ScreenPoint at; double covered; };
        std::vector<Item> items;
        for (size_t i = 0; i < labels.size(); i++) {
            const Label& label = labels[i];
            const double z = label.elevation * exaggeration_ + 1 + riseM_ * label.fraction;
            ScreenPoint at = project(label.x, label.y, z);
            if (!at.ok || at.x < 0 || at.x > width_ || at.y < 0 || at.y > height_) continue;
            if (corners_.empty()) buildCorners();
            if (hides(label.distance, at)) continue;
            items.push_back({i, at, 0});
        }
        struct Box { double l, t, r, b; };
        auto box = [&](const Item& item) {
            const Label& label = labels[item.index];
            const double pillBottom = item.at.y - (label.anchorY - label.pivotY);
            return Box{item.at.x - label.halfWidth, pillBottom - label.bodyHeight, item.at.x + label.halfWidth, pillBottom};
        };
        for (auto& item : items) {
            const Box a = box(item);
            for (const auto& other : items) {
                if (&item == &other || other.at.depth > item.at.depth) continue;
                const Box b = box(other);
                const double w = std::min(a.r, b.r) - std::max(a.l, b.l);
                const double h = std::min(a.b, b.b) - std::max(a.t, b.t);
                if (w <= 0 || h <= 0) continue;
                const double smaller = std::min(std::max(1.0, (a.r - a.l) * (a.b - a.t)), std::max(1.0, (b.r - b.l) * (b.b - b.t)));
                item.covered = std::max(item.covered, w * h / smaller);
            }
        }
        std::sort(items.begin(), items.end(), [](const Item& a, const Item& b) { return a.at.depth > b.at.depth; });
        for (const auto& item : items) {
            const Label& label = labels[item.index];
            const double shrink = std::min(0.93, 0.93 * std::pow(item.covered, 0.35));
            const double scale = shrink > 0.02 ? 1 - shrink * 0.75 : 1;
            const double imageWidth = label.pixelWidth / label.pixelRatio;
            const double imageHeight = label.pixelHeight / label.pixelRatio;
            // On whole device pixels, so the text stays as sharp as it was drawn.
            auto snap = [&](double v) { return std::round(v * label.pixelRatio) / label.pixelRatio; };
            const double left = snap(item.at.x - label.anchorX);
            const double top = snap(item.at.y - label.anchorY);
            const double pivotX = left + label.anchorX;
            const double pivotY = top + label.pivotY;
            const double pivotV = label.pivotY / imageHeight;
            Placed p{item.index, {}, {}};
            quad(p.base, left, pivotY, left + imageWidth, top + imageHeight, pivotV, 1);
            quad(p.body, pivotX - label.anchorX * scale, pivotY - label.pivotY * scale,
                 pivotX + (imageWidth - label.anchorX) * scale, pivotY, 0, pivotV);
            placed.push_back(p);
        }
        return placed;
    }

private:
    void quad(float* out, double x0, double y0, double x1, double y1, double v0, double v1) const {
        const float l = static_cast<float>(x0 / width_ * 2 - 1), r = static_cast<float>(x1 / width_ * 2 - 1);
        const float t = static_cast<float>(1 - y0 / height_ * 2), b = static_cast<float>(1 - y1 / height_ * 2);
        const float vt = static_cast<float>(v0), vb = static_cast<float>(v1);
        const float values[24] = {l, t, 0, vt, r, t, 1, vt, r, b, 1, vb, l, t, 0, vt, r, b, 1, vb, l, b, 0, vb};
        std::copy(values, values + 24, out);
    }

    ScreenPoint project(double x, double y, double z) const {
        const double* m = m_;
        const double cx = m[0] * x + m[4] * y + m[8] * z + m[12];
        const double cy = m[1] * x + m[5] * y + m[9] * z + m[13];
        const double w = m[3] * x + m[7] * y + m[11] * z + m[15];
        if (w <= 1e-9) return {};
        return {true, (cx / w + 1) / 2 * width_, (1 - cy / w) / 2 * height_, w};
    }

    // Per top: the left and right foot, then the left and right top.
    void buildCorners() {
        const size_t count = tops_.size() / kFloatsPerTop;
        corners_.resize(count);
        for (size_t k = 0; k < count; k++) {
            const float* t = &tops_[k * kFloatsPerTop];
            const double nx = t[2] * halfWidthM_, ny = t[3] * halfWidthM_;
            const double z = t[4] * exaggeration_ + 1 + riseM_ * t[5];
            corners_[k] = {project(t[0] + nx, t[1] + ny, 0), project(t[0] - nx, t[1] - ny, 0),
                           project(t[0] + nx, t[1] + ny, z), project(t[0] - nx, t[1] - ny, z)};
        }
    }

    // The web's occludedAt: a side or the top of a stretch of wall that isn't
    // right next to the label stands in front of it.
    bool hides(double distance, const ScreenPoint& at) const {
        const int count = static_cast<int>(corners_.size());
        int near = 0;
        while (near < count - 2 && tops_[(near + 1) * kFloatsPerTop + 6] <= distance) near++;
        for (int i = 0; i < count - 1; i++) {
            if (i >= near - 1 && i <= near + 1) continue;
            const auto& a = corners_[i];
            const auto& b = corners_[i + 1];
            // Most of the wall lies nowhere near the label on screen.
            double minX = at.x, maxX = at.x, minY = at.y, maxY = at.y;
            bool isNearer = false;
            for (const auto* corner : {&a[0], &a[1], &a[2], &a[3], &b[0], &b[1], &b[2], &b[3]}) {
                if (!corner->ok) continue;
                minX = std::min(minX, corner->x);
                maxX = std::max(maxX, corner->x);
                minY = std::min(minY, corner->y);
                maxY = std::max(maxY, corner->y);
                isNearer = isNearer || corner->depth < at.depth;
            }
            if (!isNearer || minX == at.x || maxX == at.x || minY == at.y || maxY == at.y) continue;
            if (quadHides(at, a[0], b[0], b[2], a[2]) || quadHides(at, a[1], b[1], b[3], a[3]) ||
                quadHides(at, a[2], b[2], b[3], a[3])) {
                return true;
            }
        }
        return false;
    }

    static bool quadHides(const ScreenPoint& at, const ScreenPoint& p, const ScreenPoint& q, const ScreenPoint& r,
                          const ScreenPoint& t) {
        return triangleHides(at, p, q, r) || triangleHides(at, p, r, t);
    }

    static bool triangleHides(const ScreenPoint& at, const ScreenPoint& p, const ScreenPoint& q, const ScreenPoint& r) {
        if (!p.ok || !q.ok || !r.ok) return false;
        const double d1 = (q.x - p.x) * (at.y - p.y) - (q.y - p.y) * (at.x - p.x);
        const double d2 = (r.x - q.x) * (at.y - q.y) - (r.y - q.y) * (at.x - q.x);
        const double d3 = (p.x - r.x) * (at.y - r.y) - (p.y - r.y) * (at.x - r.x);
        const bool hasNegative = d1 < 0 || d2 < 0 || d3 < 0;
        const bool hasPositive = d1 > 0 || d2 > 0 || d3 > 0;
        if (hasNegative && hasPositive) return false;
        const double area = d1 + d2 + d3;
        if (std::abs(area) < 1e-9) return false;
        const double inverseDepth = (d2 / p.depth + d3 / q.depth + d1 / r.depth) / area;
        return inverseDepth > kOcclusionDepthMargin / at.depth;
    }

    const double* m_;
    double width_, height_, halfWidthM_, exaggeration_, riseM_, metersPerPx_, maxElevation_;
    const std::vector<float>& tops_;
    std::vector<std::array<ScreenPoint, 4>> corners_;
};

struct Params {
    double originX = 0;
    double originY = 0;
    double latitude = 0;
    double exaggeration = 1;
    double graphIntervalM = 0;
    double totalM = 0;
    double zoomStep = 0.1;
    std::vector<float> halfWidthByZoom;
    std::vector<float> riseByZoom;
    std::vector<float> surfaceColors = std::vector<float>(12, 0.5f);
};

struct State {
    std::mutex mutex;
    std::vector<float> wall;
    std::vector<float> shadow;
    bool isMeshDirty = false;
    Params params;
    std::vector<float> tops;
    double maxElevation = 0;
    std::vector<Label> labels;
    // Bumped whenever labels change, so the host remakes its textures.
    int labelsVersion = 0;
};

float lookup(const std::vector<float>& table, double zoom, double step) {
    if (table.empty()) return 0;
    double f = std::max(0.0, zoom / step);
    size_t i = std::min(static_cast<size_t>(f), table.size() - 1);
    if (i + 1 >= table.size()) return table.back();
    double t = f - static_cast<double>(i);
    return static_cast<float>(table[i] + (table[i + 1] - table[i]) * t);
}

GLuint compile(GLenum type, const GLchar* source) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);
    GLint ok = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[2048];
        glGetShaderInfoLog(shader, sizeof(log), nullptr, log);
        __android_log_print(ANDROID_LOG_ERROR, kLogTag, "Shader compile failed: %s", log);
    }
    return shader;
}

// The window depth of the ground (z = 0) as a plane over the framebuffer's
// pixels, depth = a·x + b·y + c, in the wall's near half of the depth range.
void groundDepth(const double* m, const GLint* viewport, GLfloat* out) {
    // Clip x, y, w of a ground point are A·(x, y, 1); its NDC z is then
    // linear in NDC x, y, with the coefficients rowZ · A⁻¹.
    const double a[9] = {m[0], m[4], m[12], m[1], m[5], m[13], m[3], m[7], m[15]};
    const double rowZ[3] = {m[2], m[6], m[14]};
    const double det = a[0] * (a[4] * a[8] - a[5] * a[7]) - a[1] * (a[3] * a[8] - a[5] * a[6]) +
                       a[2] * (a[3] * a[7] - a[4] * a[6]);
    if (std::abs(det) < 1e-30) {
        out[0] = out[1] = 0;
        out[2] = 0.5f;
        return;
    }
    const double inv[9] = {
        (a[4] * a[8] - a[5] * a[7]) / det, (a[2] * a[7] - a[1] * a[8]) / det, (a[1] * a[5] - a[2] * a[4]) / det,
        (a[5] * a[6] - a[3] * a[8]) / det, (a[0] * a[8] - a[2] * a[6]) / det, (a[2] * a[3] - a[0] * a[5]) / det,
        (a[3] * a[7] - a[4] * a[6]) / det, (a[1] * a[6] - a[0] * a[7]) / det, (a[0] * a[4] - a[1] * a[3]) / det,
    };
    double c[3];
    for (int j = 0; j < 3; j++) c[j] = rowZ[0] * inv[j] + rowZ[1] * inv[3 + j] + rowZ[2] * inv[6 + j];
    // NDC from pixels: 2 (x − vx) / w − 1; window depth: NDC z / 4 + 1/4.
    const double w = viewport[2], h = viewport[3];
    out[0] = static_cast<GLfloat>(0.5 * c[0] / w);
    out[1] = static_cast<GLfloat>(0.5 * c[1] / h);
    out[2] = static_cast<GLfloat>(0.25 * (c[2] - c[0] * (1 + 2 * viewport[0] / w) - c[1] * (1 + 2 * viewport[1] / h)) + 0.25);
}

class ClimbWallHost final : public mbgl::style::CustomLayerHost {
public:
    explicit ClimbWallHost(std::shared_ptr<State> state) : state_(std::move(state)) {}

    void initialize(const mbgl::style::CustomLayerInitParameters&) override {
        program_ = glCreateProgram();
        vertexShader_ = compile(GL_VERTEX_SHADER, kVertexShader);
        fragmentShader_ = compile(GL_FRAGMENT_SHADER, kFragmentShader);
        glAttachShader(program_, vertexShader_);
        glAttachShader(program_, fragmentShader_);
        glLinkProgram(program_);
        GLint ok = 0;
        glGetProgramiv(program_, GL_LINK_STATUS, &ok);
        if (!ok) {
            char log[2048];
            glGetProgramInfoLog(program_, sizeof(log), nullptr, log);
            __android_log_print(ANDROID_LOG_ERROR, kLogTag, "Program link failed: %s", log);
        }
        uMatrix_ = glGetUniformLocation(program_, "u_matrix");
        uHalfWidth_ = glGetUniformLocation(program_, "u_halfWidthM");
        uExaggeration_ = glGetUniformLocation(program_, "u_exaggeration");
        uRise_ = glGetUniformLocation(program_, "u_riseM");
        uMetersPerPx_ = glGetUniformLocation(program_, "u_metersPerPx");
        uGraphInterval_ = glGetUniformLocation(program_, "u_graphIntervalM");
        uTotal_ = glGetUniformLocation(program_, "u_totalM");
        uSurfaceColors_ = glGetUniformLocation(program_, "u_surfaceColors");
        shadowProgram_ = glCreateProgram();
        shadowFragmentShader_ = compile(GL_FRAGMENT_SHADER, kShadowFragmentShader);
        glAttachShader(shadowProgram_, vertexShader_);
        glAttachShader(shadowProgram_, shadowFragmentShader_);
        glLinkProgram(shadowProgram_);
        uShadowMatrix_ = glGetUniformLocation(shadowProgram_, "u_matrix");
        uShadowHalfWidth_ = glGetUniformLocation(shadowProgram_, "u_halfWidthM");
        uShadowExaggeration_ = glGetUniformLocation(shadowProgram_, "u_exaggeration");
        uShadowRise_ = glGetUniformLocation(shadowProgram_, "u_riseM");
        uGroundDepth_ = glGetUniformLocation(shadowProgram_, "u_groundDepth");
        uDepthStep_ = glGetUniformLocation(shadowProgram_, "u_depthStep");
        uDepthNudge_ = glGetUniformLocation(shadowProgram_, "u_depthNudge");
        // Steps of whole depth-buffer units, so neighbouring levels never round together.
        GLint depthBits = 0;
        glGetIntegerv(GL_DEPTH_BITS, &depthBits);
        depthUnit_ = static_cast<float>(std::ldexp(1.0, -(depthBits > 0 ? depthBits : 16)));
        glGenVertexArrays(2, vaos_);
        glGenBuffers(2, buffers_);
        for (int i = 0; i < 2; i++) {
            glBindVertexArray(vaos_[i]);
            glBindBuffer(GL_ARRAY_BUFFER, buffers_[i]);
            const struct { GLuint location; GLint size; int offset; } attributes[] = {
                {0, 2, 0}, {1, 2, 2}, {2, 4, 4}, {3, 3, 8}, {4, 3, 11},
            };
            for (const auto& attribute : attributes) {
                glEnableVertexAttribArray(attribute.location);
                glVertexAttribPointer(attribute.location, attribute.size, GL_FLOAT, GL_FALSE, kStride,
                                      reinterpret_cast<const void*>(attribute.offset * sizeof(float)));
            }
        }
        glBindVertexArray(0);

        labelProgram_ = glCreateProgram();
        labelVertexShader_ = compile(GL_VERTEX_SHADER, kLabelVertexShader);
        labelFragmentShader_ = compile(GL_FRAGMENT_SHADER, kLabelFragmentShader);
        glAttachShader(labelProgram_, labelVertexShader_);
        glAttachShader(labelProgram_, labelFragmentShader_);
        glLinkProgram(labelProgram_);
        uLabelImage_ = glGetUniformLocation(labelProgram_, "u_image");
        glGenVertexArrays(1, &labelVao_);
        glGenBuffers(1, &labelBuffer_);
        glBindVertexArray(labelVao_);
        glBindBuffer(GL_ARRAY_BUFFER, labelBuffer_);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), nullptr);
        glBindVertexArray(0);
        labelTextures_.clear();
        labelsVersion_ = -1;
        // A new context has none of the old buffers.
        std::lock_guard<std::mutex> lock(state_->mutex);
        state_->isMeshDirty = true;
    }

    void render(const mbgl::style::CustomLayerRenderParameters& parameters) override {
        if (!program_) return;
        Params params;
        std::lock_guard<std::mutex> lock(state_->mutex);
        if (state_->isMeshDirty) {
            upload(0, state_->shadow, shadowCount_);
            upload(1, state_->wall, wallCount_);
            state_->isMeshDirty = false;
        }
        params = state_->params;
        if (wallCount_ == 0) return;

        // In world pixels at this zoom the origin's coordinates run to tens of
        // millions, too many for a float: the matrix takes the mesh from meters
        // around the origin to clip space in doubles, and only then is rounded.
        // Heights stay in meters: MapLibre's matrix scales z by pixels per meter.
        const double worldSize = kTileSize * std::pow(2.0, parameters.zoom);
        const double scale = worldSize / kEarthCircumferenceM;
        const double tx = params.originX * worldSize;
        const double ty = params.originY * worldSize;
        const auto& p = parameters.projectionMatrix;
        const double model[16] = {scale, 0, 0, 0, 0, scale, 0, 0, 0, 0, 1, 0, tx, ty, 0, 1};
        double frame[16];
        GLfloat matrix[16];
        for (int col = 0; col < 4; col++) {
            for (int row = 0; row < 4; row++) {
                double sum = 0;
                for (int i = 0; i < 4; i++) sum += p[i * 4 + row] * model[col * 4 + i];
                frame[col * 4 + row] = sum;
                matrix[col * 4 + row] = static_cast<GLfloat>(sum);
            }
        }
        const double metersPerPx = kEarthCircumferenceM * std::cos(params.latitude * M_PI / 180.0) / worldSize;

        const float halfWidthM = lookup(params.halfWidthByZoom, parameters.zoom, params.zoomStep);
        const float riseM = lookup(params.riseByZoom, parameters.zoom, params.zoomStep);
        glDisable(GL_STENCIL_TEST);
        glDisable(GL_CULL_FACE);

        // The map's own layers sit in the far end of the depth range: the wall
        // keeps to the near half, so it tests only against itself.
        glDepthRangef(0.0f, 0.5f);
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LESS);
        glDepthMask(GL_TRUE);
        // The ground shadow first, under everything of the wall: the darkest
        // strip's depth per pixel, then only that strip's color.
        if (shadowCount_ > 0) {
            GLint viewport[4];
            glGetIntegerv(GL_VIEWPORT, viewport);
            GLfloat ground[3];
            groundDepth(frame, viewport, ground);
            glUseProgram(shadowProgram_);
            glUniformMatrix4fv(uShadowMatrix_, 1, GL_FALSE, matrix);
            glUniform1f(uShadowHalfWidth_, halfWidthM);
            glUniform1f(uShadowExaggeration_, static_cast<float>(params.exaggeration));
            glUniform1f(uShadowRise_, riseM);
            glUniform3fv(uGroundDepth_, 1, ground);
            glUniform1f(uDepthStep_, 4 * depthUnit_);
            glBindVertexArray(vaos_[0]);
            glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
            glUniform1f(uDepthNudge_, 0);
            glDrawArrays(GL_TRIANGLES, 0, shadowCount_);
            glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
            glEnable(GL_BLEND);
            glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
            glUniform1f(uDepthNudge_, 2 * depthUnit_);
            glDrawArrays(GL_TRIANGLES, 0, shadowCount_);
        }

        glUseProgram(program_);
        glUniformMatrix4fv(uMatrix_, 1, GL_FALSE, matrix);
        glUniform1f(uHalfWidth_, halfWidthM);
        glUniform1f(uExaggeration_, static_cast<float>(params.exaggeration));
        glUniform1f(uRise_, riseM);
        glUniform1f(uMetersPerPx_, static_cast<float>(metersPerPx));
        glUniform1f(uGraphInterval_, static_cast<float>(params.graphIntervalM));
        glUniform1f(uTotal_, static_cast<float>(params.totalM));
        glUniform3fv(uSurfaceColors_, 4, params.surfaceColors.data());
        glDisable(GL_BLEND);
        glDepthFunc(GL_LEQUAL);
        glBindVertexArray(vaos_[1]);
        glDrawArrays(GL_TRIANGLES, 0, wallCount_);
        glBindVertexArray(0);

        LabelLayout layout(frame, parameters.width, parameters.height, halfWidthM, params.exaggeration, riseM,
                           metersPerPx, state_->maxElevation, state_->tops);
        drawLabels(layout.place(state_->labels));
    }

    void contextLost() override {
        program_ = 0;
        labelTextures_.clear();
    }

    void deinitialize() override {
        if (!program_) return;
        glDeleteBuffers(2, buffers_);
        glDeleteVertexArrays(2, vaos_);
        glDeleteProgram(program_);
        glDeleteProgram(shadowProgram_);
        glDeleteShader(shadowFragmentShader_);
        glDeleteShader(vertexShader_);
        glDeleteShader(fragmentShader_);
        if (!labelTextures_.empty()) glDeleteTextures(static_cast<GLsizei>(labelTextures_.size()), labelTextures_.data());
        labelTextures_.clear();
        glDeleteBuffers(1, &labelBuffer_);
        glDeleteVertexArrays(1, &labelVao_);
        glDeleteProgram(labelProgram_);
        glDeleteShader(labelVertexShader_);
        glDeleteShader(labelFragmentShader_);
        program_ = 0;
    }

private:
    // The labels stand on the wall in this very frame: drawn by the app over
    // the map, they trailed it by a frame or two. Called with the state locked.
    void drawLabels(const std::vector<Placed>& placed) {
        const auto& labels = state_->labels;
        if (labelsVersion_ != state_->labelsVersion) {
            if (!labelTextures_.empty()) glDeleteTextures(static_cast<GLsizei>(labelTextures_.size()), labelTextures_.data());
            labelTextures_.assign(labels.size(), 0);
            glGenTextures(static_cast<GLsizei>(labelTextures_.size()), labelTextures_.data());
            glActiveTexture(GL_TEXTURE0);
            for (size_t i = 0; i < labels.size(); i++) {
                glBindTexture(GL_TEXTURE_2D, labelTextures_[i]);
                glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, labels[i].pixelWidth, labels[i].pixelHeight, 0, GL_RGBA,
                             GL_UNSIGNED_BYTE, labels[i].rgba.data());
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            }
            labelsVersion_ = state_->labelsVersion;
        }
        if (placed.empty()) return;
        glUseProgram(labelProgram_);
        glUniform1i(uLabelImage_, 0);
        glActiveTexture(GL_TEXTURE0);
        glDisable(GL_DEPTH_TEST);
        glDepthMask(GL_FALSE);
        glEnable(GL_BLEND);
        glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
        // Every stem under every label, like the web: all the stems' quads,
        // then all the bodies', in one upload.
        std::vector<float> quads;
        quads.reserve(placed.size() * 48);
        for (const auto& item : placed) quads.insert(quads.end(), item.base, item.base + 24);
        for (const auto& item : placed) quads.insert(quads.end(), item.body, item.body + 24);
        glBindVertexArray(labelVao_);
        glBindBuffer(GL_ARRAY_BUFFER, labelBuffer_);
        glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(quads.size() * sizeof(float)), quads.data(), GL_STREAM_DRAW);
        for (size_t i = 0; i < placed.size() * 2; i++) {
            glBindTexture(GL_TEXTURE_2D, labelTextures_[placed[i % placed.size()].index]);
            glDrawArrays(GL_TRIANGLES, static_cast<GLint>(i * 6), 6);
        }
        glBindVertexArray(0);
    }

    void upload(int index, const std::vector<float>& data, GLsizei& count) {
        glBindBuffer(GL_ARRAY_BUFFER, buffers_[index]);
        glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(data.size() * sizeof(float)), data.data(), GL_STATIC_DRAW);
        count = static_cast<GLsizei>(data.size() / kFloatsPerVertex);
    }

    std::shared_ptr<State> state_;
    GLuint program_ = 0;
    GLuint vertexShader_ = 0;
    GLuint fragmentShader_ = 0;
    GLuint vaos_[2] = {0, 0};
    GLuint buffers_[2] = {0, 0};
    GLsizei wallCount_ = 0;
    GLsizei shadowCount_ = 0;
    GLint uMatrix_ = -1, uHalfWidth_ = -1, uExaggeration_ = -1, uRise_ = -1;
    GLint uMetersPerPx_ = -1, uGraphInterval_ = -1, uTotal_ = -1, uSurfaceColors_ = -1;
    GLuint shadowProgram_ = 0, shadowFragmentShader_ = 0;
    GLint uShadowMatrix_ = -1, uShadowHalfWidth_ = -1, uShadowExaggeration_ = -1, uShadowRise_ = -1;
    GLint uGroundDepth_ = -1, uDepthStep_ = -1, uDepthNudge_ = -1;
    float depthUnit_ = 1.0f / 65536.0f;
    GLuint labelProgram_ = 0, labelVertexShader_ = 0, labelFragmentShader_ = 0;
    GLuint labelVao_ = 0, labelBuffer_ = 0;
    GLint uLabelImage_ = -1;
    std::vector<GLuint> labelTextures_;
    int labelsVersion_ = -1;
};

std::shared_ptr<State>& handle(jlong state) { return *reinterpret_cast<std::shared_ptr<State>*>(state); }

std::vector<float> floats(JNIEnv* env, jfloatArray array) {
    if (array == nullptr) return {};
    const jsize length = env->GetArrayLength(array);
    std::vector<float> values(static_cast<size_t>(length));
    env->GetFloatArrayRegion(array, 0, length, values.data());
    return values;
}

} // namespace

extern "C" {

JNIEXPORT jlong JNICALL Java_org_maplibre_maplibregl_ClimbWallLayer_nativeCreateState(JNIEnv*, jclass) {
    return reinterpret_cast<jlong>(new std::shared_ptr<State>(std::make_shared<State>()));
}

// MapLibre's CustomLayer takes ownership of the host.
JNIEXPORT jlong JNICALL Java_org_maplibre_maplibregl_ClimbWallLayer_nativeCreateHost(JNIEnv*, jclass, jlong state) {
    auto* host = new ClimbWallHost(handle(state));
    return reinterpret_cast<jlong>(static_cast<mbgl::style::CustomLayerHost*>(host));
}

JNIEXPORT void JNICALL Java_org_maplibre_maplibregl_ClimbWallLayer_nativeSetMesh(JNIEnv* env, jclass, jlong state,
                                                                               jfloatArray wall, jfloatArray shadow) {
    auto wallData = floats(env, wall);
    auto shadowData = floats(env, shadow);
    auto& s = handle(state);
    std::lock_guard<std::mutex> lock(s->mutex);
    s->wall = std::move(wallData);
    s->shadow = std::move(shadowData);
    s->isMeshDirty = true;
}

JNIEXPORT void JNICALL Java_org_maplibre_maplibregl_ClimbWallLayer_nativeSetParams(
    JNIEnv* env, jclass, jlong state, jdouble originX, jdouble originY, jdouble latitude,
    jdouble exaggeration, jdouble graphIntervalM, jdouble totalM, jdouble zoomStep, jfloatArray halfWidthByZoom,
    jfloatArray riseByZoom, jfloatArray surfaceColors) {
    Params params;
    params.originX = originX;
    params.originY = originY;
    params.latitude = latitude;
    params.exaggeration = exaggeration;
    params.graphIntervalM = graphIntervalM;
    params.totalM = totalM;
    params.zoomStep = zoomStep > 0 ? zoomStep : 0.1;
    params.halfWidthByZoom = floats(env, halfWidthByZoom);
    params.riseByZoom = floats(env, riseByZoom);
    auto colors = floats(env, surfaceColors);
    if (colors.size() == 12) params.surfaceColors = std::move(colors);
    auto& s = handle(state);
    std::lock_guard<std::mutex> lock(s->mutex);
    s->params = std::move(params);
}

JNIEXPORT void JNICALL Java_org_maplibre_maplibregl_ClimbWallLayer_nativeSetTops(JNIEnv* env, jclass, jlong state,
                                                                               jfloatArray tops, jdouble maxElevation) {
    auto values = floats(env, tops);
    auto& s = handle(state);
    std::lock_guard<std::mutex> lock(s->mutex);
    s->tops = std::move(values);
    s->maxElevation = maxElevation;
}

JNIEXPORT void JNICALL Java_org_maplibre_maplibregl_ClimbWallLayer_nativeSetLabels(JNIEnv* env, jclass, jlong state,
                                                                                 jobjectArray images, jintArray sizes,
                                                                                 jdoubleArray meta) {
    const jsize count = env->GetArrayLength(images);
    std::vector<jint> sizeValues(static_cast<size_t>(count * 2));
    env->GetIntArrayRegion(sizes, 0, count * 2, sizeValues.data());
    std::vector<jdouble> metaValues(static_cast<size_t>(count * kLabelMeta));
    env->GetDoubleArrayRegion(meta, 0, count * kLabelMeta, metaValues.data());
    std::vector<Label> labels;
    for (jsize i = 0; i < count; i++) {
        auto image = static_cast<jbyteArray>(env->GetObjectArrayElement(images, i));
        Label label;
        label.pixelWidth = sizeValues[i * 2];
        label.pixelHeight = sizeValues[i * 2 + 1];
        const jsize length = image ? env->GetArrayLength(image) : 0;
        if (length != label.pixelWidth * label.pixelHeight * 4) {
            if (image) env->DeleteLocalRef(image);
            continue;
        }
        label.rgba.resize(static_cast<size_t>(length));
        env->GetByteArrayRegion(image, 0, length, reinterpret_cast<jbyte*>(label.rgba.data()));
        env->DeleteLocalRef(image);
        const jdouble* m = &metaValues[i * kLabelMeta];
        label.pixelRatio = m[0];
        label.anchorX = m[1];
        label.anchorY = m[2];
        label.pivotY = m[3];
        label.halfWidth = m[4];
        label.bodyHeight = m[5];
        label.x = m[6];
        label.y = m[7];
        label.elevation = m[8];
        label.fraction = m[9];
        label.distance = m[10];
        labels.push_back(std::move(label));
    }
    auto& s = handle(state);
    std::lock_guard<std::mutex> lock(s->mutex);
    s->labels = std::move(labels);
    s->labelsVersion++;
}

JNIEXPORT void JNICALL Java_org_maplibre_maplibregl_ClimbWallLayer_nativeReleaseState(JNIEnv*, jclass, jlong state) {
    delete reinterpret_cast<std::shared_ptr<State>*>(state);
}

} // extern "C"
