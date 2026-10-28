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
uniform float u_invCos;
uniform float u_exaggeration;
uniform float u_riseM;
out vec3 v_color;
out float v_across;
out float v_u;
flat out float v_kind;
flat out float v_surface;
void main() {
    vec2 p = a_center + a_normal * a_shape.x * u_halfWidthM;
    float z = a_shape.w * (a_shape.y * u_exaggeration + 1.0 + u_riseM * a_shape.z) * u_invCos;
    gl_Position = u_matrix * vec4(p, z, 1.0);
    v_color = a_color;
    v_across = a_shape.x;
    v_u = a_road.x;
    v_kind = a_road.y;
    v_surface = a_road.z;
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
out vec4 fragColor;

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
    if (kind == 3) {
        float t = clamp((abs(v_across) / 5.0 - 0.2) / 0.8, 0.0, 1.0);
        float alpha = 0.4 * (1.0 - t * t * (3.0 - 2.0 * t));
        fragColor = vec4(0.0, 0.0, 0.0, alpha);
        return;
    }
    if (kind == 0) {
        vec3 c = v_color;
        if (u_graphIntervalM > 0.0) {
            float k = floor(v_u / u_graphIntervalM + 0.5);
            float stripe = k * u_graphIntervalM;
            if (k > 0.0 && stripe < u_totalM && abs(v_u - stripe) < mpp) c = mix(c, vec3(1.0), 0.6);
        }
        fragColor = vec4(c, 1.0);
        return;
    }
    float a = abs(v_across);
    if (hw * 0.22 / mpp >= 0.5 && a > 0.78) {
        fragColor = vec4(0.078, 0.078, 0.078, 1.0);
        return;
    }
    int surface = int(v_surface + 0.5);
    vec3 c = road(surface, v_u, v_across * hw + hw, hw, mpp);
    bool isDashed = kind == 1 && (surface == 0 || surface == 2) && hw * 0.22 / mpp >= 0.5;
    if (isDashed && a < 0.11 && mod(v_u, 4.0 * hw) < 2.0 * hw) c = vec3(0.969);
    fragColor = vec4(c, 1.0);
}
)";

struct Params {
    double originX = 0;
    double originY = 0;
    double latitude = 0;
    double invCos = 1;
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
        uInvCos_ = glGetUniformLocation(program_, "u_invCos");
        uExaggeration_ = glGetUniformLocation(program_, "u_exaggeration");
        uRise_ = glGetUniformLocation(program_, "u_riseM");
        uMetersPerPx_ = glGetUniformLocation(program_, "u_metersPerPx");
        uGraphInterval_ = glGetUniformLocation(program_, "u_graphIntervalM");
        uTotal_ = glGetUniformLocation(program_, "u_totalM");
        uSurfaceColors_ = glGetUniformLocation(program_, "u_surfaceColors");
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
        // A new context has none of the old buffers.
        std::lock_guard<std::mutex> lock(state_->mutex);
        state_->isMeshDirty = true;
    }

    void render(const mbgl::style::CustomLayerRenderParameters& parameters) override {
        if (!program_) return;
        Params params;
        {
            std::lock_guard<std::mutex> lock(state_->mutex);
            if (state_->isMeshDirty) {
                upload(0, state_->shadow, shadowCount_);
                upload(1, state_->wall, wallCount_);
                state_->isMeshDirty = false;
            }
            params = state_->params;
        }
        if (wallCount_ == 0) return;

        // In world pixels at this zoom the origin's coordinates run to tens of
        // millions, too many for a float: the matrix takes the mesh from meters
        // around the origin to clip space in doubles, and only then is rounded.
        const double worldSize = kTileSize * std::pow(2.0, parameters.zoom);
        const double scale = worldSize / kEarthCircumferenceM;
        const double tx = params.originX * worldSize;
        const double ty = params.originY * worldSize;
        const auto& p = parameters.projectionMatrix;
        const double model[16] = {scale, 0, 0, 0, 0, scale, 0, 0, 0, 0, scale, 0, tx, ty, 0, 1};
        GLfloat matrix[16];
        for (int col = 0; col < 4; col++) {
            for (int row = 0; row < 4; row++) {
                double sum = 0;
                for (int i = 0; i < 4; i++) sum += p[i * 4 + row] * model[col * 4 + i];
                matrix[col * 4 + row] = static_cast<GLfloat>(sum);
            }
        }
        const double metersPerPx = kEarthCircumferenceM * std::cos(params.latitude * M_PI / 180.0) / worldSize;

        glUseProgram(program_);
        glUniformMatrix4fv(uMatrix_, 1, GL_FALSE, matrix);
        glUniform1f(uHalfWidth_, lookup(params.halfWidthByZoom, parameters.zoom, params.zoomStep));
        glUniform1f(uInvCos_, static_cast<float>(params.invCos));
        glUniform1f(uExaggeration_, static_cast<float>(params.exaggeration));
        glUniform1f(uRise_, lookup(params.riseByZoom, parameters.zoom, params.zoomStep));
        glUniform1f(uMetersPerPx_, static_cast<float>(metersPerPx));
        glUniform1f(uGraphInterval_, static_cast<float>(params.graphIntervalM));
        glUniform1f(uTotal_, static_cast<float>(params.totalM));
        glUniform3fv(uSurfaceColors_, 4, params.surfaceColors.data());
        glDisable(GL_STENCIL_TEST);
        glDisable(GL_CULL_FACE);

        // The ground shadow first, blended and under everything of the wall.
        glDisable(GL_DEPTH_TEST);
        glDepthMask(GL_FALSE);
        glEnable(GL_BLEND);
        glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
        glBindVertexArray(vaos_[0]);
        glDrawArrays(GL_TRIANGLES, 0, shadowCount_);

        // The map's own layers sit in the far end of the depth range: the wall
        // keeps to the near half, so it tests only against itself.
        glDisable(GL_BLEND);
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LEQUAL);
        glDepthMask(GL_TRUE);
        glDepthRangef(0.0f, 0.5f);
        glBindVertexArray(vaos_[1]);
        glDrawArrays(GL_TRIANGLES, 0, wallCount_);
        glBindVertexArray(0);
    }

    void contextLost() override { program_ = 0; }

    void deinitialize() override {
        if (!program_) return;
        glDeleteBuffers(2, buffers_);
        glDeleteVertexArrays(2, vaos_);
        glDeleteProgram(program_);
        glDeleteShader(vertexShader_);
        glDeleteShader(fragmentShader_);
        program_ = 0;
    }

private:
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
    GLint uMatrix_ = -1, uHalfWidth_ = -1, uInvCos_ = -1, uExaggeration_ = -1, uRise_ = -1;
    GLint uMetersPerPx_ = -1, uGraphInterval_ = -1, uTotal_ = -1, uSurfaceColors_ = -1;
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
    JNIEnv* env, jclass, jlong state, jdouble originX, jdouble originY, jdouble latitude, jdouble invCos,
    jdouble exaggeration, jdouble graphIntervalM, jdouble totalM, jdouble zoomStep, jfloatArray halfWidthByZoom,
    jfloatArray riseByZoom, jfloatArray surfaceColors) {
    Params params;
    params.originX = originX;
    params.originY = originY;
    params.latitude = latitude;
    params.invCos = invCos;
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

JNIEXPORT void JNICALL Java_org_maplibre_maplibregl_ClimbWallLayer_nativeReleaseState(JNIEnv*, jclass, jlong state) {
    delete reinterpret_cast<std::shared_ptr<State>*>(state);
}

} // extern "C"
