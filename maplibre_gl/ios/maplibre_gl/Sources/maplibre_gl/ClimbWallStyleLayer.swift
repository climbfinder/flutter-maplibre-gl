import Flutter
import MapLibre
import MetalKit
import simd

/// Climbfinder fork: the 3D climb wall as a MapLibre custom layer, in Metal.
/// The Android twin is `climb_wall_layer.cpp`; keep the shaders of both in step.
///
/// The app sends one mesh (see its `ClimbWallMesh` for the vertex layout) in
/// meters around an origin, and tables for what depends on the zoom. Every
/// frame only the matrix and those uniforms change: no rebuild while the camera
/// moves. The road's textures, center dashes, ink edges and the side stripes
/// are drawn per pixel in the fragment shader, like the web's WebGL layer.
final class ClimbWallStyleLayer: MLNCustomStyleLayer {
    private static let earthCircumferenceM = 40075016.686
    private static let tileSize = 512.0
    private static let floatsPerVertex = 14

    /// Laid out like `Uniforms` in the shader.
    private struct Uniforms {
        var matrix: simd_float4x4
        var halfWidthM: Float
        var metersPerPx: Float
        var exaggeration: Float
        var riseM: Float
        var graphIntervalM: Float
        var totalM: Float
        var pad0: Float = 0
        var pad1: Float = 0
        var surface0: SIMD4<Float>
        var surface1: SIMD4<Float>
        var surface2: SIMD4<Float>
        var surface3: SIMD4<Float>
    }

    /// See `shadowFragment`.
    private struct ShadowUniforms {
        var groundDepth: SIMD4<Float>
        var depthStep: Float
        var depthNudge: Float
        var pad0: Float = 0
        var pad1: Float = 0
    }

    private var device: MTLDevice?
    private weak var view: MTKView?
    private var wallPipeline: MTLRenderPipelineState?
    private var shadowPipeline: MTLRenderPipelineState?
    private var shadowDepthPipeline: MTLRenderPipelineState?
    private var wallDepth: MTLDepthStencilState?
    private var shadowDepth: MTLDepthStencilState?
    private var wallBuffer: MTLBuffer?
    private var shadowBuffer: MTLBuffer?
    private var wallCount = 0
    private var shadowCount = 0
    private var pendingWall: Data?
    private var pendingShadow: Data?

    private var originX = 0.0
    private var originY = 0.0
    private var latitude = 0.0
    private var exaggeration = 1.0
    private var graphIntervalM = 0.0
    private var totalM = 0.0
    private var zoomStep = 0.1
    private var halfWidthByZoom: [Float] = []
    private var riseByZoom: [Float] = []
    private var surfaceColors: [SIMD4<Float>] = Array(repeating: SIMD4(0.5, 0.5, 0.5, 1), count: 4)

    /// The wall's top at every cut, `ClimbWallMesh.floatsPerTop` floats each.
    private var tops: [Float] = []
    private var maxElevation = 0.0
    private var labels: [ClimbWallLabel] = []
    private var labelPipeline: MTLRenderPipelineState?
    private var labelDepth: MTLDepthStencilState?
    private var labelSampler: MTLSamplerState?

    func update(wall: Data?, shadow: Data?, params: [String: Any]) {
        if let wall = wall, let shadow = shadow {
            pendingWall = wall
            pendingShadow = shadow
        }
        func number(_ key: String) -> Double? { (params[key] as? NSNumber)?.doubleValue }
        func floats(_ key: String) -> [Float]? {
            guard let typed = params[key] as? FlutterStandardTypedData else { return nil }
            return typed.data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        }
        originX = number("originX") ?? originX
        originY = number("originY") ?? originY
        latitude = number("latitude") ?? latitude
        exaggeration = number("exaggeration") ?? exaggeration
        graphIntervalM = number("graphIntervalM") ?? graphIntervalM
        totalM = number("totalM") ?? totalM
        zoomStep = number("zoomStep") ?? zoomStep
        halfWidthByZoom = floats("halfWidthByZoom") ?? halfWidthByZoom
        riseByZoom = floats("riseByZoom") ?? riseByZoom
        if let colors = floats("surfaceColors"), colors.count == 12 {
            surfaceColors = (0..<4).map { SIMD4(colors[$0 * 3], colors[$0 * 3 + 1], colors[$0 * 3 + 2], 1) }
        }
        tops = floats("tops") ?? tops
        maxElevation = number("maxElevation") ?? maxElevation
        if let list = params["labels"] as? [[String: Any]] {
            labels = list.compactMap(ClimbWallLabel.init)
        }
        setNeedsDisplay()
    }

    override func didMove(to mapView: MLNMapView) {
        let resource = mapView.backendResource()
        guard let device = resource.device, let view = resource.mtkView else { return }
        self.device = device
        self.view = view
        let library: MTLLibrary
        do {
            library = try device.makeLibrary(source: Self.shaderSource, options: nil)
        } catch {
            NSLog("ClimbWallStyleLayer: shader compile failed: \(error)")
            return
        }

        func pipeline(fragment: String, isBlended: Bool, writesColor: Bool = true) -> MTLRenderPipelineState? {
            let descriptor = MTLRenderPipelineDescriptor()
            descriptor.vertexFunction = library.makeFunction(name: "wallVertex")
            descriptor.fragmentFunction = library.makeFunction(name: fragment)
            descriptor.colorAttachments[0].pixelFormat = view.colorPixelFormat
            // MapLibre's own render pass, as in its CustomStyleLayerExample.
            descriptor.depthAttachmentPixelFormat = .depth32Float_stencil8
            descriptor.stencilAttachmentPixelFormat = .depth32Float_stencil8
            if isBlended {
                let color = descriptor.colorAttachments[0]!
                color.isBlendingEnabled = true
                color.sourceRGBBlendFactor = .one
                color.sourceAlphaBlendFactor = .one
                color.destinationRGBBlendFactor = .oneMinusSourceAlpha
                color.destinationAlphaBlendFactor = .oneMinusSourceAlpha
            }
            if !writesColor {
                descriptor.colorAttachments[0].writeMask = []
            }
            do {
                return try device.makeRenderPipelineState(descriptor: descriptor)
            } catch {
                NSLog("ClimbWallStyleLayer: pipeline failed: \(error)")
                return nil
            }
        }
        wallPipeline = pipeline(fragment: "wallFragment", isBlended: false)
        shadowPipeline = pipeline(fragment: "shadowFragment", isBlended: true)
        shadowDepthPipeline = pipeline(fragment: "shadowFragment", isBlended: false, writesColor: false)

        let labelDescriptor = MTLRenderPipelineDescriptor()
        labelDescriptor.vertexFunction = library.makeFunction(name: "labelVertex")
        labelDescriptor.fragmentFunction = library.makeFunction(name: "labelFragment")
        labelDescriptor.colorAttachments[0].pixelFormat = view.colorPixelFormat
        labelDescriptor.depthAttachmentPixelFormat = .depth32Float_stencil8
        labelDescriptor.stencilAttachmentPixelFormat = .depth32Float_stencil8
        let labelColor = labelDescriptor.colorAttachments[0]!
        labelColor.isBlendingEnabled = true
        labelColor.sourceRGBBlendFactor = .one
        labelColor.sourceAlphaBlendFactor = .one
        labelColor.destinationRGBBlendFactor = .oneMinusSourceAlpha
        labelColor.destinationAlphaBlendFactor = .oneMinusSourceAlpha
        labelPipeline = try? device.makeRenderPipelineState(descriptor: labelDescriptor)
        let samplerDescriptor = MTLSamplerDescriptor()
        samplerDescriptor.minFilter = .linear
        samplerDescriptor.magFilter = .linear
        labelSampler = device.makeSamplerState(descriptor: samplerDescriptor)

        let wallDescriptor = MTLDepthStencilDescriptor()
        wallDescriptor.depthCompareFunction = .lessEqual
        wallDescriptor.isDepthWriteEnabled = true
        wallDepth = device.makeDepthStencilState(descriptor: wallDescriptor)
        let shadowDescriptor = MTLDepthStencilDescriptor()
        shadowDescriptor.depthCompareFunction = .less
        shadowDescriptor.isDepthWriteEnabled = true
        shadowDepth = device.makeDepthStencilState(descriptor: shadowDescriptor)
        let labelDepthDescriptor = MTLDepthStencilDescriptor()
        labelDepthDescriptor.depthCompareFunction = .always
        labelDepthDescriptor.isDepthWriteEnabled = false
        labelDepth = device.makeDepthStencilState(descriptor: labelDepthDescriptor)
    }

    override func willMove(from _: MLNMapView) {
        wallBuffer = nil
        shadowBuffer = nil
        wallPipeline = nil
        shadowPipeline = nil
        shadowDepthPipeline = nil
        labelPipeline = nil
        for label in labels {
            label.texture = nil
        }
    }

    override func draw(in _: MLNMapView, with context: MLNStyleLayerDrawingContext) {
        guard let encoder = renderEncoder, let device = device,
              let wallPipeline = wallPipeline, let shadowPipeline = shadowPipeline,
              let shadowDepthPipeline = shadowDepthPipeline
        else { return }
        if let wall = pendingWall, let shadow = pendingShadow {
            wallBuffer = wall.isEmpty ? nil : wall.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: wall.count) }
            shadowBuffer = shadow.isEmpty ? nil : shadow.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: shadow.count) }
            wallCount = wall.count / (MemoryLayout<Float>.size * Self.floatsPerVertex)
            shadowCount = shadow.count / (MemoryLayout<Float>.size * Self.floatsPerVertex)
            pendingWall = nil
            pendingShadow = nil
        }
        guard let wallBuffer = wallBuffer, wallCount > 0 else { return }

        let matrix = frameMatrix(context)
        var uniforms = makeUniforms(context, matrix: matrix)
        let length = MemoryLayout<Uniforms>.stride
        encoder.setCullMode(.none)
        // The ground shadow first, under everything of the wall: the darkest
        // strip's depth per pixel, then only that strip's color.
        if let shadowBuffer = shadowBuffer, shadowCount > 0 {
            let size = view?.drawableSize ?? CGSize(width: context.size.width, height: context.size.height)
            var shadow = ShadowUniforms(
                groundDepth: Self.groundDepth(matrix, width: Double(size.width), height: Double(size.height)),
                depthStep: Self.depthStep,
                depthNudge: 0
            )
            encoder.setDepthStencilState(shadowDepth)
            encoder.setVertexBuffer(shadowBuffer, offset: 0, index: 0)
            encoder.setVertexBytes(&uniforms, length: length, index: 1)
            for (pipeline, nudge) in [(shadowDepthPipeline, Float(0)), (shadowPipeline, Self.depthStep / 2)] {
                shadow.depthNudge = nudge
                encoder.setRenderPipelineState(pipeline)
                encoder.setFragmentBytes(&shadow, length: MemoryLayout<ShadowUniforms>.stride, index: 2)
                encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: shadowCount)
            }
        }
        encoder.setRenderPipelineState(wallPipeline)
        encoder.setDepthStencilState(wallDepth)
        encoder.setVertexBuffer(wallBuffer, offset: 0, index: 0)
        encoder.setVertexBytes(&uniforms, length: length, index: 1)
        encoder.setFragmentBytes(&uniforms, length: length, index: 1)
        encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: wallCount)
        drawLabels(encoder, device: device, matrix: matrix, context: context, uniforms: uniforms)
    }

    /// The labels stand on the wall in this very frame: drawn by the app over
    /// the map, they trailed it by a frame or two.
    private func drawLabels(
        _ encoder: MTLRenderCommandEncoder,
        device: MTLDevice,
        matrix: [Double],
        context: MLNStyleLayerDrawingContext,
        uniforms: Uniforms
    ) {
        guard let pipeline = labelPipeline, !labels.isEmpty else { return }
        let layout = ClimbWallLabelLayout(
            matrix: matrix,
            width: Double(context.size.width),
            height: Double(context.size.height),
            halfWidthM: Double(uniforms.halfWidthM),
            exaggeration: exaggeration,
            riseM: Double(uniforms.riseM),
            metersPerPx: Double(uniforms.metersPerPx),
            maxElevation: maxElevation,
            tops: tops
        )
        let placed = layout.place(labels)
        guard !placed.isEmpty else { return }
        encoder.setRenderPipelineState(pipeline)
        encoder.setDepthStencilState(labelDepth)
        encoder.setFragmentSamplerState(labelSampler, index: 0)
        // Every stem under every label, like the web.
        for isBody in [false, true] {
            for item in placed {
                guard let texture = item.label.texture(device) else { continue }
                var quad = isBody ? item.body : item.base
                encoder.setVertexBytes(&quad, length: MemoryLayout<Float>.size * quad.count, index: 0)
                encoder.setFragmentTexture(texture, index: 0)
                encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6)
            }
        }
    }

    /// In world pixels at this zoom the origin's coordinates run to tens of
    /// millions, too many for a float: the matrix takes the mesh from meters
    /// around the origin to clip space in doubles, and only then is rounded.
    /// Heights stay in meters: MapLibre's matrix scales z by pixels per meter.
    private func frameMatrix(_ context: MLNStyleLayerDrawingContext) -> [Double] {
        let worldSize = Self.tileSize * pow(2.0, context.zoomLevel)
        let scale = worldSize / Self.earthCircumferenceM
        let m = context.projectionMatrix
        let p: [Double] = [
            m.m00, m.m01, m.m02, m.m03, m.m10, m.m11, m.m12, m.m13,
            m.m20, m.m21, m.m22, m.m23, m.m30, m.m31, m.m32, m.m33,
        ]
        let model: [Double] = [scale, 0, 0, 0, 0, scale, 0, 0, 0, 0, 1, 0, originX * worldSize, originY * worldSize, 0, 1]
        var result = [Double](repeating: 0, count: 16)
        for col in 0..<4 {
            for row in 0..<4 {
                var sum = 0.0
                for i in 0..<4 { sum += p[i * 4 + row] * model[col * 4 + i] }
                result[col * 4 + row] = sum
            }
        }
        return result
    }

    /// Whole steps of the 32-bit float depth buffer near the ground's depth
    /// (0.25–0.5, one unit 2⁻²⁵), so neighbouring levels never round together.
    private static let depthStep = Float(pow(2.0, -22.0))

    /// The window depth of the ground (z = 0) as a plane over the drawable's
    /// pixels, depth = a·x + b·y + c, in the wall's near half of the depth range.
    private static func groundDepth(_ m: [Double], width: Double, height: Double) -> SIMD4<Float> {
        // Clip x, y, w of a ground point are A·(x, y, 1); its NDC z is then
        // linear in NDC x, y, with the coefficients rowZ · A⁻¹.
        let a = [m[0], m[4], m[12], m[1], m[5], m[13], m[3], m[7], m[15]]
        let rowZ = [m[2], m[6], m[14]]
        let det = a[0] * (a[4] * a[8] - a[5] * a[7]) - a[1] * (a[3] * a[8] - a[5] * a[6])
            + a[2] * (a[3] * a[7] - a[4] * a[6])
        guard abs(det) > 1e-30, width > 0, height > 0 else { return SIMD4(0, 0, 0.5, 0) }
        let inv = [
            (a[4] * a[8] - a[5] * a[7]) / det, (a[2] * a[7] - a[1] * a[8]) / det, (a[1] * a[5] - a[2] * a[4]) / det,
            (a[5] * a[6] - a[3] * a[8]) / det, (a[0] * a[8] - a[2] * a[6]) / det, (a[2] * a[3] - a[0] * a[5]) / det,
            (a[3] * a[7] - a[4] * a[6]) / det, (a[1] * a[6] - a[0] * a[7]) / det, (a[0] * a[4] - a[1] * a[3]) / det,
        ]
        let c = (0..<3).map { j in rowZ[0] * inv[j] + rowZ[1] * inv[3 + j] + rowZ[2] * inv[6 + j] }
        // NDC from pixels (y down): x 2x/w − 1, y 1 − 2y/h; window depth: NDC z / 4 + 1/4.
        return SIMD4(
            Float(0.5 * c[0] / width),
            Float(-0.5 * c[1] / height),
            Float(0.25 * (c[2] - c[0] + c[1]) + 0.25),
            0
        )
    }

    private func makeUniforms(_ context: MLNStyleLayerDrawingContext, matrix: [Double]) -> Uniforms {
        let zoom = context.zoomLevel
        let worldSize = Self.tileSize * pow(2.0, zoom)
        let columns = (0..<4).map { col in SIMD4<Float>((0..<4).map { Float(matrix[col * 4 + $0]) }) }
        let metersPerPx = Self.earthCircumferenceM * cos(latitude * .pi / 180) / worldSize
        return Uniforms(
            matrix: simd_float4x4(columns: (columns[0], columns[1], columns[2], columns[3])),
            halfWidthM: lookup(halfWidthByZoom, zoom),
            metersPerPx: Float(metersPerPx),
            exaggeration: Float(exaggeration),
            riseM: lookup(riseByZoom, zoom),
            graphIntervalM: Float(graphIntervalM),
            totalM: Float(totalM),
            surface0: surfaceColors[0],
            surface1: surfaceColors[1],
            surface2: surfaceColors[2],
            surface3: surfaceColors[3]
        )
    }

    private func lookup(_ table: [Float], _ zoom: Double) -> Float {
        guard !table.isEmpty else { return 0 }
        let f = max(0, zoom / zoomStep)
        let i = min(Int(f), table.count - 1)
        guard i + 1 < table.count else { return table[table.count - 1] }
        return table[i] + (table[i + 1] - table[i]) * Float(f - Double(i))
    }

    private static let shaderSource = """
    #include <metal_stdlib>
    using namespace metal;

    struct Vertex {
        packed_float2 center;
        packed_float2 normal;
        float across;
        float elevation;
        float fraction;
        float isTop;
        float u;
        float kind;
        float surface;
        packed_float3 color;
    };

    struct Uniforms {
        float4x4 matrix;
        float halfWidthM;
        float metersPerPx;
        float exaggeration;
        float riseM;
        float graphIntervalM;
        float totalM;
        float pad0;
        float pad1;
        float4 surfaceColors[4];
    };

    struct Out {
        float4 position [[position]];
        float3 color;
        float across;
        float u;
        float kind [[flat]];
        float surface [[flat]];
        float3 world;
    };

    vertex Out wallVertex(uint vid [[vertex_id]],
                          const device Vertex* vertices [[buffer(0)]],
                          constant Uniforms& U [[buffer(1)]]) {
        Vertex v = vertices[vid];
        float2 p = float2(v.center) + float2(v.normal) * v.across * U.halfWidthM;
        float z = v.isTop * (v.elevation * U.exaggeration + 1.0 + U.riseM * v.fraction);
        float4 position = U.matrix * float4(p, z, 1.0);
        // The map's own layers sit in the far end of the depth range: the
        // wall keeps to the near half, so it tests only against itself.
        position.z = (position.z + position.w) * 0.25;
        Out out;
        out.position = position;
        out.color = float3(v.color);
        out.across = v.across;
        out.u = v.u;
        out.kind = v.kind;
        out.surface = v.surface;
        out.world = float3(p, z);
        return out;
    }

    float hash(int x, int y, int salt) {
        uint h = uint(x) * 374761393u + uint(y) * 668265263u + uint(salt) * 1442695041u;
        h = (h ^ (h >> 13)) * 1274126177u;
        h ^= h >> 16;
        return float(h & 0xffffffu) / 16777216.0;
    }

    float3 road(int surface, float x, float y, float hw, float mpp, constant Uniforms& U) {
        float3 c = U.surfaceColors[surface].rgb;
        if (surface == 3) {
            float cell = max(hw * 0.5, mpp * 4.0);
            float row = floor(y / cell);
            float shift = fmod(row, 2.0) * 0.5;
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
            if (h > 0.9) return min(c * 1.06, float3(1.0));
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
                    float2 center = float2(float(k) + 0.5 + (hash(k, r, 3) - 0.5) * 0.6,
                                           float(r) + 0.5 + (hash(k, r, 9) - 0.5) * 0.6) * cell;
                    float radius = (0.24 + 0.28 * hash(k, r, 5)) * hw;
                    float2 d = float2(x, y) - center;
                    float angle = atan2(d.y, d.x);
                    float wobble = 1.0 + 0.18 * sin(angle * 3.0 + hash(k, r, 11) * 6.2832)
                                       + 0.1 * sin(angle * 5.0 + hash(k, r, 17) * 6.2832);
                    float t = length(d) / (radius * wobble);
                    if (t < 0.8) shade = min(shade, 0.45);
                    else if (t < 1.1) shade = min(shade, 0.88);
                }
            }
            return c * shade;
        }
        return c;
    }

    // How much of this pixel lies past `edge` along `x`, which changes by `w`
    // per pixel: a soft step one pixel wide, so the details painted on the
    // wall don't stair-step (the map's Metal view has no multisampling).
    float coverage(float x, float edge, float w) {
        return clamp((x - edge) / max(w, 1e-6) + 0.5, 0.0, 1.0);
    }

    // The road's top lit by a low sun from the north-west (mercator y points
    // south): stretches that climb towards it brighten, those that turn away
    // darken, flat road stays as it was.
    float3 lit(float3 c, float3 world) {
        float3 n = cross(dfdx(world), dfdy(world));
        float len = length(n);
        if (len < 1e-12) return c;
        n /= len;
        if (n.z < 0.0) n = -n;
        const float3 sun = normalize(float3(-0.6, -0.6, 1.0));
        float shade = clamp(dot(n, sun) / sun.z, 0.7, 1.2);
        return min(c * mix(1.0, shade, 0.6), float3(1.0));
    }

    fragment float4 wallFragment(Out in [[stage_in]], constant Uniforms& U [[buffer(1)]]) {
        float hw = U.halfWidthM;
        float mpp = U.metersPerPx;
        int kind = int(in.kind + 0.5);
        if (kind == 0) {
            float3 c = in.color;
            if (U.graphIntervalM > 0.0) {
                float k = floor(in.u / U.graphIntervalM + 0.5);
                float stripe = k * U.graphIntervalM;
                if (k > 0.0 && stripe < U.totalM) c = mix(c, float3(1.0), 0.6 * coverage(-abs(in.u - stripe), -mpp, fwidth(in.u)));
            }
            return float4(c, 1.0);
        }
        float a = abs(in.across);
        float wa = fwidth(a);
        int surface = int(in.surface + 0.5);
        float3 c = road(surface, in.u, in.across * hw + hw, hw, mpp, U);
        bool isDashed = kind == 1 && (surface == 0 || surface == 2) && hw * 0.22 / mpp >= 0.5;
        if (isDashed) {
            // Measured along the road itself: the phase jumps at every dash.
            float wu = fwidth(in.u);
            float phase = fmod(in.u, 4.0 * hw);
            float along = max(coverage(min(phase, 2.0 * hw - phase), 0.0, wu), coverage(phase, 4.0 * hw, wu));
            c = mix(c, float3(0.969), along * (1.0 - coverage(a, 0.11, wa)));
        }
        if (hw * 0.22 / mpp >= 0.5) c = mix(c, float3(0.078), coverage(a, 0.78, wa));
        return float4(lit(c, in.world), 1.0);
    }

    struct LabelOut {
        float4 position [[position]];
        float2 uv;
    };

    vertex LabelOut labelVertex(uint vid [[vertex_id]], const device packed_float4* quad [[buffer(0)]]) {
        float4 v = float4(quad[vid]);
        LabelOut out;
        out.position = float4(v.xy, 0.0, 1.0);
        out.uv = v.zw;
        return out;
    }

    fragment float4 labelFragment(LabelOut in [[stage_in]], texture2d<float> image [[texture(0)]], sampler s [[sampler(0)]]) {
        return image.sample(s, in.uv);
    }

    struct ShadowUniforms {
        float4 groundDepth;
        float depthStep;
        float depthNudge;
        float pad0;
        float pad1;
    };

    struct ShadowOut {
        float4 color [[color(0)]];
        float depth [[depth(any)]];
    };

    // Where the shadow's strips overlap (inside a bend) each pixel must be
    // shaded once, by the darkest strip. Their interpolated depths differ by
    // rounding, so the depth comes from the ground plane under the pixel
    // instead, the same for every strip, nudged nearer by the quantized
    // darkness: a first pass keeps the darkest per pixel in the depth buffer,
    // a second draws only that. Every level stays behind the ground: nudged
    // in front of it, the shadow hid the wall's foot, so the wall floated and
    // its low start vanished.
    fragment ShadowOut shadowFragment(Out in [[stage_in]], constant ShadowUniforms& S [[buffer(2)]]) {
        float t = clamp((abs(in.across) / 5.0 - 0.2) / 0.8, 0.0, 1.0);
        float alpha = 0.4 * (1.0 - t * t * (3.0 - 2.0 * t));
        float level = floor(alpha * 64.0 + 0.5);
        ShadowOut out;
        out.depth = dot(S.groundDepth.xyz, float3(in.position.xy, 1.0)) + (65.0 - level) * S.depthStep - S.depthNudge;
        out.color = float4(0.0, 0.0, 0.0, alpha);
        return out;
    }
    """
}

/// One label image from the app, standing on the wall's top
/// `distance` meters along it; see the app's `ClimbWallLabelSprite`.
final class ClimbWallLabel {
    let rgba: Data
    let pixelWidth: Int
    let pixelHeight: Int
    let pixelRatio: Double
    let anchorX: Double
    let anchorY: Double
    let pivotY: Double
    let halfWidth: Double
    let bodyHeight: Double
    let x: Double
    let y: Double
    let elevation: Double
    let fraction: Double
    let distance: Double
    var texture: MTLTexture?

    init?(_ map: [String: Any]) {
        func number(_ key: String) -> Double? { (map[key] as? NSNumber)?.doubleValue }
        guard let rgba = (map["rgba"] as? FlutterStandardTypedData)?.data,
              let width = number("width"), let height = number("height"), let pixelRatio = number("pixelRatio"),
              let anchorX = number("anchorX"), let anchorY = number("anchorY"), let pivotY = number("pivotY"),
              let halfWidth = number("halfWidth"), let bodyHeight = number("bodyHeight"),
              let x = number("x"), let y = number("y"), let elevation = number("elevation"),
              let fraction = number("fraction"), let distance = number("distance"),
              rgba.count == Int(width) * Int(height) * 4
        else { return nil }
        self.rgba = rgba
        pixelWidth = Int(width)
        pixelHeight = Int(height)
        self.pixelRatio = pixelRatio
        self.anchorX = anchorX
        self.anchorY = anchorY
        self.pivotY = pivotY
        self.halfWidth = halfWidth
        self.bodyHeight = bodyHeight
        self.x = x
        self.y = y
        self.elevation = elevation
        self.fraction = fraction
        self.distance = distance
    }

    func texture(_ device: MTLDevice) -> MTLTexture? {
        if let texture = texture { return texture }
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm, width: pixelWidth, height: pixelHeight, mipmapped: false
        )
        guard let made = device.makeTexture(descriptor: descriptor) else { return nil }
        rgba.withUnsafeBytes {
            made.replace(
                region: MTLRegionMake2D(0, 0, pixelWidth, pixelHeight),
                mipmapLevel: 0,
                withBytes: $0.baseAddress!,
                bytesPerRow: pixelWidth * 4
            )
        }
        texture = made
        return made
    }
}

/// Where this frame's labels go: the app's `ClimbWallLabelsPainter` (the
/// web's `ClimbWallLabels`) ported to run inside the map's own frame. A label
/// hidden behind the wall is left out; one a nearer label covers shrinks. The
/// Android twin is `LabelLayout` in `climb_wall_layer.cpp`.
struct ClimbWallLabelLayout {
    static let floatsPerTop = 7
    static let minWallPx = 8.0
    static let occlusionDepthMargin = 1.004

    struct Placed {
        let label: ClimbWallLabel
        /// Two triangles, x y u v per corner, in clip space.
        var base: [Float]
        var body: [Float]
    }

    /// A point on screen; `isValid` is false behind the camera.
    private struct Point {
        var x = 0.0
        var y = 0.0
        var depth = 0.0
        var isValid = false
    }

    let matrix: [Double]
    let width: Double
    let height: Double
    let halfWidthM: Double
    let exaggeration: Double
    let riseM: Double
    let metersPerPx: Double
    let maxElevation: Double
    let tops: [Float]

    func place(_ labels: [ClimbWallLabel]) -> [Placed] {
        let wallTop = maxElevation * exaggeration + 1 + riseM
        guard wallTop / metersPerPx >= Self.minWallPx else { return [] }
        var corners: ContiguousArray<Point>?
        var items: [(label: ClimbWallLabel, at: Point, covered: Double)] = []
        for label in labels {
            let z = label.elevation * exaggeration + 1 + riseM * label.fraction
            let at = project(label.x, label.y, z)
            guard at.isValid, at.x >= 0, at.x <= width, at.y >= 0, at.y <= height else { continue }
            if corners == nil { corners = buildCorners() }
            if hides(corners!, distance: label.distance, at: at) { continue }
            items.append((label, at, 0))
        }
        func box(_ item: (label: ClimbWallLabel, at: Point, covered: Double)) -> CGRect {
            let pillBottom = item.at.y - (item.label.anchorY - item.label.pivotY)
            return CGRect(
                x: item.at.x - item.label.halfWidth, y: pillBottom - item.label.bodyHeight,
                width: 2 * item.label.halfWidth, height: item.label.bodyHeight
            )
        }
        for i in items.indices {
            let a = box(items[i])
            for j in items.indices where i != j && items[j].at.depth <= items[i].at.depth {
                let overlap = a.intersection(box(items[j]))
                guard !overlap.isNull, overlap.width > 0, overlap.height > 0 else { continue }
                let b = box(items[j])
                let smaller = min(max(1, a.width * a.height), max(1, b.width * b.height))
                items[i].covered = max(items[i].covered, Double(overlap.width * overlap.height / smaller))
            }
        }
        items.sort { $0.at.depth > $1.at.depth }
        return items.map { item in
            let label = item.label
            let shrink = min(0.93, 0.93 * pow(item.covered, 0.35))
            let scale = shrink > 0.02 ? 1 - shrink * 0.75 : 1
            let imageWidth = Double(label.pixelWidth) / label.pixelRatio
            let imageHeight = Double(label.pixelHeight) / label.pixelRatio
            // On whole device pixels, so the text stays as sharp as it was drawn.
            func snap(_ v: Double) -> Double { (v * label.pixelRatio).rounded() / label.pixelRatio }
            let left = snap(item.at.x - label.anchorX)
            let top = snap(item.at.y - label.anchorY)
            let pivotX = left + label.anchorX
            let pivotY = top + label.pivotY
            let pivotV = label.pivotY / imageHeight
            let base = quad(left, pivotY, left + imageWidth, top + imageHeight, v0: pivotV, v1: 1)
            let body = quad(
                pivotX - label.anchorX * scale, pivotY - label.pivotY * scale,
                pivotX + (imageWidth - label.anchorX) * scale, pivotY,
                v0: 0, v1: pivotV
            )
            return Placed(label: label, base: base, body: body)
        }
    }

    private func quad(_ x0: Double, _ y0: Double, _ x1: Double, _ y1: Double, v0: Double, v1: Double) -> [Float] {
        func cx(_ x: Double) -> Float { Float(x / width * 2 - 1) }
        func cy(_ y: Double) -> Float { Float(1 - y / height * 2) }
        let (l, r, t, b) = (cx(x0), cx(x1), cy(y0), cy(y1))
        let (vt, vb) = (Float(v0), Float(v1))
        return [l, t, 0, vt, r, t, 1, vt, r, b, 1, vb, l, t, 0, vt, r, b, 1, vb, l, b, 0, vb]
    }

    private func project(_ x: Double, _ y: Double, _ z: Double) -> Point {
        let m = matrix
        let cx = m[0] * x + m[4] * y + m[8] * z + m[12]
        let cy = m[1] * x + m[5] * y + m[9] * z + m[13]
        let w = m[3] * x + m[7] * y + m[11] * z + m[15]
        guard w > 1e-9 else { return Point() }
        return Point(x: (cx / w + 1) / 2 * width, y: (1 - cy / w) / 2 * height, depth: w, isValid: true)
    }

    /// Four per top: the left and right foot, then the left and right top.
    private func buildCorners() -> ContiguousArray<Point> {
        let count = tops.count / Self.floatsPerTop
        var corners = ContiguousArray<Point>()
        corners.reserveCapacity(count * 4)
        for k in 0..<count {
            let t = k * Self.floatsPerTop
            let cx = Double(tops[t]), cy = Double(tops[t + 1])
            let nx = Double(tops[t + 2]) * halfWidthM, ny = Double(tops[t + 3]) * halfWidthM
            let z = Double(tops[t + 4]) * exaggeration + 1 + riseM * Double(tops[t + 5])
            corners.append(project(cx + nx, cy + ny, 0))
            corners.append(project(cx - nx, cy - ny, 0))
            corners.append(project(cx + nx, cy + ny, z))
            corners.append(project(cx - nx, cy - ny, z))
        }
        return corners
    }

    /// The web's `occludedAt`: a side or the top of a stretch of wall that
    /// isn't right next to the label stands in front of it.
    private func hides(_ corners: ContiguousArray<Point>, distance: Double, at: Point) -> Bool {
        let count = corners.count / 4
        var near = 0
        while near < count - 2, Double(tops[(near + 1) * Self.floatsPerTop + 6]) <= distance {
            near += 1
        }
        for i in 0..<max(0, count - 1) where i < near - 1 || i > near + 1 {
            let a = i * 4, b = (i + 1) * 4
            // Most of the wall lies nowhere near the label on screen.
            var minX = at.x, maxX = at.x, minY = at.y, maxY = at.y
            var isNearer = false
            for c in a..<(b + 4) where corners[c].isValid {
                let corner = corners[c]
                minX = min(minX, corner.x)
                maxX = max(maxX, corner.x)
                minY = min(minY, corner.y)
                maxY = max(maxY, corner.y)
                isNearer = isNearer || corner.depth < at.depth
            }
            if !isNearer || minX == at.x || maxX == at.x || minY == at.y || maxY == at.y { continue }
            if quadHides(at, corners[a], corners[b], corners[b + 2], corners[a + 2])
                || quadHides(at, corners[a + 1], corners[b + 1], corners[b + 3], corners[a + 3])
                || quadHides(at, corners[a + 2], corners[b + 2], corners[b + 3], corners[a + 3])
            {
                return true
            }
        }
        return false
    }

    private func quadHides(_ at: Point, _ p: Point, _ q: Point, _ r: Point, _ t: Point) -> Bool {
        triangleHides(at, p, q, r) || triangleHides(at, p, r, t)
    }

    private func triangleHides(_ at: Point, _ p: Point, _ q: Point, _ r: Point) -> Bool {
        guard p.isValid, q.isValid, r.isValid else { return false }
        let d1 = (q.x - p.x) * (at.y - p.y) - (q.y - p.y) * (at.x - p.x)
        let d2 = (r.x - q.x) * (at.y - q.y) - (r.y - q.y) * (at.x - q.x)
        let d3 = (p.x - r.x) * (at.y - r.y) - (p.y - r.y) * (at.x - r.x)
        let hasNegative = d1 < 0 || d2 < 0 || d3 < 0
        let hasPositive = d1 > 0 || d2 > 0 || d3 > 0
        if hasNegative && hasPositive { return false }
        let area = d1 + d2 + d3
        if abs(area) < 1e-9 { return false }
        let inverseDepth = (d2 / p.depth + d3 / q.depth + d1 / r.depth) / area
        return inverseDepth > Self.occlusionDepthMargin / at.depth
    }
}
