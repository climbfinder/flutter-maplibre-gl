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
        var invCos: Float
        var exaggeration: Float
        var riseM: Float
        var graphIntervalM: Float
        var totalM: Float
        var pad: Float = 0
        var surface0: SIMD4<Float>
        var surface1: SIMD4<Float>
        var surface2: SIMD4<Float>
        var surface3: SIMD4<Float>
    }

    private var device: MTLDevice?
    private var wallPipeline: MTLRenderPipelineState?
    private var shadowPipeline: MTLRenderPipelineState?
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
    private var invCos = 1.0
    private var exaggeration = 1.0
    private var graphIntervalM = 0.0
    private var totalM = 0.0
    private var zoomStep = 0.1
    private var halfWidthByZoom: [Float] = []
    private var riseByZoom: [Float] = []
    private var surfaceColors: [SIMD4<Float>] = Array(repeating: SIMD4(0.5, 0.5, 0.5, 1), count: 4)

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
        invCos = number("invCos") ?? invCos
        exaggeration = number("exaggeration") ?? exaggeration
        graphIntervalM = number("graphIntervalM") ?? graphIntervalM
        totalM = number("totalM") ?? totalM
        zoomStep = number("zoomStep") ?? zoomStep
        halfWidthByZoom = floats("halfWidthByZoom") ?? halfWidthByZoom
        riseByZoom = floats("riseByZoom") ?? riseByZoom
        if let colors = floats("surfaceColors"), colors.count == 12 {
            surfaceColors = (0..<4).map { SIMD4(colors[$0 * 3], colors[$0 * 3 + 1], colors[$0 * 3 + 2], 1) }
        }
        setNeedsDisplay()
    }

    override func didMove(to mapView: MLNMapView) {
        let resource = mapView.backendResource()
        guard let device = resource.device, let view = resource.mtkView else { return }
        self.device = device
        let library: MTLLibrary
        do {
            library = try device.makeLibrary(source: Self.shaderSource, options: nil)
        } catch {
            NSLog("ClimbWallStyleLayer: shader compile failed: \(error)")
            return
        }

        func pipeline(fragment: String, isBlended: Bool) -> MTLRenderPipelineState? {
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
            do {
                return try device.makeRenderPipelineState(descriptor: descriptor)
            } catch {
                NSLog("ClimbWallStyleLayer: pipeline failed: \(error)")
                return nil
            }
        }
        wallPipeline = pipeline(fragment: "wallFragment", isBlended: false)
        shadowPipeline = pipeline(fragment: "shadowFragment", isBlended: true)

        let wallDescriptor = MTLDepthStencilDescriptor()
        wallDescriptor.depthCompareFunction = .lessEqual
        wallDescriptor.isDepthWriteEnabled = true
        wallDepth = device.makeDepthStencilState(descriptor: wallDescriptor)
        let shadowDescriptor = MTLDepthStencilDescriptor()
        shadowDescriptor.depthCompareFunction = .always
        shadowDescriptor.isDepthWriteEnabled = false
        shadowDepth = device.makeDepthStencilState(descriptor: shadowDescriptor)
    }

    override func willMove(from _: MLNMapView) {
        wallBuffer = nil
        shadowBuffer = nil
        wallPipeline = nil
        shadowPipeline = nil
    }

    override func draw(in _: MLNMapView, with context: MLNStyleLayerDrawingContext) {
        guard let encoder = renderEncoder, let device = device,
              let wallPipeline = wallPipeline, let shadowPipeline = shadowPipeline
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

        var uniforms = makeUniforms(context)
        let length = MemoryLayout<Uniforms>.stride
        encoder.setCullMode(.none)
        if let shadowBuffer = shadowBuffer, shadowCount > 0 {
            encoder.setRenderPipelineState(shadowPipeline)
            encoder.setDepthStencilState(shadowDepth)
            encoder.setVertexBuffer(shadowBuffer, offset: 0, index: 0)
            encoder.setVertexBytes(&uniforms, length: length, index: 1)
            encoder.setFragmentBytes(&uniforms, length: length, index: 1)
            encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: shadowCount)
        }
        encoder.setRenderPipelineState(wallPipeline)
        encoder.setDepthStencilState(wallDepth)
        encoder.setVertexBuffer(wallBuffer, offset: 0, index: 0)
        encoder.setVertexBytes(&uniforms, length: length, index: 1)
        encoder.setFragmentBytes(&uniforms, length: length, index: 1)
        encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: wallCount)
    }

    /// In world pixels at this zoom the origin's coordinates run to tens of
    /// millions, too many for a float: the matrix takes the mesh from meters
    /// around the origin to clip space in doubles, and only then is rounded.
    private func makeUniforms(_ context: MLNStyleLayerDrawingContext) -> Uniforms {
        let zoom = context.zoomLevel
        let worldSize = Self.tileSize * pow(2.0, zoom)
        let scale = worldSize / Self.earthCircumferenceM
        let m = context.projectionMatrix
        let p: [Double] = [
            m.m00, m.m01, m.m02, m.m03, m.m10, m.m11, m.m12, m.m13,
            m.m20, m.m21, m.m22, m.m23, m.m30, m.m31, m.m32, m.m33,
        ]
        let model: [Double] = [scale, 0, 0, 0, 0, scale, 0, 0, 0, 0, scale, 0, originX * worldSize, originY * worldSize, 0, 1]
        var columns = [SIMD4<Float>](repeating: .zero, count: 4)
        for col in 0..<4 {
            for row in 0..<4 {
                var sum = 0.0
                for i in 0..<4 { sum += p[i * 4 + row] * model[col * 4 + i] }
                columns[col][row] = Float(sum)
            }
        }
        let metersPerPx = Self.earthCircumferenceM * cos(latitude * .pi / 180) / worldSize
        return Uniforms(
            matrix: simd_float4x4(columns: (columns[0], columns[1], columns[2], columns[3])),
            halfWidthM: lookup(halfWidthByZoom, zoom),
            metersPerPx: Float(metersPerPx),
            invCos: Float(invCos),
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
        float invCos;
        float exaggeration;
        float riseM;
        float graphIntervalM;
        float totalM;
        float pad;
        float4 surfaceColors[4];
    };

    struct Out {
        float4 position [[position]];
        float3 color;
        float across;
        float u;
        float kind [[flat]];
        float surface [[flat]];
    };

    vertex Out wallVertex(uint vid [[vertex_id]],
                          const device Vertex* vertices [[buffer(0)]],
                          constant Uniforms& U [[buffer(1)]]) {
        Vertex v = vertices[vid];
        float2 p = float2(v.center) + float2(v.normal) * v.across * U.halfWidthM;
        float z = v.isTop * (v.elevation * U.exaggeration + 1.0 + U.riseM * v.fraction) * U.invCos;
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

    fragment float4 wallFragment(Out in [[stage_in]], constant Uniforms& U [[buffer(1)]]) {
        float hw = U.halfWidthM;
        float mpp = U.metersPerPx;
        int kind = int(in.kind + 0.5);
        if (kind == 0) {
            float3 c = in.color;
            if (U.graphIntervalM > 0.0) {
                float k = floor(in.u / U.graphIntervalM + 0.5);
                float stripe = k * U.graphIntervalM;
                if (k > 0.0 && stripe < U.totalM && abs(in.u - stripe) < mpp) c = mix(c, float3(1.0), 0.6);
            }
            return float4(c, 1.0);
        }
        float a = abs(in.across);
        if (hw * 0.22 / mpp >= 0.5 && a > 0.78) return float4(0.078, 0.078, 0.078, 1.0);
        int surface = int(in.surface + 0.5);
        float3 c = road(surface, in.u, in.across * hw + hw, hw, mpp, U);
        bool isDashed = kind == 1 && (surface == 0 || surface == 2) && hw * 0.22 / mpp >= 0.5;
        if (isDashed && a < 0.11 && fmod(in.u, 4.0 * hw) < 2.0 * hw) c = float3(0.969);
        return float4(c, 1.0);
    }

    fragment float4 shadowFragment(Out in [[stage_in]]) {
        float t = clamp((abs(in.across) / 5.0 - 0.2) / 0.8, 0.0, 1.0);
        float alpha = 0.4 * (1.0 - t * t * (3.0 - 2.0 * t));
        return float4(0.0, 0.0, 0.0, alpha);
    }
    """
}
