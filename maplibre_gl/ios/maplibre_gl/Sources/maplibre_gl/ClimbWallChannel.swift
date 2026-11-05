import Flutter
import MapLibre

/// Climbfinder fork: the climb wall's calls of one map. They travel on a
/// method channel of their own rather than on the plugin's Pigeon messages, so
/// a bump of upstream never has to merge into its generated code.
final class ClimbWallChannel {
    private let channel: FlutterMethodChannel
    /// The style the wall goes on, nil until it is ready.
    private let style: () -> MLNStyle?
    private var layers = [String: ClimbWallStyleLayer]()

    init(binaryMessenger: FlutterBinaryMessenger, suffix: String, style: @escaping () -> MLNStyle?) {
        channel = FlutterMethodChannel(
            name: "plugins.climbfinder.com/climb_wall_\(suffix)",
            binaryMessenger: binaryMessenger
        )
        self.style = style
        channel.setMethodCallHandler { [weak self] call, result in
            guard let self else {
                result(MapHostApi.disposedError())
                return
            }
            self.handle(call, result: result)
        }
    }

    /// A call after this finds no handler, which the Dart side answers with
    /// MAP_DISPOSED. Only for the map that still holds the channel: after a
    /// hot restart a newer map with the same view id may have taken it over.
    func unregister() {
        channel.setMethodCallHandler(nil)
    }

    private func handle(_ call: FlutterMethodCall, result: FlutterResult) {
        let arguments = call.arguments as? [String: Any]
        switch call.method {
        case "climbWall#set":
            guard let style = style() else {
                result(FlutterError(code: "STYLE_NOT_READY", message: "The style is not loaded yet.", details: nil))
                return
            }
            guard let layerId = arguments?["layerId"] as? String,
                  let params = arguments?["params"] as? [String: Any]
            else {
                result(FlutterError(code: "INVALID_ARGUMENT", message: "climbWall#set needs a layerId and params.", details: nil))
                return
            }
            set(layerId, on: style, arguments: arguments ?? [:], params: params)
            result(nil)

        case "climbWall#remove":
            guard let layerId = arguments?["layerId"] as? String else {
                result(FlutterError(code: "INVALID_ARGUMENT", message: "climbWall#remove needs a layerId.", details: nil))
                return
            }
            layers.removeValue(forKey: layerId)
            if let style = style(), let layer = style.layer(withIdentifier: layerId) {
                style.removeLayer(layer)
            }
            result(nil)

        default:
            result(FlutterMethodNotImplemented)
        }
    }

    private func set(_ layerId: String, on style: MLNStyle, arguments: [String: Any], params: [String: Any]) {
        let layer: ClimbWallStyleLayer
        // A style reload drops the layer but not our handle to it.
        if let existing = layers[layerId], style.layer(withIdentifier: layerId) != nil {
            layer = existing
        } else {
            if let stale = style.layer(withIdentifier: layerId) {
                style.removeLayer(stale)
            }
            layer = ClimbWallStyleLayer(identifier: layerId)
            style.addLayer(layer)
            layers[layerId] = layer
        }
        layer.update(
            wall: (arguments["wall"] as? FlutterStandardTypedData)?.data,
            shadow: (arguments["shadow"] as? FlutterStandardTypedData)?.data,
            params: params
        )
    }
}
