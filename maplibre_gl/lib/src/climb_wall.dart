part of '../maplibre_gl.dart';

/// Climbfinder fork: a climb's 3D wall as a native custom layer, Metal on iOS
/// and OpenGL ES on Android, on top of every other layer.
///
/// Its calls travel on a method channel of their own, next to the plugin's
/// Pigeon messages, so a bump of upstream never has to merge into generated
/// code. Like those messages, a call on a disposed map fails with
/// `MAP_DISPOSED`, and one before the style loaded with `STYLE_NOT_READY`.
extension ClimbWallMapLibreMapController on MapLibreMapController {
  /// Stands the wall with [layerId] on the map. Calling it again with the
  /// same [layerId] replaces the mesh; [params] alone, without meshes, only
  /// updates the uniforms. See the app's `ClimbWallMesh` for the vertex
  /// layout.
  Future<void> setClimbWall(
    String layerId, {
    Float32List? wall,
    Float32List? shadow,
    required Map<String, Object> params,
  }) {
    return _climbWallChannel().invokeMethod<void>(
      'climbWall#set',
      <String, Object>{
        'layerId': layerId,
        'wall': ?wall,
        'shadow': ?shadow,
        'params': params,
      },
    );
  }

  Future<void> removeClimbWall(String layerId) {
    return _climbWallChannel().invokeMethod<void>(
      'climbWall#remove',
      <String, Object>{
        'layerId': layerId,
      },
    );
  }

  MethodChannel _climbWallChannel() {
    final platform = _maplibrePlatform;
    if (platform is! _MapLibreNativePlatform) {
      throw UnsupportedError(
        'The climb wall is only implemented on iOS and Android',
      );
    }
    final viewId = platform._viewId;
    if (viewId == null) {
      throw PlatformException(
        code: 'STYLE_NOT_READY',
        message: 'The map has no platform view yet.',
      );
    }
    return MethodChannel(
      'plugins.climbfinder.com/climb_wall_$viewId',
      const StandardMethodCodec(),
      const _MapCallMessenger(),
    );
  }
}
