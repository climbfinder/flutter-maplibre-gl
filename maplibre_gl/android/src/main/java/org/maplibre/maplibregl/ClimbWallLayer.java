package org.maplibre.maplibregl;

import java.util.Map;
import org.maplibre.android.style.layers.CustomLayer;

/**
 * Climbfinder fork: a climb's 3D wall drawn by our own OpenGL ES code (climb_wall_layer.cpp) as a
 * MapLibre custom layer. The mesh is uploaded once; the camera only changes a matrix, so panning,
 * zooming and orbiting cost no rebuild and no GeoJSON tiling.
 *
 * <p>MapLibre owns the native host once the layer is made; the state the host draws from is shared,
 * so this handle can still feed it new meshes and outlive the host.
 */
final class ClimbWallLayer {
  static {
    System.loadLibrary("climbwall");
  }

  final CustomLayer layer;
  private long state;

  ClimbWallLayer(String id) {
    state = nativeCreateState();
    layer = new CustomLayer(id, nativeCreateHost(state));
  }

  void update(float[] wall, float[] shadow, Map<String, Object> params) {
    if (state == 0) return;
    if (wall != null && shadow != null) nativeSetMesh(state, wall, shadow);
    if (params == null) return;
    nativeSetParams(
        state,
        number(params, "originX"),
        number(params, "originY"),
        number(params, "latitude"),
        number(params, "invCos"),
        number(params, "exaggeration"),
        number(params, "graphIntervalM"),
        number(params, "totalM"),
        number(params, "zoomStep"),
        (float[]) params.get("halfWidthByZoom"),
        (float[]) params.get("riseByZoom"),
        (float[]) params.get("surfaceColors"));
  }

  void release() {
    if (state == 0) return;
    nativeReleaseState(state);
    state = 0;
  }

  private static double number(Map<String, Object> params, String key) {
    Object value = params.get(key);
    return value instanceof Number ? ((Number) value).doubleValue() : 0;
  }

  private static native long nativeCreateState();

  private static native long nativeCreateHost(long state);

  private static native void nativeSetMesh(long state, float[] wall, float[] shadow);

  private static native void nativeSetParams(
      long state,
      double originX,
      double originY,
      double latitude,
      double invCos,
      double exaggeration,
      double graphIntervalM,
      double totalM,
      double zoomStep,
      float[] halfWidthByZoom,
      float[] riseByZoom,
      float[] surfaceColors);

  private static native void nativeReleaseState(long state);
}
