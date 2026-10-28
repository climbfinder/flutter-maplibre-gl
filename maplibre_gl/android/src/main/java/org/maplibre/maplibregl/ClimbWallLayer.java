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

  private static final int LABEL_META = 11;

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
        number(params, "exaggeration"),
        number(params, "graphIntervalM"),
        number(params, "totalM"),
        number(params, "zoomStep"),
        (float[]) params.get("halfWidthByZoom"),
        (float[]) params.get("riseByZoom"),
        (float[]) params.get("surfaceColors"));
    Object tops = params.get("tops");
    if (tops instanceof float[]) nativeSetTops(state, (float[]) tops, number(params, "maxElevation"));
    Object labels = params.get("labels");
    if (labels instanceof java.util.List) setLabels((java.util.List<?>) labels);
  }

  /** See the app's ClimbWallLabelSprite: an image and where it stands, per label. */
  private void setLabels(java.util.List<?> labels) {
    byte[][] images = new byte[labels.size()][];
    int[] sizes = new int[labels.size() * 2];
    double[] meta = new double[labels.size() * LABEL_META];
    for (int i = 0; i < labels.size(); i++) {
      @SuppressWarnings("unchecked")
      Map<String, Object> label = (Map<String, Object>) labels.get(i);
      images[i] = (byte[]) label.get("rgba");
      sizes[i * 2] = (int) number(label, "width");
      sizes[i * 2 + 1] = (int) number(label, "height");
      String[] keys = {
        "pixelRatio", "anchorX", "anchorY", "pivotY", "halfWidth", "bodyHeight",
        "x", "y", "elevation", "fraction", "distance"
      };
      for (int k = 0; k < keys.length; k++) meta[i * LABEL_META + k] = number(label, keys[k]);
    }
    nativeSetLabels(state, images, sizes, meta);
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
      double exaggeration,
      double graphIntervalM,
      double totalM,
      double zoomStep,
      float[] halfWidthByZoom,
      float[] riseByZoom,
      float[] surfaceColors);

  private static native void nativeSetTops(long state, float[] tops, double maxElevation);

  private static native void nativeSetLabels(long state, Object[] images, int[] sizes, double[] meta);

  private static native void nativeReleaseState(long state);
}
