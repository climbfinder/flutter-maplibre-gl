package org.maplibre.maplibregl;

import androidx.annotation.NonNull;
import io.flutter.plugin.common.BinaryMessenger;
import io.flutter.plugin.common.MethodCall;
import io.flutter.plugin.common.MethodChannel;
import java.util.HashMap;
import java.util.Map;
import org.maplibre.android.maps.MapLibreMap;
import org.maplibre.android.maps.Style;

/**
 * Climbfinder fork: the climb wall's calls of one map. They travel on a method channel of their
 * own rather than on the plugin's Pigeon messages, so a bump of upstream never has to merge into
 * its generated code.
 */
final class ClimbWallChannel implements MethodChannel.MethodCallHandler {
  private final MethodChannel channel;
  private final MapLibreMapController controller;
  private final Map<String, ClimbWallLayer> layers = new HashMap<>();

  ClimbWallChannel(BinaryMessenger messenger, int id, MapLibreMapController controller) {
    this.controller = controller;
    channel = new MethodChannel(messenger, "plugins.climbfinder.com/climb_wall_" + id);
    channel.setMethodCallHandler(this);
  }

  @Override
  public void onMethodCall(@NonNull MethodCall call, @NonNull MethodChannel.Result result) {
    switch (call.method) {
      case "climbWall#set":
        set(call, result);
        break;
      case "climbWall#remove":
        remove(call.argument("layerId"));
        result.success(null);
        break;
      default:
        result.notImplemented();
    }
  }

  private void set(MethodCall call, MethodChannel.Result result) {
    final Style style = controller.climbWallStyle();
    final MapLibreMap map = controller.climbWallMap();
    if (style == null || map == null) {
      result.error(StyleChecks.STYLE_NOT_READY, StyleChecks.STYLE_NOT_READY_MESSAGE, null);
      return;
    }
    final String layerId = call.argument("layerId");
    ClimbWallLayer wall = layers.get(layerId);
    // A style reload drops the layer but not our handle to it.
    if (wall == null || style.getLayer(layerId) == null) {
      if (wall != null) wall.release();
      if (style.getLayer(layerId) != null) style.removeLayer(layerId);
      wall = new ClimbWallLayer(layerId);
      style.addLayer(wall.layer);
      layers.put(layerId, wall);
    }
    wall.update(call.argument("wall"), call.argument("shadow"), call.argument("params"));
    map.triggerRepaint();
    result.success(null);
  }

  private void remove(String layerId) {
    final ClimbWallLayer wall = layers.remove(layerId);
    final Style style = controller.climbWallStyle();
    if (style != null) style.removeLayer(layerId);
    if (wall != null) wall.release();
  }

  /** A call after this finds no handler, which the Dart side answers with MAP_DISPOSED. */
  void dispose() {
    channel.setMethodCallHandler(null);
    for (ClimbWallLayer wall : layers.values()) wall.release();
    layers.clear();
  }
}
