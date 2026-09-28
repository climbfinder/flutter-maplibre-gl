// Integration test for dragging a feature on Android.
//
// The drag lives in MapLibreMapController.java, between the MotionEvents the
// view receives and the gesture detectors of the map itself; no unit test can
// see it. The gestures below go through Flutter's platform view, which hands
// the view the same MotionEvents a finger would.
//
// Run on an Android device or emulator:
//   flutter test integration_test/android_feature_drag_test.dart -d <device>
//
// The assertions only run on Android; elsewhere the tests are no-ops.
import 'dart:async';
import 'dart:developer' show Timeline;

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:maplibre_gl/maplibre_gl.dart';

// A blank style: no network, nothing on the map but the circle.
const _style =
    '{"version":8,"sources":{},"layers":[{"id":"background",'
    '"type":"background","paint":{"background-color":"#ffffff"}}]}';

const _center = LatLng(0, 0);

final bool _onAndroid =
    !kIsWeb && defaultTargetPlatform == TargetPlatform.android;

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets('a second finger ends the drag', (tester) async {
    if (!_onAndroid) return;
    final map = await _pumpMapWithDraggableCircle(tester);

    final first = _Finger(tester);
    await first.down(map.circleOnScreen);
    for (var i = 0; i < 5; i++) {
      await first.moveBy(const Offset(10, 0));
    }
    final second = _Finger(tester);
    await second.down(map.circleOnScreen + const Offset(0, 150));
    await second.moveBy(const Offset(0, 20));
    await first.moveBy(const Offset(10, 0));
    await second.up();
    await first.up();
    await _settle();

    expect(map.drags.first, DragEventType.start);
    expect(map.drags.last, DragEventType.end);
    expect(map.drags.where((e) => e == DragEventType.end), hasLength(1));
  });
}

/// Waits for the platform view to answer over the method channel.
Future<void> _settle() =>
    Future<void>.delayed(const Duration(milliseconds: 500));

/// One finger whose events carry real timestamps.
///
/// Android's gesture detectors time long presses from the event's down time;
/// the zero timestamps a plain `TestGesture` sends would make every touch a
/// long press. Timeline.now reads the monotonic clock the MotionEvents use.
class _Finger {
  _Finger(this.tester);

  final WidgetTester tester;
  late TestGesture _gesture;

  Duration get _now => Duration(microseconds: Timeline.now);

  Future<void> down(Offset at) async {
    _gesture = await tester.createGesture();
    await _gesture.down(at, timeStamp: _now);
    await Future<void>.delayed(const Duration(milliseconds: 16));
  }

  Future<void> moveBy(Offset by, {int after = 16}) async {
    await _gesture.moveBy(by, timeStamp: _now);
    await Future<void>.delayed(Duration(milliseconds: after));
  }

  Future<void> up() async {
    await _gesture.up(timeStamp: _now);
  }
}

class _MapProbe {
  final drags = <DragEventType>[];
  final featureTaps = <String>[];
  final clicks = <LatLng>[];
  final longClicks = <LatLng>[];
  late MapLibreMapController controller;
  late Offset circleOnScreen;
}

Future<_MapProbe> _pumpMapWithDraggableCircle(WidgetTester tester) async {
  final probe = _MapProbe();
  final created = Completer<MapLibreMapController>();
  final styled = Completer<void>();

  await tester.pumpWidget(
    MaterialApp(
      home: Scaffold(
        body: MapLibreMap(
          styleString: _style,
          initialCameraPosition: const CameraPosition(
            target: _center,
            zoom: 10,
          ),
          onMapCreated: created.complete,
          onStyleLoadedCallback: styled.complete,
          onMapClick: (_, latLng) => probe.clicks.add(latLng),
          onMapLongClick: (_, latLng) => probe.longClicks.add(latLng),
        ),
      ),
    ),
  );

  final controller = probe.controller = await created.future;
  await styled.future;
  controller.onFeatureDrag.add(
    (point, origin, current, delta, id, annotation, eventType) =>
        probe.drags.add(eventType),
  );
  controller.onFeatureTapped.add(
    (point, latLng, id, layerId, annotation) => probe.featureTaps.add(id),
  );
  await controller.addCircle(
    const CircleOptions(
      geometry: _center,
      circleRadius: 30,
      circleColor: '#e53935',
      draggable: true,
    ),
  );
  // Let the circle render: the feature query only sees what is drawn.
  await Future<void>.delayed(const Duration(seconds: 1));
  await tester.pump();

  probe.circleOnScreen = tester.getCenter(find.byType(MapLibreMap));
  return probe;
}
