// Integration test for the zoom-in when Android starts tracking the user.
//
// MapLibre iOS zooms in to 14 when follow mode starts below zoom 10.5; the
// Android LocationComponent only pans, so MapLibreMapController.java passes the
// same zoom along with the tracking transition. That lives between the plugin
// and the SDK's LocationComponent, where no unit test can see it, and an SDK
// bump can change the component's behaviour underneath it. A manual location
// source keeps the fixes deterministic and needs no location permission.
//
// Run on an Android device or emulator:
//   flutter test integration_test/android_tracking_zoom_test.dart -d <device>
//
// The assertions only run on Android; elsewhere the tests are no-ops.
import 'dart:async';

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:maplibre_gl/maplibre_gl.dart';

const _style =
    '{"version":8,"sources":{},"layers":[{"id":"background",'
    '"type":"background","paint":{"background-color":"#ffffff"}}]}';

const _start = LatLng(52.0, 5.0);
const _user = LatLng(45.9, 6.9);

final bool _onAndroid =
    !kIsWeb && defaultTargetPlatform == TargetPlatform.android;

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets('tracking from a zoomed-out map zooms in to the user', (
    tester,
  ) async {
    if (!_onAndroid) return;
    final controller = await _pumpMap(tester, zoom: 5);
    await controller.updateManualLocation(
      const ManualLocationUpdate(target: _user),
    );
    await _settle();

    await controller.updateMyLocationTrackingMode(
      MyLocationTrackingMode.tracking,
    );
    await _settle();

    final camera = (await controller.queryCameraPosition())!;
    expect(camera.zoom, closeTo(14, 0.01));
    expect(camera.target.latitude, closeTo(_user.latitude, 0.001));
    expect(camera.target.longitude, closeTo(_user.longitude, 0.001));
  });

  testWidgets('tracking from a zoomed-in map keeps the zoom', (tester) async {
    if (!_onAndroid) return;
    final controller = await _pumpMap(tester, zoom: 12);
    await controller.updateManualLocation(
      const ManualLocationUpdate(target: _user),
    );
    await _settle();

    await controller.updateMyLocationTrackingMode(
      MyLocationTrackingMode.tracking,
    );
    await _settle();

    final camera = (await controller.queryCameraPosition())!;
    expect(camera.zoom, closeTo(12, 0.01));
    expect(camera.target.latitude, closeTo(_user.latitude, 0.001));
  });

  testWidgets('without a fix yet, the first fix zooms in', (tester) async {
    if (!_onAndroid) return;
    final controller = await _pumpMap(tester, zoom: 5);

    await controller.updateMyLocationTrackingMode(
      MyLocationTrackingMode.tracking,
    );
    await _settle();
    expect((await controller.queryCameraPosition())!.zoom, closeTo(5, 0.01));

    await controller.updateManualLocation(
      const ManualLocationUpdate(target: _user),
    );
    await _settle();

    final camera = (await controller.queryCameraPosition())!;
    expect(camera.zoom, closeTo(14, 0.01));
    expect(camera.target.latitude, closeTo(_user.latitude, 0.001));
  });
}

/// Longer than the 750 ms tracking transition and zoom animation.
Future<void> _settle() =>
    Future<void>.delayed(const Duration(milliseconds: 2000));

Future<MapLibreMapController> _pumpMap(
  WidgetTester tester, {
  required double zoom,
}) async {
  final created = Completer<MapLibreMapController>();
  final styled = Completer<void>();

  await tester.pumpWidget(
    MaterialApp(
      home: Scaffold(
        body: MapLibreMap(
          // A fresh key per test, so each one gets a new native map and
          // location component instead of reusing the previous test's.
          key: UniqueKey(),
          styleString: _style,
          locationSource: const ManualLocationSource(),
          myLocationEnabled: true,
          initialCameraPosition: CameraPosition(target: _start, zoom: zoom),
          onMapCreated: created.complete,
          onStyleLoadedCallback: styled.complete,
        ),
      ),
    ),
  );

  final controller = await created.future;
  await styled.future;
  // The location component is activated from the style load; give it a frame.
  await Future<void>.delayed(const Duration(milliseconds: 500));
  return controller;
}
