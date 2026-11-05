// Climbfinder fork: the climb wall's own method channel, next to the Pigeon
// messages of the map.
@TestOn('vm')
library;

import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:maplibre_gl/maplibre_gl.dart';
import 'package:maplibre_gl/src/native/messages.g.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  const viewId = 7;
  const climbWallChannel = MethodChannel(
    'plugins.climbfinder.com/climb_wall_$viewId',
  );
  const waitForMap =
      'dev.flutter.pigeon.maplibre_gl.MapLibreMapHostApi.mapWaitForMap.$viewId';
  final messenger =
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger;

  late MapLibrePlatform Function() previousFactory;
  late MapLibrePlatform platform;
  late MapLibreMapController controller;

  MapLibreMapController controllerOn(MapLibrePlatform platform) =>
      MapLibreMapController(
        maplibrePlatform: platform,
        initialCameraPosition: const CameraPosition(target: LatLng(0, 0)),
        annotationOrder: AnnotationType.values,
        annotationConsumeTapEvents: AnnotationType.values,
      );

  setUp(() async {
    previousFactory = MapLibrePlatform.createInstance;
    MapLibreNativePlugin.registerWith();
    platform = MapLibrePlatform.createInstance();
    messenger.setMockMessageHandler(
      waitForMap,
      (_) async =>
          MapLibreMapHostApi.pigeonChannelCodec.encodeMessage(<Object?>[null]),
    );
    controller = controllerOn(platform);
    await platform.initPlatform(viewId);
  });

  tearDown(() {
    messenger.setMockMethodCallHandler(climbWallChannel, null);
    messenger.setMockMessageHandler(waitForMap, null);
    platform.dispose();
    MapLibrePlatform.createInstance = previousFactory;
  });

  test(
    'setClimbWall sends the meshes and params on the channel of its map',
    () async {
      final calls = <MethodCall>[];
      messenger.setMockMethodCallHandler(climbWallChannel, (call) async {
        calls.add(call);
        return null;
      });

      await controller.setClimbWall(
        'wall',
        wall: Float32List.fromList([1, 2]),
        shadow: Float32List.fromList([3]),
        params: {'zoomStep': 0.5},
      );
      await controller.setClimbWall('wall', params: {'zoomStep': 1.0});
      await controller.removeClimbWall('wall');

      expect(calls.map((call) => call.method), [
        'climbWall#set',
        'climbWall#set',
        'climbWall#remove',
      ]);
      final first = calls[0].arguments as Map<Object?, Object?>;
      expect(first['layerId'], 'wall');
      expect(first['wall'], Float32List.fromList([1, 2]));
      expect(first['shadow'], Float32List.fromList([3]));
      expect(first['params'], {'zoomStep': 0.5});
      expect(calls[1].arguments, {
        'layerId': 'wall',
        'params': {'zoomStep': 1.0},
      });
      expect(calls[2].arguments, {'layerId': 'wall'});
    },
  );

  test('a call native has no handler for fails with MAP_DISPOSED', () async {
    await expectLater(
      controller.setClimbWall('wall', params: {}),
      throwsA(
        isA<PlatformException>().having((e) => e.code, 'code', 'MAP_DISPOSED'),
      ),
    );
  });

  test('a native error reaches the caller', () async {
    messenger.setMockMethodCallHandler(
      climbWallChannel,
      (_) async => throw PlatformException(code: 'STYLE_NOT_READY'),
    );

    await expectLater(
      controller.setClimbWall('wall', params: {}),
      throwsA(
        isA<PlatformException>().having(
          (e) => e.code,
          'code',
          'STYLE_NOT_READY',
        ),
      ),
    );
  });

  test(
    'a call before the platform view exists fails with STYLE_NOT_READY',
    () async {
      final early = MapLibrePlatform.createInstance();
      addTearDown(early.dispose);

      expect(
        () => controllerOn(early).setClimbWall('wall', params: {}),
        throwsA(
          isA<PlatformException>().having(
            (e) => e.code,
            'code',
            'STYLE_NOT_READY',
          ),
        ),
      );
    },
  );
}
