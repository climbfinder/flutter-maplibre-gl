// Climbfinder fork: an expression holding a null literal, such as
// ['==', ['get', 'score'], null], reached MapLibre Android as a Java null in
// the expression array and aborted the app in JNI ("jarray was NULL" in
// Layer.nativeSetPaintProperty).
//
//   flutter test integration_test/null_literal_expression_test.dart -d <device>
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:maplibre_gl/maplibre_gl.dart';

import 'support/map_harness.dart';

const _sourceId = 'lines';
const _layerId = 'line';

const List<Object?> _noScore = [
  '==',
  ['get', 'score'],
  null,
];

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets('a layer property and a filter take a null literal', (
    tester,
  ) async {
    final controller = await pumpMap(tester);
    await pumpUntil(
      tester,
      controller.addGeoJsonSource(_sourceId, const {
        'type': 'FeatureCollection',
        'features': [
          {
            'type': 'Feature',
            'properties': {'score': 3},
            'geometry': {
              'type': 'LineString',
              'coordinates': [
                [0, 0],
                [1, 1],
              ],
            },
          },
        ],
      }),
      'addGeoJsonSource',
    );
    await pumpUntil(
      tester,
      controller.addLineLayer(
        _sourceId,
        _layerId,
        const LineLayerProperties(lineWidth: 3),
      ),
      'addLineLayer',
    );

    await pumpUntil(
      tester,
      controller.setLayerProperties(
        _layerId,
        const LineLayerProperties(
          lineColor: ['case', _noScore, '#ffffff', '#ff0000'],
        ),
      ),
      'setLayerProperties',
    );
    await pumpUntil(
      tester,
      controller.setLayerFilter(_layerId, ['!', _noScore]),
      'setLayerFilter',
    );

    expect(
      await pumpUntil(tester, controller.getLayerIds(), 'getLayerIds'),
      contains(_layerId),
    );
  });
}
