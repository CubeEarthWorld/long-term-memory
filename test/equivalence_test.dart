import 'dart:convert';
import 'dart:io';

import 'package:test/test.dart';

import 'support/workload.dart';

/// Same-language regression: a seeded workload (remember with cues, recall,
/// cite, clusters, dream with both verdicts, eviction) whose full-precision
/// outcome was recorded with 1.0.1 in `test/equivalence/golden.json`. Every
/// score is compared as the shortest round-trip string of the double, so the
/// batched cue embeddings, the one-pass cluster scan and the [VectorIndex]
/// must reproduce 1.0.1 bit for bit. Regenerate (only when the algorithm is
/// meant to change) with `EQUIVALENCE_WRITE=1 dart test test/equivalence_test.dart`.
void main() {
  final golden = File('test/equivalence/golden.json');

  test('seeded workload reproduces the 1.0.1 outcome bit for bit', () async {
    final trace = await runWorkload();
    if (Platform.environment['EQUIVALENCE_WRITE'] == '1') {
      golden.createSync(recursive: true);
      golden.writeAsStringSync(
          '${const JsonEncoder.withIndent(' ').convert(trace)}\n');
    }
    expect(trace, jsonDecode(golden.readAsStringSync()));
  });
}
