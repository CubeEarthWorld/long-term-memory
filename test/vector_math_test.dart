import 'dart:math';
import 'dart:typed_data';

import 'package:long_term_memory/long_term_memory.dart';
import 'package:test/test.dart';

Float32List _randomVec(int dim, int seed) {
  final rng = Random(seed);
  final v = Float32List(dim);
  for (var i = 0; i < dim; i++) {
    v[i] = rng.nextDouble() * 2 - 1;
  }
  return v;
}

void main() {
  test('l2Normalized produces unit norm and keeps zero vectors intact', () {
    final v = l2Normalized(_randomVec(768, 1));
    expect(sqrt(dot(v, v)), closeTo(1.0, 1e-6));
    final zero = Float32List(8);
    expect(l2Normalized(zero), equals(zero));
  });

  test('truncateNormalize keeps the prefix direction (MRL property)', () {
    final v = l2Normalized(_randomVec(768, 2));
    final t = truncateNormalize(v, 256);
    expect(t.length, 256);
    expect(sqrt(dot(t, t)), closeTo(1.0, 1e-6));
    // Direction of the prefix is preserved: cosine with the raw prefix is 1.
    final prefix = Float32List.sublistView(v, 0, 256);
    expect(cosine(t, prefix), closeTo(1.0, 1e-6));
  });

  test('f32 pack/unpack round-trips exactly', () {
    final v = l2Normalized(_randomVec(64, 3));
    expect(unpackF32(packF32(v)), equals(v));
  });

  test('int8 quantisation round-trips with cosine > 0.99', () {
    final v = l2Normalized(_randomVec(256, 4));
    final q = quantizeInt8(v);
    expect(q.bytes.length, 256);
    final back = l2Normalized(dequantizeInt8(q.bytes, q.scale));
    expect(cosine(v, back), greaterThan(0.99));
  });

  test(
    'cohesion is 1.0 for singletons and high for near-identical vectors',
    () {
      final v = l2Normalized(_randomVec(128, 5));
      expect(cohesion([v]), 1.0);
      expect(cohesion([v, v, v]), closeTo(1.0, 1e-5));
      final w = l2Normalized(_randomVec(128, 6));
      expect(cohesion([v, w]), lessThan(0.5));
    },
  );

  test('sphericalKMeans is deterministic and separates distinct groups', () {
    final a = l2Normalized(_randomVec(64, 10));
    final b = l2Normalized(_randomVec(64, 20));
    Float32List jitter(Float32List base, int seed) {
      final noise = _randomVec(64, seed);
      final out = Float32List(64);
      for (var i = 0; i < 64; i++) {
        out[i] = base[i] + 0.05 * noise[i];
      }
      return l2Normalized(out);
    }

    final rows = [
      jitter(a, 1),
      jitter(a, 2),
      jitter(a, 3),
      jitter(b, 4),
      jitter(b, 5),
      jitter(b, 6),
    ];
    final labels1 = sphericalKMeans(rows, 2);
    final labels2 = sphericalKMeans(rows, 2);
    expect(labels1, equals(labels2));
    expect(labels1.sublist(0, 3).toSet().length, 1);
    expect(labels1.sublist(3).toSet().length, 1);
    expect(labels1[0], isNot(labels1[3]));
  });

  group('sphericalKMeans edge cases', () {
    test('empty rows returns empty list', () {
      expect(sphericalKMeans([], 3), isEmpty);
    });

    test('k=1 assigns all to cluster 0', () {
      final rows = [
        l2Normalized(_randomVec(16, 1)),
        l2Normalized(_randomVec(16, 2)),
        l2Normalized(_randomVec(16, 3)),
      ];
      final labels = sphericalKMeans(rows, 1);
      expect(labels.every((l) => l == 0), isTrue);
    });

    test('k >= n assigns each row its own cluster', () {
      final rows = [
        l2Normalized(_randomVec(16, 1)),
        l2Normalized(_randomVec(16, 2)),
      ];
      final labels = sphericalKMeans(rows, 10);
      expect(labels.toSet().length, 2);
    });
  });

  group('cohesion edge cases', () {
    test('two identical vectors have cohesion 1.0', () {
      final v = l2Normalized(Float32List.fromList([1.0, 2.0, 3.0, 4.0]));
      expect(cohesion([v, v]), closeTo(1.0, 1e-6));
    });

    test('two orthogonal vectors have cohesion near 0', () {
      final a = l2Normalized(Float32List.fromList([1.0, 0.0]));
      final b = l2Normalized(Float32List.fromList([0.0, 1.0]));
      expect(cohesion([a, b]), closeTo(0.0, 1e-6));
    });

    test('larger set of vectors computes correctly', () {
      final vectors = List.generate(
        10,
        (i) => l2Normalized(_randomVec(32, i * 7)),
      );
      final c = cohesion(vectors);
      expect(c, greaterThan(0.0));
      expect(c, lessThanOrEqualTo(1.0));
    });
  });

  group('dot product edge cases', () {
    test('dot with different length vectors uses shorter', () {
      final a = Float32List.fromList([1.0, 0.0, 0.0]);
      final b = Float32List.fromList([1.0, 0.0]);
      expect(dot(a, b), closeTo(1.0, 1e-9));
    });

    test('dot with empty vectors returns 0', () {
      expect(dot(Float32List(0), Float32List(0)), 0.0);
    });
  });

  group('cosine edge cases', () {
    test('cosine with zero vector returns 0', () {
      final zero = Float32List(4);
      final v = _randomVec(4, 1);
      expect(cosine(zero, v), 0.0);
      expect(cosine(v, zero), 0.0);
    });

    test('cosine with both zero vectors returns 0', () {
      expect(cosine(Float32List(4), Float32List(4)), 0.0);
    });
  });

  group('quantizeInt8 edge cases', () {
    test('all-zero vector quantizes with scale 1.0', () {
      final v = Float32List(16);
      final q = quantizeInt8(v);
      expect(q.scale, 1.0);
      expect(q.bytes.every((b) => b == 0), isTrue);
    });

    test('uniform vector round-trips with high fidelity', () {
      final v = Float32List(32);
      final val = 1.0 / sqrt(32);
      for (var i = 0; i < 32; i++) {
        v[i] = val;
      }
      final q = quantizeInt8(v);
      final back = l2Normalized(dequantizeInt8(q.bytes, q.scale));
      expect(cosine(v, back), greaterThan(0.99));
    });
  });
}
