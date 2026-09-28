/// The similarity index behind every search the engine runs.
library;

import 'dart:typed_data';

import 'vector_math.dart';

/// Holds the unit vectors of the engine's indexed traces and scores them
/// against query vectors. The engine keeps it in step with its traces; the
/// default [DartVectorIndex] is pure Dart and an application may plug in a
/// native one. Implementations must compute each score exactly as [dot] does
/// (float32 inputs widened to double, products summed in double precision in
/// dimension order), so scores — and therefore tie orders — match the pure
/// implementation bit for bit.
abstract interface class VectorIndex {
  /// Stores (or replaces) the vector for [id].
  void put(String id, Float32List vector);

  /// Forgets [id] (no-op when absent).
  void remove(String id);

  /// Forgets everything.
  void clear();

  /// Scores: `out[i * queries.length + j] == dot(vector(ids[i]), queries[j])`.
  /// Every id must have been [put]. Throws ArgumentError on a dimension mismatch.
  Float64List scores(List<String> ids, List<Float32List> queries);
}

/// The pure-Dart [VectorIndex]: a map from id to vector, scored with [dot].
///
/// Vectors are kept by reference; the engine never mutates a vector it has
/// handed over (a re-embedded trace gets a new one, which is [put] again).
final class DartVectorIndex implements VectorIndex {
  final Map<String, Float32List> _vectors = {};

  /// Number of stored vectors.
  int get length => _vectors.length;

  @override
  void put(String id, Float32List vector) => _vectors[id] = vector;

  @override
  void remove(String id) => _vectors.remove(id);

  @override
  void clear() => _vectors.clear();

  @override
  Float64List scores(List<String> ids, List<Float32List> queries) {
    final k = queries.length;
    final out = Float64List(ids.length * k);
    for (var i = 0; i < ids.length; i++) {
      final v = _vectors[ids[i]] ??
          (throw ArgumentError.value(ids[i], 'ids', 'not in the index'));
      for (var j = 0; j < k; j++) {
        out[i * k + j] = dot(v, queries[j]);
      }
    }
    return out;
  }
}
