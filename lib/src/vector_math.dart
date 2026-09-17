/// Pure-Dart vector helpers. Every vector the engine holds is a unit-norm
/// [Float32List]; accumulation happens in doubles.
library;

import 'dart:math';
import 'dart:typed_data';

/// Returns an L2-normalised copy of the first [dim] components of [v]
/// (zero vectors are returned as-is).
Float32List l2Normalized(List<double> v, {int? dim}) {
  final n = dim == null ? v.length : min(dim, v.length);
  var sum = 0.0;
  for (var i = 0; i < n; i++) {
    sum += v[i] * v[i];
  }
  final norm = sqrt(sum);
  final out = Float32List(n);
  for (var i = 0; i < n; i++) {
    out[i] = norm == 0.0 ? v[i] : v[i] / norm;
  }
  return out;
}

/// Dot product (== cosine for unit-norm inputs).
double dot(Float32List a, Float32List b) {
  final n = min(a.length, b.length);
  var sum = 0.0;
  for (var i = 0; i < n; i++) {
    sum += a[i] * b[i];
  }
  return sum;
}

/// Packs a float32 vector into little-endian bytes (4 bytes/dim).
Uint8List packF32(Float32List v) {
  final data = ByteData(v.length * 4);
  for (var i = 0; i < v.length; i++) {
    data.setFloat32(i * 4, v[i], Endian.little);
  }
  return data.buffer.asUint8List();
}

/// Inverse of [packF32].
Float32List unpackF32(Uint8List bytes) {
  final data = ByteData.sublistView(bytes);
  final out = Float32List(bytes.length ~/ 4);
  for (var i = 0; i < out.length; i++) {
    out[i] = data.getFloat32(i * 4, Endian.little);
  }
  return out;
}
