/// Deterministic test doubles: a token-overlap embedder whose vectors are
/// bit-for-bit reproducible in the Python reference tests (FNV-1a token
/// seeds → splitmix64 streams), a virtual clock and canned adjudicators.
library;

import 'dart:convert';
import 'dart:typed_data';

import 'package:long_term_memory/long_term_memory.dart';

/// Virtual clock: lets scenarios accelerate time at zero real cost.
class VirtualClock {
  VirtualClock([this.t = 1700000000]);

  int t;

  int call() => t;

  void advance(int seconds) => t += seconds;

  void advanceDays(num days) => t += (days * 86400).round();
}

/// Embedder whose vectors are sums of per-token pseudo-random vectors:
/// `cos ≈ shared / √(n1·n2)` for texts with overlapping token sets, and
/// identical token multisets give identical vectors.
class FakeEmbedder implements Embedder {
  FakeEmbedder({this.dimension = 64, this.modelId = 'fake/token-overlap'});

  final int dimension;

  @override
  final String modelId;

  final Map<String, Float64List> _cache = {};

  static List<String> tokens(String text) {
    final lower = text.toLowerCase();
    return [
      ...RegExp(r'[a-z0-9]+').allMatches(lower).map((m) => m.group(0)!),
      for (final rune in lower.runes)
        if ((rune >= 0x3040 && rune <= 0x9FFF) ||
            (rune >= 0xFF66 && rune <= 0xFF9D))
          String.fromCharCode(rune),
    ];
  }

  static int _fnv1a(String s) {
    var h = 0xcbf29ce484222325;
    for (final b in utf8.encode(s)) {
      h = (h ^ b) * 0x100000001b3;
    }
    return h;
  }

  static int _splitmix(int x) {
    x = (x ^ (x >>> 30)) * 0xBF58476D1CE4E5B9;
    x = (x ^ (x >>> 27)) * 0x94D049BB133111EB;
    return x ^ (x >>> 31);
  }

  Float64List _tokenVec(String tok) => _cache.putIfAbsent(tok, () {
        var state = _fnv1a(tok);
        final v = Float64List(dimension);
        for (var i = 0; i < dimension; i++) {
          state += 0x9E3779B97F4A7C15;
          v[i] = (_splitmix(state) >>> 11) / 9007199254740992.0 * 2 - 1;
        }
        return v;
      });

  Float32List embed(String text) {
    final toks = tokens(text);
    final v = Float64List(dimension);
    if (toks.isEmpty) {
      v.fillRange(0, dimension, 1.0);
    } else {
      for (final t in toks) {
        final tv = _tokenVec(t);
        for (var i = 0; i < dimension; i++) {
          v[i] += tv[i];
        }
      }
    }
    return l2Normalized(v);
  }

  @override
  Future<List<Float32List>> embedDocuments(List<String> texts) async =>
      [for (final t in texts) embed(t)];

  @override
  Future<List<Float32List>> embedQueries(List<String> texts) async =>
      [for (final t in texts) embed(t)];
}

/// Embedder that always throws (simulates an unavailable model).
class BrokenEmbedder implements Embedder {
  @override
  String get modelId => 'fake/broken';
  @override
  Future<List<Float32List>> embedDocuments(List<String> texts) =>
      throw StateError('embedder offline');
  @override
  Future<List<Float32List>> embedQueries(List<String> texts) =>
      throw StateError('embedder offline');
}

/// Adjudicator that merges the cluster into one gist (texts joined).
DreamDecision mergeToGist(DreamRequest request) =>
    DreamDecision([request.members.map((m) => m.text).join(' / ')]);

/// Adjudicator that always answers "keep".
DreamDecision alwaysKeep(DreamRequest request) => const DreamDecision.keep();

/// Builds an engine over an [InMemoryStore] with a [VirtualClock].
Future<(EngramMemory, VirtualClock, InMemoryStore)> build({
  EngramConfig config = const EngramConfig(),
  Embedder? embedder,
  InMemoryStore? store,
}) async {
  final clock = VirtualClock();
  final s = store ?? InMemoryStore();
  final memory = EngramMemory(
    store: s,
    embedder: embedder ?? FakeEmbedder(),
    config: config,
    clock: clock.call,
    defaultTimezone: const MemoryTimezone('Asia/Tokyo', Duration(hours: 9)),
  );
  await memory.initialize();
  return (memory, clock, s);
}
