import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:long_term_memory/long_term_memory.dart';
import 'package:test/test.dart';

import 'support/fakes.dart';
import 'support/workload.dart';

/// An independent [VectorIndex] that records every call, so a test can check
/// that the engine keeps it exactly in step with its traces.
class RecordingIndex implements VectorIndex {
  final Map<String, Float32List> vectors = {};
  var puts = 0;
  var removes = 0;
  var clears = 0;
  var scoreCalls = 0;

  @override
  void put(String id, Float32List vector) {
    puts++;
    vectors[id] = vector;
  }

  @override
  void remove(String id) {
    removes++;
    vectors.remove(id);
  }

  @override
  void clear() {
    clears++;
    vectors.clear();
  }

  @override
  Float64List scores(List<String> ids, List<Float32List> queries) {
    scoreCalls++;
    final out = Float64List(ids.length * queries.length);
    var o = 0;
    for (final id in ids) {
      final v = vectors[id]!;
      for (final q in queries) {
        out[o++] = dot(v, q);
      }
    }
    return out;
  }
}

/// Counts the [Embedder.embedQueries] batches; optionally fails them, or
/// returns a wrong-dimension vector for one cue.
class CountingEmbedder implements Embedder {
  CountingEmbedder({this.offlineQueries = false, this.badCue});

  final FakeEmbedder _real = FakeEmbedder();
  final bool offlineQueries;
  final String? badCue;
  final queryBatches = <List<String>>[];

  @override
  String get modelId => _real.modelId;

  @override
  int get dimension => _real.dimension;

  @override
  Future<List<Float32List>> embedDocuments(List<String> texts) =>
      _real.embedDocuments(texts);

  @override
  Future<List<Float32List>> embedQueries(List<String> texts) async {
    queryBatches.add(List.of(texts));
    if (offlineQueries) throw StateError('embedder offline');
    return [
      for (final t in texts)
        t == badCue ? Float32List(dimension ~/ 2) : _real.embed(t),
    ];
  }
}

const _tz = MemoryTimezone('Asia/Tokyo', Duration(hours: 9));

/// Asserts the index holds exactly the indexed traces, with their vectors.
Future<void> expectInSync(
    EngramMemory memory, RecordingIndex index, Embedder embedder) async {
  final expected = {
    for (final m in await memory.memories())
      if (m.modelId == embedder.modelId &&
          m.vector.length == embedder.dimension)
        m.id: m.vector,
  };
  expect(index.vectors.keys.toSet(), expected.keys.toSet());
  for (final e in expected.entries) {
    expect(identical(index.vectors[e.key], e.value), isTrue,
        reason: 'index holds the trace\'s current vector');
  }
}

/// clusters() as 1.0.1 computed it: one embedding and one full scan per seed.
List<List<String>> referenceClusters(List<Memory> all,
    Float32List Function(Memory seed) cueVector, EngramConfig c, int budget) {
  final seeds = all.where((m) => !m.consolidated).take(8 * budget);
  final out = <List<String>>[];
  for (final seed in seeds) {
    final q = cueVector(seed);
    final near = <(double, int)>[
      for (var i = 0; i < all.length; i++)
        if (all[i].id != seed.id && all[i].createdAt <= seed.createdAt)
          for (final cos in [dot(all[i].vector, q)])
            if (cos >= c.thetaRelated) (cos, i),
    ]..sort(
        (a, b) => a.$1 == b.$1 ? a.$2.compareTo(b.$2) : b.$1.compareTo(a.$1));
    final cluster = [
      seed.text,
      for (final e in near.take(c.dreamMaxMembers - 1)) all[e.$2].text,
    ];
    if (cluster.length >= 2) out.add(cluster);
  }
  return out;
}

/// Forty related traces, most of them labile; cues repeat and some are empty.
Future<(EngramMemory, VirtualClock)> related(Embedder embedder,
    {VectorIndex? index}) async {
  final clock = VirtualClock();
  final memory = EngramMemory(
    store: InMemoryStore(),
    embedder: embedder,
    clock: clock.call,
    defaultTimezone: _tz,
    index: index,
  );
  await memory.initialize();
  const words = ['kyoto', 'trip', 'hotel', 'train', 'temple', 'autumn'];
  for (var i = 0; i < 40; i++) {
    clock.advance(3600);
    await memory.remember(
        '${words[i % 6]} ${words[(i + 1) % 6]} ${words[(i + 2) % 6]} n$i',
        cue: i % 4 == 0 ? '' : 'where ${words[i % 3]} ${words[(i + 3) % 6]}');
  }
  return (memory, clock);
}

void main() {
  group('DartVectorIndex', () {
    final a = l2Normalized([1, 2, 3, 4]);
    final b = l2Normalized([4, 3, 2, 1]);
    final c = l2Normalized([1, -1, 1, -1]);

    test('scores are dot, row-major by id then query', () {
      final index = DartVectorIndex()
        ..put('a', a)
        ..put('b', b);
      expect(index.length, 2);
      final s = index.scores(['b', 'a'], [a, c, b]);
      expect(s,
          [dot(b, a), dot(b, c), dot(b, b), dot(a, a), dot(a, c), dot(a, b)]);
      index.put('a', c); // replace
      expect(index.scores(['a'], [c]), [dot(c, c)]);
    });

    test('remove, clear, unknown ids and dimension mismatches', () {
      final index = DartVectorIndex()
        ..put('a', a)
        ..put('b', b)
        ..remove('a')
        ..remove('missing');
      expect(index.length, 1);
      expect(() => index.scores(['a'], [a]), throwsArgumentError);
      expect(() => index.scores(['b'], [Float32List(3)]), throwsArgumentError);
      expect(index.scores(['b'], const []), isEmpty);
      index.clear();
      expect(index.length, 0);
    });
  });

  group('the engine keeps its index in step', () {
    test('initialize, remember, recall, cite, forget, dream, reset', () async {
      final embedder = FakeEmbedder();
      final store = InMemoryStore();
      // A stale row (another model) and a valid one, loaded from the store.
      await store.put(Memory(
          id: 'stale',
          text: 'kyoto trip hotel stale',
          createdAt: 1600000000,
          tz: 'UTC;+00:00',
          lastRecall: 1600000000,
          stability: 86400,
          consolidated: true,
          modelId: 'old/model',
          vector: Float32List(8)));
      await store.put(Memory(
          id: 'valid',
          text: 'kyoto trip hotel valid',
          createdAt: 1600000000,
          tz: 'UTC;+00:00',
          lastRecall: 1600000000,
          stability: 86400,
          consolidated: true,
          modelId: embedder.modelId,
          vector: embedder.embed('kyoto trip hotel valid')));
      final index = RecordingIndex()..put('leftover', Float32List(64));
      final clock = VirtualClock();
      final memory = EngramMemory(
          store: store,
          embedder: embedder,
          clock: clock.call,
          defaultTimezone: _tz,
          index: index);
      await memory.initialize();
      expect(index.vectors.containsKey('leftover'), isFalse,
          reason: 'initialize starts from an empty index');
      expect(index.vectors.keys.toSet(), {'stale', 'valid'},
          reason: 'the stale row is re-embedded and indexed');
      await expectInSync(memory, index, embedder);

      for (final t in [
        'trip kyoto plan schedule note',
        'trip kyoto hotel schedule note',
        'trip kyoto food schedule note',
        'completely different topic zebra',
      ]) {
        clock.advance(60);
        await memory.remember(t, cue: 'kyoto trip');
        await expectInSync(memory, index, embedder);
      }
      final puts = index.puts;
      final recall = await memory.recall('kyoto trip');
      expect(recall.recalled, isNotEmpty);
      await memory
          .cite(recall.recalled.map((c) => '《id:${c.memory.id}》').join());
      await memory.remember('trip kyoto plan schedule note'); // rehearsal
      expect(index.puts, puts,
          reason: 'strength updates keep the vector: no re-put');
      await expectInSync(memory, index, embedder);

      expect(await memory.forget(recall.recalled.first.memory.id), isTrue);
      await expectInSync(memory, index, embedder);

      final reports = await memory.dream(adjudicate: mergeToGist);
      expect(reports.map((r) => r.action), contains(DreamAction.replace));
      await expectInSync(memory, index, embedder);

      await memory.reset();
      expect(index.vectors, isEmpty);
      await expectInSync(memory, index, embedder);
    });

    test('eviction removes the victims from the index', () async {
      final embedder = FakeEmbedder();
      final index = RecordingIndex();
      final clock = VirtualClock();
      final memory = EngramMemory(
          store: InMemoryStore(),
          embedder: embedder,
          config: const EngramConfig(capacity: 5, gracePeriod: 10),
          clock: clock.call,
          index: index);
      await memory.initialize();
      var evicted = 0;
      for (var i = 0; i < 20; i++) {
        clock.advance(3600);
        final r = await memory.remember('fact number $i about topic ${i % 3}');
        if (r.evicted != null) evicted++;
        await expectInSync(memory, index, embedder);
      }
      expect(evicted, 15);
      expect(index.vectors.length, 5);
    });

    test('an unindexed trace joins the index when it is re-embedded', () async {
      final embedder = GlitchingEmbedder();
      final index = RecordingIndex();
      final memory = EngramMemory(
          store: InMemoryStore(), embedder: embedder, index: index);
      await memory.initialize();
      await memory.remember('glitched on the way in');
      expect(index.vectors, isEmpty);
      await expectInSync(memory, index, embedder);
      await memory.dream(adjudicate: alwaysKeep); // re-indexes stale traces
      expect(index.vectors.length, 1);
      await expectInSync(memory, index, embedder);
    });

    test('an offline embedder indexes nothing', () async {
      final embedder = BrokenEmbedder();
      final index = RecordingIndex();
      final memory = EngramMemory(
          store: InMemoryStore(), embedder: embedder, index: index);
      await memory.initialize();
      await memory.remember('kept as text only');
      expect(index.vectors, isEmpty);
      expect((await memory.recall('text')).recalled, isEmpty);
      expect(index.scoreCalls, 0);
    });

    test('the seeded workload is identical through any VectorIndex', () async {
      final index = RecordingIndex();
      final trace = await runWorkload(
        create: (embedder, config, clock) => EngramMemory(
          store: InMemoryStore(),
          embedder: embedder,
          config: config,
          clock: clock,
          defaultTimezone: _tz,
          index: index,
        ),
        afterStep: (memory) => expectInSync(memory, index, FakeEmbedder()),
      );
      final golden =
          jsonDecode(File('test/equivalence/golden.json').readAsStringSync())
              as List;
      expect(trace, golden);
      expect(index.scoreCalls, greaterThan(0));
    });
  });

  group('cue embeddings are batched', () {
    test('clusters(): one embedQueries call, distinct non-empty cues in order',
        () async {
      final embedder = CountingEmbedder();
      final index = RecordingIndex();
      final (memory, _) = await related(embedder, index: index);
      final all = await memory.memories();
      final seeds = all.where((m) => !m.consolidated).take(80).toList();
      expect(seeds.length, greaterThan(10));
      embedder.queryBatches.clear();
      index.scoreCalls = 0;
      final clusters = await memory.clusters(budget: 10);
      expect(embedder.queryBatches.length, 1);
      final cues = <String>[];
      for (final s in seeds) {
        if (s.cue.isNotEmpty && !cues.contains(s.cue)) cues.add(s.cue);
      }
      expect(embedder.queryBatches.single, cues);
      expect(index.scoreCalls, 1, reason: 'one pass for every seed');
      final fake = FakeEmbedder();
      expect(
          [
            for (final c in clusters) [for (final m in c) m.text]
          ],
          referenceClusters(
              all,
              (s) => s.cue.isEmpty ? s.vector : l2Normalized(fake.embed(s.cue)),
              const EngramConfig(),
              10));
      expect(clusters, isNotEmpty);
    });

    test('dream(): one embedQueries call for the whole seed list', () async {
      final embedder = CountingEmbedder();
      final (memory, _) = await related(embedder);
      embedder.queryBatches.clear();
      final reports = await memory.dream(adjudicate: mergeToGist, budget: 10);
      expect(reports, isNotEmpty);
      expect(embedder.queryBatches.length, 1);
      expect(embedder.queryBatches.single.toSet().length,
          embedder.queryBatches.single.length,
          reason: 'deduplicated');
    });

    test('no cues, no embedQueries call', () async {
      final embedder = CountingEmbedder();
      final memory = EngramMemory(store: InMemoryStore(), embedder: embedder);
      await memory.initialize();
      for (final t in ['alpha beta gamma', 'alpha beta delta', 'alpha beta']) {
        await memory.remember(t);
      }
      embedder.queryBatches.clear();
      expect(await memory.clusters(), isNotEmpty);
      await memory.dream(adjudicate: alwaysKeep);
      expect(embedder.queryBatches, isEmpty);
    });

    test('offline queries: every seed falls back to its own vector', () async {
      final embedder = CountingEmbedder(offlineQueries: true);
      final (memory, _) = await related(embedder);
      final all = await memory.memories();
      embedder.queryBatches.clear();
      final clusters = await memory.clusters(budget: 10);
      expect(embedder.queryBatches.length, 1);
      expect([
        for (final c in clusters) [for (final m in c) m.text]
      ], referenceClusters(all, (s) => s.vector, const EngramConfig(), 10));
    });

    test('a wrong-dimension cue vector falls back for that cue only', () async {
      const bad = 'where kyoto train';
      final embedder = CountingEmbedder(badCue: bad);
      final (memory, _) = await related(embedder);
      final all = await memory.memories();
      expect(all.where((m) => m.cue == bad && !m.consolidated), isNotEmpty);
      final fake = FakeEmbedder();
      expect(
          [
            for (final c in await memory.clusters(budget: 10))
              [for (final m in c) m.text]
          ],
          referenceClusters(
              all,
              (s) => s.cue.isEmpty || s.cue == bad
                  ? s.vector
                  : l2Normalized(fake.embed(s.cue)),
              const EngramConfig(),
              10));
    });
  });

  test('a native index returning the wrong shape is an error, not a ranking',
      () async {
    final memory = EngramMemory(
        store: InMemoryStore(), embedder: FakeEmbedder(), index: _ShortIndex());
    await memory.initialize();
    await memory.remember('kyoto trip');
    await expectLater(memory.recall('kyoto'), throwsStateError);
  });
}

/// A broken native index: drops every score.
class _ShortIndex extends RecordingIndex {
  @override
  Float64List scores(List<String> ids, List<Float32List> queries) =>
      Float64List(0);
}
