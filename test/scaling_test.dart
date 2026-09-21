import 'dart:math' as math;
import 'dart:typed_data';

import 'package:long_term_memory/long_term_memory.dart';
import 'package:test/test.dart';

import 'support/fakes.dart';

class RecordingStore extends InMemoryStore {
  final removed = <String>[];
  @override
  Future<void> remove(String id) async {
    removed.add(id);
    await super.remove(id);
  }
}

void main() {
  for (final capacity in [1, 10, 40, 119, 120]) {
    for (final mode in ['mixed', 'young', 'old']) {
      test('bulk eviction matches repeated scan: $capacity $mode', () async {
        const now = 1700000000;
        final store = RecordingStore();
        for (var i = 0; i < 120; i++) {
          final young = mode == 'young' || (mode == 'mixed' && i % 3 != 0);
          await store.put(Memory(
            id: '$i',
            text: 'row $i',
            createdAt: now - (young ? 10 : 200),
            tz: 'UTC;+00:00',
            lastRecall: now - (i % 5) * 100,
            stability: math.pow(2, i % 8).toDouble(),
            consolidated: true,
            modelId: '',
            vector: Float32List(0),
          ));
        }
        final (memory, _, _) = await build(
            store: store,
            embedder: BrokenEmbedder(),
            config: EngramConfig(capacity: capacity, gracePeriod: 100));
        final rows = [
          ...await memory.memories(),
          Memory(
            id: 'new',
            text: 'new row',
            createdAt: now,
            tz: 'UTC;+00:00',
            lastRecall: now,
            stability: 86400,
            consolidated: false,
            modelId: '',
            vector: Float32List(0),
          )
        ];
        final expected = <String>[];
        double rank(Memory m) =>
            math.log(m.stability) / math.ln2 -
            math.max(0, now - m.lastRecall) / m.stability;
        // Reference: rescan all survivors after every eviction.
        while (rows.length > capacity) {
          final young = rows
              .where((m) => now - m.createdAt >= 0 && now - m.createdAt < 100)
              .toList();
          final old = rows.where((m) => !young.contains(m)).toList();
          final eligible = old.isEmpty || young.length > capacity ~/ 10
              ? [...old, ...young]
              : old;
          final victim = eligible.reduce((a, b) => rank(b) < rank(a) ? b : a);
          expected.add(victim.id);
          rows.remove(victim);
        }
        final result = await memory.remember('new row');
        expect(store.removed, expected);
        expect(result.evicted!.id, expected.last);
        expect((await memory.memories()).map((m) => m.text),
            rows.map((m) => m.text));
      });
    }
  }

  test('clusters(budget:) previews the seeds dream scans, first in first out',
      () async {
    final store = InMemoryStore();
    final embedder = FakeEmbedder();
    for (var i = 0; i < 100; i++) {
      await store.put(Memory(
        id: '$i',
        text: 'row $i',
        createdAt: 1700000000 - (i % 7),
        tz: 'UTC;+00:00',
        lastRecall: 1700000000,
        stability: 86400.0 + i % 3,
        consolidated: false,
        modelId: embedder.modelId,
        vector: embedder.embed('same vector'),
      ));
    }
    final (memory, _, _) = await build(store: store, embedder: embedder);
    List<String> seeds(List<List<Memory>> cs) => [for (final c in cs) c[0].id];
    final fifo = [for (var i = 0; i < 100; i++) '$i'];
    expect(seeds(await memory.clusters()), fifo.take(8 * 5));
    expect(seeds(await memory.clusters(budget: 0)), isEmpty);
    expect(seeds(await memory.clusters(budget: 2)), fifo.take(16));
    expect(seeds(await memory.clusters(budget: 100)), fifo);
  });
}
