import 'dart:typed_data';

import 'package:long_term_memory/long_term_memory.dart';
import 'package:test/test.dart';

import 'support/fakes.dart';

void main() {
  group('remember', () {
    test('inserts, rehearses exact duplicates, rejects empty', () async {
      final (memory, clock, _) = await build();
      final a = await memory.remember('user lives in kyoto');
      expect(a.action, RememberAction.inserted);
      expect(a.memory!.stability, 86400);
      expect(a.memory!.consolidated, isTrue, reason: 'no neighbour');
      clock.advanceDays(2);
      final b = await memory.remember('  user  lives in kyoto ');
      expect(b.action, RememberAction.reinforced);
      expect(b.memory!.id, a.memory!.id);
      expect(b.memory!.stability, greaterThan(86400));
      expect((await memory.remember('   ')).action, RememberAction.rejected);
      expect((await memory.memories()).length, 1);
    });

    test('a wrong-dimension vector is treated as stale', () async {
      // One modelId means one dimension: a vector of any other length must not
      // reach dot(), which throws, and must be re-embedded instead (SPEC §7).
      final emb = GlitchingEmbedder();
      final (memory, _, _) = await build(embedder: emb);
      await memory.remember('a trace the embedder mis-embedded');
      await memory.remember('user lives in kyoto');   // would throw before
      expect((await memory.recall('kyoto')).recalled, isNotEmpty);
      final before = await memory.memories();
      expect(before.where((m) => m.modelId.isEmpty).length, 1,
          reason: 'the mis-embedded trace is unindexed, not poisoning search');
      await memory.initialize();                      // reindex re-embeds it
      for (final m in await memory.memories()) {
        expect(m.vector.length, emb.dimension);
        expect(m.modelId, emb.modelId);
      }
    });

    test('a row with no text is skipped on load', () async {
      final store = InMemoryStore();
      await store.put(Memory(
        id: '01EMPTY',
        text: '',
        createdAt: 1700000000,
        tz: 'UTC;+00:00',
        lastRecall: 1700000000,
        stability: 86400,
        consolidated: true,
        modelId: FakeEmbedder().modelId,
        vector: Float32List(64),
      ));
      final (memory, _, _) = await build(store: store);
      expect(await memory.memories(), isEmpty);
    });

    test('cleans delimiters, whitespace and length', () async {
      final (memory, _, _) =
          await build(config: const EngramConfig(textMax: 20));
      final r = await memory.remember('a《id:x》b\n\n c ${'x' * 40}');
      expect(r.memory!.text, isNot(contains('《')));
      expect(r.memory!.text, isNot(contains('\n')));
      expect(r.memory!.text.length, lessThanOrEqualTo(20));
    });

    test('a write that reactivates the past is born labile', () async {
      final (memory, _, _) = await build();
      final old = (await memory.remember('cat dog bird fish')).memory!;
      expect(old.consolidated, isTrue, reason: 'nothing to reactivate');
      final r = await memory.remember('cat dog bird fish!');
      expect(r.cosine, greaterThan(0.9));
      expect(r.memory!.consolidated, isFalse);
      expect((await memory.memory(old.id))!.consolidated, isTrue,
          reason: 'the old trace is a candidate, not rewritten on the way in');
      expect((await memory.memories()).length, 2, reason: 'never overwritten');
      final cluster = (await memory.clusters()).single;
      expect(cluster.first.id, r.memory!.id, reason: 'newest evidence seeds');
      expect(cluster.last.id, old.id);
    });

    test('salience scales and clamps initial stability', () async {
      final (memory, _, _) = await build();
      final s3 = (await memory.remember('alpha', salience: 3)).memory!;
      expect(s3.stability, 3 * 86400);
      final huge = (await memory.remember('beta', salience: 1e9)).memory!;
      expect(huge.stability, const EngramConfig().maxStability / 365);
      final zero = (await memory.remember('gamma', salience: 0)).memory!;
      expect(zero.stability, 1.0);
      expect(
          memory.retrievability(zero, zero.lastRecall + 100), lessThan(1e-20));
    });

    test('daily write limit', () async {
      final (memory, clock, _) =
          await build(config: const EngramConfig(writesPerDay: 2));
      await memory.remember('a1 b1');
      await memory.remember('a2 b2');
      expect(
          (await memory.remember('a3 b3')).action, RememberAction.rateLimited);
      clock.advanceDays(1.01);
      expect((await memory.remember('a3 b3')).action, RememberAction.inserted);
    });

    test('embedder offline keeps the text and indexes it later', () async {
      final store = InMemoryStore();
      final (broken, _, _) =
          await build(embedder: BrokenEmbedder(), store: store);
      final r = await broken.remember('kept while offline');
      expect(r.action, RememberAction.inserted);
      expect((await broken.recall('kept while offline')).recalled, isEmpty);
      final (healed, _, _) = await build(
          embedder: FakeEmbedder(modelId: 'fake/broken'), store: store);
      final rec = await healed.recall('kept while offline');
      expect(rec.recalled.single.memory.text, 'kept while offline');
    });
  });

  group('capacity', () {
    test('evicts the weakest old trace, never a fresh one', () async {
      final (memory, clock, _) = await build(
          config: const EngramConfig(capacity: 3, gracePeriod: 86400));
      await memory.remember('old weak one');
      final strong =
          (await memory.remember('old strong two', salience: 10)).memory!;
      clock.advanceDays(2);
      await memory.remember('fresh three');
      final r = await memory.remember('fresh four');
      expect(r.evicted!.text, 'old weak one');
      expect((await memory.memory(strong.id)), isNotNull);
    });

    test('a flood evicts its own members past a tenth of capacity', () async {
      final (memory, _, _) = await build(
          config: const EngramConfig(capacity: 20, gracePeriod: 1e9));
      final keep = (await memory.remember('precious', salience: 10)).memory!;
      for (var i = 0; i < 40; i++) {
        await memory.remember('junk $i x$i');
      }
      expect((await memory.memories()).length, 20);
      expect(await memory.memory(keep.id), isNotNull);
    });

    test('everything young: weakest young goes', () async {
      final (memory, _, _) =
          await build(config: const EngramConfig(capacity: 2));
      await memory.remember('one', salience: 2);
      await memory.remember('two', salience: 3);
      final r = await memory.remember('three', salience: 1);
      expect(r.evicted!.text, 'three');
    });
  });

  group('recall', () {
    test('packs relevant traces with header and id, strengthens them',
        () async {
      final (memory, clock, _) = await build();
      final cat = (await memory.remember('user has a cat named tama')).memory!;
      await memory.remember('user works at a bank');
      clock.advanceDays(1);
      final r = await memory.recall('tell me about the cat tama');
      expect(r.recalled.first.memory.id, cat.id);
      expect(
          r.packText,
          startsWith(
              '[${cat.createdAt} Asia/Tokyo;+09:00] user has a cat named tama　《id:${cat.id}》\n'));
      final after = (await memory.memory(cat.id))!;
      expect(after.stability, greaterThan(cat.stability));
      expect(after.lastRecall, clock.t);
    });

    test('injects nothing when nothing matches', () async {
      final (memory, _, _) =
          await build(config: const EngramConfig(minScore: 0.25));
      await memory.remember('alpha beta gamma');
      final r = await memory.recall('zzz yyy');
      expect(r.recalled, isEmpty);
    });

    test('relative cut drops the noisy tail', () async {
      final (memory, _, _) = await build();
      await memory.remember('kyoto trip hotel booking');
      await memory.remember('kyoto weather');
      await memory.remember('random unrelated note');
      final r = await memory.recall('kyoto trip hotel booking');
      expect(r.recalled.map((c) => c.memory.text),
          isNot(contains('random unrelated note')));
    });

    test('multi-cue queries surface traces for each part', () async {
      final (memory, _, _) = await build();
      await memory.remember('user likes matcha ice cream');
      await memory.remember('user plays tennis on sunday');
      final r = await memory
          .recall('matcha ice cream is great.\nalso tennis on sunday?');
      expect(r.recalled.length, 2);
    });

    test('respects the character budget after the first line', () async {
      final (memory, _, _) =
          await build(config: const EngramConfig(budgetChars: 120));
      for (var i = 0; i < 4; i++) {
        await memory.remember('kyoto note number $i about kyoto');
      }
      final r = await memory.recall('kyoto note');
      expect(r.recalled.length, 1);
    });

    test('cite completes the strengthening of used traces', () async {
      final (memory, clock, _) = await build();
      final m = (await memory.remember('user birthday is 1990-05-03')).memory!;
      clock.advanceDays(3);
      final r = await memory.recall('when is the birthday?');
      final half = (await memory.memory(m.id))!.stability;
      final ids = await memory.cite('It is on 1990-05-03 《id:${m.id}》');
      expect(ids, [m.id]);
      final full = (await memory.memory(m.id))!.stability;
      expect(full, greaterThan(half));
      expect(await memory.cite('again 《id:${m.id}》'), isEmpty,
          reason: 'one turn only');
      expect(r.recalled.single.cosine, greaterThan(0));
    });
  });

  test('forget by id', () async {
    final (memory, _, _) = await build();
    final m = (await memory.remember('to be forgotten')).memory!;
    expect(await memory.forget(m.id), isTrue);
    expect(await memory.forget(m.id), isFalse);
    expect(await memory.memories(), isEmpty);
  });

  test('initialize clamps future timestamps and stability', () async {
    final store = InMemoryStore();
    final (memory, clock, _) = await build(store: store);
    final m = (await memory.remember('future')).memory!;
    await store.put(m.copyWith(
        lastRecall: clock.t + 1000000,
        createdAt: clock.t + 1000000,
        stability: 1e12));
    final (reopened, _, _) = await build(store: store);
    final loaded = (await reopened.memories()).single;
    expect(loaded.lastRecall, lessThanOrEqualTo(clock.t));
    expect(loaded.stability, const EngramConfig().maxStability);
  });

  test('operations are serialized', () async {
    final (memory, _, _) = await build();
    await Future.wait([
      for (var i = 0; i < 20; i++) memory.remember('parallel $i'),
    ]);
    expect((await memory.memories()).length, 20);
  });
}
