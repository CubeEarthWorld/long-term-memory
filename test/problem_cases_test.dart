// Problem-prone edge cases ("問題が発生しそうなケース").
//
// These target boundaries the happy-path tests don't exercise: Unicode that
// straddles the soft char limit, configuration footguns that silently empty
// the recall, the serialization guarantee under concurrent calls, mass
// arithmetic on fully-decayed clusters, and timestamp/identifier extremes.
import 'dart:convert';
import 'dart:math';

import 'package:long_term_memory/long_term_memory.dart';
import 'package:test/test.dart';

import 'support/fakes.dart';

const tokyo = MemoryTimezone('Asia/Tokyo', Duration(hours: 9));

Future<(VirtualClock, InMemoryStore, EngramMemory)> newEngine([
  EngramConfig config = const EngramConfig(),
]) async {
  final clock = VirtualClock();
  final store = InMemoryStore();
  final memory = EngramMemory(
    store: store,
    embedder: FakeEmbedder(),
    config: config,
    clock: clock.call,
    defaultTimezone: tokyo,
  );
  await memory.initialize();
  return (clock, store, memory);
}

void main() {
  // =========================================================================
  // Unicode / text-boundary problems
  // =========================================================================
  group('Unicode boundary problems', () {
    test(
      'shorten splits an emoji surrogate pair at an odd char boundary '
      '(documents the limitation, must not throw)',
      () {
        // 'a' + 85 emoji = 1 + 170 = 171 UTF-16 code units. The soft limit is
        // 170, so the 85th emoji is cut in half: substring(0,170) keeps its
        // high surrogate but drops the low one.
        final text = 'a${'😀' * 85}';
        expect(text.length, 171);

        late final String out;
        expect(() => out = shorten(text, 170), returnsNormally);
        expect(out.length, lessThanOrEqualTo(170));

        final lastUnit = out.codeUnitAt(out.length - 1);
        final isLoneHighSurrogate = lastUnit >= 0xD800 && lastUnit <= 0xDBFF;
        // This is the "problem": shorten is grapheme-unaware, so it can leave a
        // dangling high surrogate. We lock the behaviour so a future fix is a
        // conscious choice, not an accident.
        expect(isLoneHighSurrogate, isTrue,
            reason: 'shorten cut through a surrogate pair');

        // The malformed string still encodes (U+FFFD replacement), never throws.
        expect(() => utf8.encode(out), returnsNormally);
      },
    );

    test('saveMemory with emoji at the soft limit does not crash', () async {
      final (_, _, memory) = await newEngine();
      final r = await memory.saveMemory('a${'😀' * 85}');
      expect(r.action, isNot(SaveAction.rejected));
      expect(utf8.encode(r.text).length, lessThanOrEqualTo(1024));
    });

    test(
      'two different long texts that shorten to the same prefix collapse to '
      'one reinforced memory',
      () async {
        final (_, _, memory) = await newEngine();
        final t1 = List.filled(60, 'alpha').join(' '); // 359 chars
        final t2 = List.filled(80, 'alpha').join(' '); // 479 chars
        final r1 = await memory.saveMemory(t1);
        final r2 = await memory.saveMemory(t2);

        expect(r1.action, SaveAction.inserted);
        // Both shorten to the identical 170-char prefix, so the second write is
        // an exact-text rehearsal — a single stored row, mass reinforced.
        expect(r2.action, SaveAction.reinforced);
        expect(r2.id, r1.id);
        expect(await memory.totalRecords(), 1);
      },
    );
  });

  // =========================================================================
  // Configuration footguns that silently empty the recall
  // =========================================================================
  group('retrieval config footguns', () {
    test('empty scoreThresholds is rejected at initialize (fail fast)',
        () async {
      final memory = EngramMemory(
        store: InMemoryStore(),
        embedder: FakeEmbedder(),
        config: EngramConfig(scoreThresholds: const []),
        clock: VirtualClock().call,
        defaultTimezone: tokyo,
      );
      // Was a silent empty recall; now a loud ArgumentError before any use.
      await expectLater(memory.initialize(), throwsArgumentError);
    });

    test('budgetChars smaller than one line still returns the top match',
        () async {
      final (_, _, memory) = await newEngine(
        const EngramConfig(budgetChars: 1, injectN: 5),
      );
      await memory.saveMemory('alpha beta gamma delta epsilon');
      final r = await memory.retrieve('alpha beta gamma delta epsilon');
      // The top-ranked match is always packed (soft budget), so a clear hit is
      // never silently dropped — and it still receives a recall update.
      expect(r.recalled.length, 1);
      expect(r.recalled.single.text, 'alpha beta gamma delta epsilon');
      expect(r.packText, isNotEmpty);
    });

    test('writeRatePerDay of 0 rate-limits the very first write', () async {
      final (_, _, memory) = await newEngine(
        const EngramConfig(writeRatePerDay: 0),
      );
      final r = await memory.saveMemory('anything at all');
      expect(r.action, SaveAction.rateLimited);
      expect(await memory.totalRecords(), 0);
    });

    test(
      'query past the chunk cap is ignored by default but recoverable via '
      'maxQueryChunks',
      () async {
        const target = 'phlogiston quokka aardvark zugzwang';

        String queryWithMatchAtLine(int line) {
          final lines = [
            for (var i = 1; i <= 20; i++)
              i == line ? target : 'unrelated filler clause number $i here',
          ];
          return lines.join('\n');
        }

        // Default maxQueryChunks = 16. Threshold 0.3 cleanly separates the
        // exact-match chunk (cos≈1) from unrelated filler (cos well below 0.3).
        final (_, _, m1) = await newEngine(
          const EngramConfig(scoreThresholds: [0.3]),
        );
        await m1.saveMemory(target);
        // Control: a match in chunk 1 is found.
        expect(
          (await m1.retrieve(queryWithMatchAtLine(1)))
              .recalled
              .map((m) => m.text),
          contains(target),
        );
        // Default cap drops chunk 20.
        expect(
          (await m1.retrieve(queryWithMatchAtLine(20))).recalled,
          isEmpty,
        );

        // Raising the (now configurable) cap recovers the late-placed match.
        final (_, _, m2) = await newEngine(
          const EngramConfig(scoreThresholds: [0.3], maxQueryChunks: 20),
        );
        await m2.saveMemory(target);
        expect(
          (await m2.retrieve(queryWithMatchAtLine(20)))
              .recalled
              .map((m) => m.text),
          contains(target),
        );
      },
    );
  });

  // =========================================================================
  // Concurrency / serialization guarantees
  // =========================================================================
  group('concurrent call serialization', () {
    test('concurrent identical saves produce exactly one row (no race)',
        () async {
      final (_, _, memory) = await newEngine();
      final results = await Future.wait([
        for (var i = 0; i < 10; i++) memory.saveMemory('one exact shared fact'),
      ]);
      // Serialized writes: first inserts, the other nine are rehearsals.
      expect(results.where((r) => r.action == SaveAction.inserted).length, 1);
      expect(
        results.where((r) => r.action == SaveAction.reinforced).length,
        9,
      );
      expect(await memory.totalRecords(), 1);
    });

    test('concurrent distinct saves all persist with unique ids', () async {
      final (_, _, memory) = await newEngine();
      final results = await Future.wait([
        // Per-index tokens share nothing → cosine ≈ 0 → all inserted.
        for (var i = 0; i < 20; i++)
          memory.saveMemory('alpha$i beta$i gamma$i delta$i'),
      ]);
      expect(results.every((r) => r.action == SaveAction.inserted), isTrue);
      expect(results.map((r) => r.id).toSet().length, 20);
      expect(await memory.totalRecords(), 20);
    });

    test('interleaved save/retrieve/delete/maintain never corrupts the engine',
        () async {
      final (_, _, memory) = await newEngine();
      await memory.saveMemory('seed alpha beta gamma');
      final id = (await memory.listMemories()).first.id;
      await expectLater(
        Future.wait<Object?>([
          memory.saveMemory('concurrent one delta epsilon'),
          memory.retrieve('alpha'),
          memory.saveMemory('concurrent two zeta eta'),
          memory.deleteMemory(id),
          memory.maintain(),
        ]),
        completes,
      );
    });
  });

  // =========================================================================
  // Decay / consolidation arithmetic at extremes
  // =========================================================================
  group('decay and consolidation extremes', () {
    Future<List<String>> saveCluster(VirtualClock clock, EngramMemory m) async {
      final ids = <String>[];
      for (final text in const [
        'user likes hot coffee in the morning routine',
        'user likes iced coffee in the evening routine',
        'user likes black coffee on sunday routine',
      ]) {
        final r = await m.saveMemory(text);
        ids.add(r.id!);
        clock.advance(4000);
      }
      return ids;
    }

    test(
      'merging a fully-decayed cluster floors the new mass at 1.0 (no silent '
      'data loss)',
      () async {
        final (clock, store, memory) = await newEngine();
        await saveCluster(clock, memory);
        // Decay everything to exactly 0 activation (double underflow).
        clock.advanceDays(60000);

        final reports = await memory.dream(adjudicate: mergeToGist);
        final merged = reports.firstWhere((r) => r.action == DreamAction.merge);
        expect(merged.after, isNotEmpty);

        final row = (await store.getMemory(merged.after.single.id))!;
        // sumA == 0, but the floor guarantees the consolidated gist is born at
        // least as alive as a fresh write (mass 1.0) instead of mass 0 — so it
        // is recallable rather than an instant eviction victim.
        expect(row.mass, 1.0);
        expect(await memory.activationOf(row.id), 1.0);
      },
    );

    test(
      'activationOf reports a future-dated row at full mass (no sanitation '
      'on the read path)',
      () async {
        final (clock, store, memory) = await newEngine();
        // Inject a row whose lastAccess is in the future relative to the clock.
        await store.insertMemory(MemoryRecord(
          id: 'future',
          text: 'time-traveller',
          tier: 1,
          gen: 0,
          createdAt: clock.t,
          tz: tokyo.storageField,
          lastAccess: clock.t + 1000000,
          lastBonusAt: clock.t,
          mass: 5.0,
        ));
        // max(0, now-lastAccess) clamps Δt to 0, so no decay is applied — and
        // activationOf does not run clampFutureTimestamps first.
        expect(await memory.activationOf('future'), 5.0);
      },
    );

    test('cap1 of 0 demotes every L1 row to L2 on maintain (no infinite loop)',
        () async {
      final (_, _, memory) = await newEngine(
        const EngramConfig(cap1: 0, cap2: 3000, cap3: 6000),
      );
      await memory.saveMemory('apple orange banana');
      await memory.saveMemory('desk lamp window');
      await memory.saveMemory('river mountain forest');
      expect((await memory.listMemories(tier: 1)).length, 3);

      await memory.maintain();

      expect(await memory.listMemories(tier: 1), isEmpty);
      expect((await memory.listMemories(tier: 2)).length, 3);
      expect(await memory.totalRecords(), 3);
    });

    test('negative dream budget runs housekeeping only, mutates nothing',
        () async {
      final (clock, _, memory) = await newEngine();
      await saveCluster(clock, memory);
      final reports = await memory.dream(adjudicate: mergeToGist, budget: -10);
      expect(reports, isEmpty);
      expect(await memory.totalRecords(), 3);
    });
  });

  // =========================================================================
  // Timestamp / identifier extremes
  // =========================================================================
  group('time and id extremes', () {
    test('formatLocal renders 5-digit years (>9999) via pure arithmetic', () {
      final s = formatLocal(300000000000, 'UTC;+00:00'); // ~year 11476
      expect(
        RegExp(r'^\d{4,}-\d{2}-\d{2} \d{2}:\d{2} \+00:00$').hasMatch(s),
        isTrue,
        reason: s,
      );
      final year = int.parse(s.split('-').first);
      expect(year, greaterThan(9999));
    });

    test('ULID clamps a negative (pre-epoch) timestamp to all-zero time bits',
        () {
      final g = UlidGenerator(millis: () => -123, random: Random(1));
      final id = g.next();
      expect(id.length, 26);
      expect(id.substring(0, 10), '0000000000');
    });

    test('InMemoryStore.fromJson on an empty map yields an empty store', () async {
      final store = InMemoryStore.fromJson(<String, Object?>{});
      await store.initialize();
      expect(await store.countMemories(), 0);
      expect(await store.listVectors(), isEmpty);
      expect(await store.countConflicts(), 0);
      expect(await store.countDreamLog(), 0);
      expect(await store.getMeta('anything'), isNull);
    });
  });
}
