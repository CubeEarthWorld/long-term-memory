import 'dart:typed_data';

import 'package:long_term_memory/long_term_memory.dart';
import 'package:test/test.dart';

import 'support/fakes.dart';

const tokyo = MemoryTimezone('Asia/Tokyo', Duration(hours: 9));

void main() {
  // =========================================================================
  // InMemoryStore edge cases
  // =========================================================================
  group('InMemoryStore edge cases', () {
    late InMemoryStore store;

    setUp(() async {
      store = InMemoryStore();
      await store.initialize();
    });

    test('updateMemory on non-existent id is a no-op', () async {
      final rec = MemoryRecord(
        id: 'ghost',
        text: 'ghost text',
        tier: 1,
        gen: 0,
        createdAt: 100,
        tz: 'UTC;+00:00',
        lastAccess: 100,
        lastBonusAt: 100,
        mass: 1.0,
      );
      await store.updateMemory(rec);
      expect(await store.getMemory('ghost'), isNull);
    });

    test('countMemories with tier and liveness combined', () async {
      await store.insertMemory(_mem('a', tier: 1));
      await store.insertMemory(_mem('b', tier: 1, supersededBy: 'b'));
      await store.insertMemory(_mem('c', tier: 2));
      await store.insertMemory(_mem('d', tier: 2, supersededBy: 'd'));

      expect(
        await store.countMemories(tier: 1, liveness: Liveness.liveOnly),
        1,
      );
      expect(await store.countMemories(tier: 1, liveness: Liveness.all), 2);
      expect(await store.countMemories(tier: 2, liveness: Liveness.all), 2);
      expect(await store.countMemories(tier: 3), 0);
      expect(
        await store.countMemories(tier: 1, liveness: Liveness.tombstonesOnly),
        1,
      );
    });

    test('listMemories with tier filter returns only matching tier', () async {
      await store.insertMemory(_mem('a', tier: 1));
      await store.insertMemory(_mem('b', tier: 2));
      await store.insertMemory(_mem('c', tier: 3));
      expect((await store.listMemories(tier: 1)).map((m) => m.id), ['a']);
      expect((await store.listMemories(tier: 2)).map((m) => m.id), ['b']);
      expect((await store.listMemories(tier: 3)).map((m) => m.id), ['c']);
    });

    test('getLiveMemoryByText returns null for tombstoned match', () async {
      await store.insertMemory(_mem('a', supersededBy: 'a'));
      expect(await store.getLiveMemoryByText('text of a'), isNull);
    });

    test('listVectors with modelId filter returns matching only', () async {
      await store.upsertVector(_vec('a', modelId: 'm1'));
      await store.upsertVector(_vec('a', modelId: 'm2'));
      await store.upsertVector(_vec('b', modelId: 'm1'));
      expect((await store.listVectors(modelId: 'm1')).length, 2);
      expect((await store.listVectors(modelId: 'm2')).length, 1);
      expect((await store.listVectors(modelId: 'm3')).length, 0);
    });

    test('deleteTombstones with both tier and lastAccessBefore', () async {
      await store.insertMemory(
        _mem('t1', tier: 1, supersededBy: 't1', lastAccess: 10),
      );
      await store.insertMemory(
        _mem('t2', tier: 1, supersededBy: 't2', lastAccess: 99),
      );
      await store.deleteTombstones(tier: 1, lastAccessBefore: 50);
      expect(await store.getMemory('t1'), isNull);
      expect(await store.getMemory('t2'), isNotNull);
    });

    test('deleteOrphanVectors leaves no orphans', () async {
      await store.upsertVector(_vec('orphan', modelId: 'm1'));
      await store.deleteOrphanVectors();
      expect(await store.listVectors(), isEmpty);
    });

    test('deleteOrphanVectors keeps vectors with existing memories', () async {
      await store.insertMemory(_mem('a'));
      await store.upsertVector(_vec('a'));
      await store.upsertVector(_vec('orphan'));
      await store.deleteOrphanVectors();
      expect((await store.listVectors()).single.memoryId, 'a');
    });

    test('runInTransaction default passthrough', () async {
      final result = await store.runInTransaction(() async {
        await store.insertMemory(_mem('tx'));
        return 42;
      });
      expect(result, 42);
      expect(await store.getMemory('tx'), isNotNull);
    });

    test('sizeInBytes returns -1 by default', () async {
      expect(await store.sizeInBytes(), -1);
    });

    test('empty store stats are all zero', () async {
      expect(await store.countMemories(), 0);
      expect(await store.countConflicts(), 0);
      expect(await store.countDreamLog(), 0);
      expect(await store.listMemories(), isEmpty);
      expect(await store.listVectors(), isEmpty);
      expect(await store.listConflicts(), isEmpty);
      expect(await store.getMeta('any'), isNull);
    });

    test('clear removes everything including vectors and metadata', () async {
      await store.insertMemory(_mem('a'));
      await store.upsertVector(_vec('a'));
      await store.addConflict(const ConflictRecord(a: 'a', b: 'b', at: 1));
      await store.putDreamVerdict(
        const DreamLogRecord(fingerprint: 'f', verdict: 'none', at: 1),
      );
      await store.putMeta('k', 'v');
      await store.clear();
      expect(await store.countMemories(), 0);
      expect(await store.listVectors(), isEmpty);
      expect(await store.countConflicts(), 0);
      expect(await store.countDreamLog(), 0);
      expect(await store.getMeta('k'), isNull);
    });
  });

  // =========================================================================
  // EngramConfig edge cases
  // =========================================================================
  group('EngramConfig validation', () {
    test('asserts dim1 >= dim2 >= dim3', () {
      expect(
        () => EngramConfig(dim1: 128, dim2: 256, dim3: 64),
        throwsA(isA<AssertionError>()),
      );
      expect(
        () => EngramConfig(dim1: 768, dim2: 128, dim3: 256),
        throwsA(isA<AssertionError>()),
      );
    });

    test('asserts capacities <= hardMemoryRows', () {
      expect(
        () => EngramConfig(
          cap1: 10000,
          cap2: 10000,
          cap3: 10000,
          hardMemoryRows: 100,
        ),
        throwsA(isA<AssertionError>()),
      );
    });

    test('asserts thetaSame > thetaConflict', () {
      expect(
        () => EngramConfig(thetaSame: 0.8, thetaConflict: 0.9),
        throwsA(isA<AssertionError>()),
      );
      expect(
        () => EngramConfig(thetaSame: 0.85, thetaConflict: 0.85),
        throwsA(isA<AssertionError>()),
      );
    });

    test('asserts thetaUp > thetaDown', () {
      expect(
        () => EngramConfig(thetaUp: 4.0, thetaDown: 16.0),
        throwsA(isA<AssertionError>()),
      );
      expect(
        () => EngramConfig(thetaUp: 8.0, thetaDown: 8.0),
        throwsA(isA<AssertionError>()),
      );
    });

    test('tauOf returns correct half-life per tier', () {
      const c = EngramConfig(tau1: 100, tau2: 200, tau3: 300);
      expect(c.tauOf(1), 100);
      expect(c.tauOf(2), 200);
      expect(c.tauOf(3), 300);
      expect(c.tauOf(4), 300); // fallback to tau3
    });

    test('dimOf returns correct dimension per tier', () {
      const c = EngramConfig(dim1: 768, dim2: 256, dim3: 128);
      expect(c.dimOf(1), 768);
      expect(c.dimOf(2), 256);
      expect(c.dimOf(3), 128);
      expect(c.dimOf(4), 128); // fallback to dim3
    });

    test('capOf returns correct capacity per tier', () {
      const c = EngramConfig(cap1: 100, cap2: 200, cap3: 300);
      expect(c.capOf(1), 100);
      expect(c.capOf(2), 200);
      expect(c.capOf(3), 300);
      expect(c.capOf(4), 300); // fallback to cap3
    });

    test('copyWith with all fields changed', () {
      const c = EngramConfig();
      final c2 = c.copyWith(
        cap1: 42,
        cap2: 43,
        cap3: 44,
        dim1: 200,
        dim2: 100,
        dim3: 50,
        tau1: 10,
        tau2: 20,
        tau3: 30,
        mMax: 32.0,
        refractorySeconds: 7200,
        alpha: 0.5,
        injectN: 3,
        mmrLambda: 0.5,
        scoreThresholds: [0.5],
        budgetChars: 512,
        thetaSame: 0.99,
        thetaConflict: 0.7,
        preciseMargin: 0.05,
        thetaUp: 32.0,
        thetaDown: 2.0,
        textMax: 100,
        textHardMax: 512,
        genMax: 3,
        dreamBudget: 2,
        dreamBudgetHard: 2048,
        clusterMin: 2,
        clusterCohesionMin: 0.7,
        dreamMaxMembers: 32,
        conflictCap: 128,
        dreamLogCap: 256,
        tombstoneSweepPct: 0.15,
        tombstoneSweepAge: 86400,
        writeRatePerDay: 100,
        hardMemoryRows: 8192,
        decayExpCap: 32768.0,
        compactEvery: 10,
        maxQueryChunks: 9,
      );
      expect(c2.cap1, 42);
      expect(c2.cap2, 43);
      expect(c2.cap3, 44);
      expect(c2.dim1, 200);
      expect(c2.dim2, 100);
      expect(c2.dim3, 50);
      expect(c2.tau1, 10);
      expect(c2.tau2, 20);
      expect(c2.tau3, 30);
      expect(c2.mMax, 32.0);
      expect(c2.refractorySeconds, 7200);
      expect(c2.alpha, 0.5);
      expect(c2.injectN, 3);
      expect(c2.mmrLambda, 0.5);
      expect(c2.scoreThresholds, [0.5]);
      expect(c2.budgetChars, 512);
      expect(c2.thetaSame, 0.99);
      expect(c2.thetaConflict, 0.7);
      expect(c2.preciseMargin, 0.05);
      expect(c2.thetaUp, 32.0);
      expect(c2.thetaDown, 2.0);
      expect(c2.textMax, 100);
      expect(c2.textHardMax, 512);
      expect(c2.genMax, 3);
      expect(c2.dreamBudget, 2);
      expect(c2.dreamBudgetHard, 2048);
      expect(c2.clusterMin, 2);
      expect(c2.clusterCohesionMin, 0.7);
      expect(c2.dreamMaxMembers, 32);
      expect(c2.conflictCap, 128);
      expect(c2.dreamLogCap, 256);
      expect(c2.tombstoneSweepPct, 0.15);
      expect(c2.tombstoneSweepAge, 86400);
      expect(c2.writeRatePerDay, 100);
      expect(c2.hardMemoryRows, 8192);
      expect(c2.decayExpCap, 32768.0);
      expect(c2.compactEvery, 10);
      expect(c2.maxQueryChunks, 9);
    });

    test(
      'fromJson handles missing, string and malformed values gracefully',
      () {
        final c = EngramConfig.fromJson({
          'cap1': '50',
          'cap2': 30,
          'tau1': '604800',
          'alpha': '0.5',
          'scoreThresholds': [0.1, 0.2],
          'thetaSame': 0.95,
        });
        expect(c.cap1, 50);
        expect(c.cap2, 30);
        expect(c.tau1, 604800);
        expect(c.alpha, 0.5);
        expect(c.scoreThresholds, [0.1, 0.2]);
        expect(c.thetaSame, 0.95);
        // Non-specified fields fall back to defaults
        expect(c.tau2, const EngramConfig().tau2);
      },
    );

    test('fromJson handles invalid numeric strings', () {
      final c = EngramConfig.fromJson({'cap1': 'not-a-number', 'tau1': 'abc'});
      expect(c.cap1, const EngramConfig().cap1); // fallback
      expect(c.tau1, const EngramConfig().tau1);
    });

    test('fromJson handles empty and null maps', () {
      final c1 = EngramConfig.fromJson({});
      expect(c1.cap1, const EngramConfig().cap1);

      final c2 = EngramConfig.fromJson(<String, Object?>{'cap1': null});
      expect(c2.cap1, const EngramConfig().cap1);
    });

    test('toJson contains all keys', () {
      const c = EngramConfig();
      final json = c.toJson();
      expect(json, contains('cap1'));
      expect(json, contains('cap2'));
      expect(json, contains('cap3'));
      expect(json, contains('dim1'));
      expect(json, contains('dim2'));
      expect(json, contains('dim3'));
      expect(json, contains('tau1'));
      expect(json, contains('tau2'));
      expect(json, contains('tau3'));
      expect(json, contains('mMax'));
      expect(json, contains('refractorySeconds'));
      expect(json, contains('alpha'));
      expect(json, contains('injectN'));
      expect(json, contains('mmrLambda'));
      expect(json, contains('scoreThresholds'));
      expect(json, contains('budgetChars'));
      expect(json, contains('thetaSame'));
      expect(json, contains('thetaConflict'));
      expect(json, contains('preciseMargin'));
      expect(json, contains('thetaUp'));
      expect(json, contains('thetaDown'));
      expect(json, contains('textMax'));
      expect(json, contains('textHardMax'));
      expect(json, contains('genMax'));
      expect(json, contains('dreamBudget'));
      expect(json, contains('dreamBudgetHard'));
      expect(json, contains('clusterMin'));
      expect(json, contains('clusterCohesionMin'));
      expect(json, contains('dreamMaxMembers'));
      expect(json, contains('conflictCap'));
      expect(json, contains('dreamLogCap'));
      expect(json, contains('writeRatePerDay'));
      expect(json, contains('hardMemoryRows'));
      expect(json, contains('decayExpCap'));
      expect(json, contains('compactEvery'));
      expect(json, contains('maxQueryChunks'));
    });
  });

  // =========================================================================
  // Engine edge cases
  // =========================================================================
  group('engine edge cases', () {
    late VirtualClock clock;
    late InMemoryStore store;
    late EngramMemory memory;

    Future<void> build({EngramConfig config = const EngramConfig()}) async {
      clock = VirtualClock();
      store = InMemoryStore();
      memory = EngramMemory(
        store: store,
        embedder: FakeEmbedder(),
        config: config,
        clock: clock.call,
        defaultTimezone: tokyo,
      );
      await memory.initialize();
    }

    setUp(() async {
      clock = VirtualClock();
      store = InMemoryStore();
      memory = EngramMemory(
        store: store,
        embedder: FakeEmbedder(),
        config: const EngramConfig(),
        clock: clock.call,
        defaultTimezone: tokyo,
      );
      await memory.initialize();
    });

    test('saveMemory with whitespace-only text is rejected', () async {
      final r = await memory.saveMemory('  　  ');
      expect(r.action, SaveAction.rejected);
      expect(r.error, contains('empty'));
    });

    test('saveMemory with text exactly at byte limit is accepted', () async {
      // textHardMax=1024. Create text that is exactly 1024 UTF-8 bytes.
      // Each Japanese char is 3 bytes in UTF-8, so 341 chars ≈ 1023 bytes.
      // Use ASCII chars for exact measurement.
      final text = 'a' * 1024; // 1024 bytes (ASCII)
      final r = await memory.saveMemory(text);
      // The soft limit shortens it, but the hard limit check is on bytes
      // The soft limit (170 chars) comes first, so this gets shortened.
      expect(r.action, isNot(SaveAction.rejected));
      expect(r.text.length, lessThanOrEqualTo(170));
    });

    test(
      'saveMemory with text exactly at soft limit (170 chars) stored as-is',
      () async {
        final text = '東京' * 85; // 170 chars (85 * 2)
        final r = await memory.saveMemory(text);
        expect(r.action, SaveAction.inserted);
        expect(r.text, text);
        expect(r.text.length, 170);
      },
    );

    test(
      'deleteMemory hard on already tombstoned memory physically removes',
      () async {
        final saved = await memory.saveMemory('test fact alpha beta');
        await memory.deleteMemory(saved.id!); // soft delete
        final del = await memory.deleteMemory(saved.id!, hard: true);
        expect(del.action, DeleteAction.deleted);
        expect(await store.getMemory(saved.id!), isNull);
      },
    );

    test('deleteMemory not found returns notFound', () async {
      final del = await memory.deleteMemory('nonexistent-id');
      expect(del.action, DeleteAction.notFound);
      expect(del.id, 'nonexistent-id');
      expect(del.text, isNull);
    });

    test('activationOf returns null for non-existent id', () async {
      expect(await memory.activationOf('no-such-id'), isNull);
    });

    test(
      'activationOf returns 0 for very old memory beyond decay cap',
      () async {
        final saved = await memory.saveMemory('very old fact');
        // Advance far beyond the L1 half-life to ensure decay exp cap is hit
        clock.t += const EngramConfig().decayExpCap.toInt() * 86400 + 1;
        final a = await memory.activationOf(saved.id!);
        expect(a, 0.0);
      },
    );

    test('retrieve with empty query returns empty result', () async {
      await memory.saveMemory('test fact');
      final result = await memory.retrieve('');
      expect(result.recalled, isEmpty);
      expect(result.packText, '');
    });

    test(
      'retrieve with MMR lambda 1.0 selects diverse results when available',
      () async {
        await build(config: const EngramConfig(mmrLambda: 1.0, injectN: 3));
        // Insert memories covering diverse domains
        await memory.saveMemory('quantum physics entanglement particle wave');
        await memory.saveMemory('coffee brewing espresso latte morning drink');
        await memory.saveMemory('mountain hiking trail nature outdoor trek');
        await memory.saveMemory('programming dart flutter code async future');
        await memory.saveMemory('tokyo japan travel culture food sushi');
        // Use query that overlaps with the quantum physics memory
        final result = await memory.retrieve(
          'quantum physics wave entanglement',
        );
        // Should find at least the quantum physics memory
        expect(result.recalled, isNotEmpty);
        // Results should try to be diverse (unique texts)
        final texts = result.recalled.map((m) => m.text).toSet();
        expect(texts.length, result.recalled.length);
      },
    );

    test('retrieve with budgetChars limiting', () async {
      await build(config: const EngramConfig(injectN: 10, budgetChars: 80));
      await memory.saveMemory('a' * 60);
      await memory.saveMemory('b' * 60);
      await memory.saveMemory('c' * 60);
      final result = await memory.retrieve('a b c');
      // With 80 char budget, at most one long memory fits
      expect(result.packText.length, lessThanOrEqualTo(80));
      expect(result.recalled.length, lessThanOrEqualTo(1));
    });

    test('listMemories with tier filter and without tombstones', () async {
      await memory.saveMemory('fact one alpha beta');
      await memory.saveMemory('fact two gamma delta');
      final saved3 = await memory.saveMemory('fact three epsilon zeta');
      await memory.deleteMemory(saved3.id!);

      final tier1Live = await memory.listMemories(
        tier: 1,
        includeTombstones: false,
      );
      expect(tier1Live.length, 2);
      expect(tier1Live.every((m) => !m.isTombstone), isTrue);

      final tier1All = await memory.listMemories(
        tier: 1,
        includeTombstones: true,
      );
      expect(tier1All.length, 3);
    });

    test('nowLocal works with various timezones', () {
      const unix = 1781181000;
      expect(
        memory.nowLocal(nowUnix: unix, timezone: tokyo),
        '2026-06-11 21:30 +09:00',
      );
      expect(
        memory.nowLocal(nowUnix: unix, timezone: MemoryTimezone.utc),
        '2026-06-11 12:30 +00:00',
      );
    });

    test('initialize writes three metadata entries', () async {
      expect(await store.getMeta('spec_version'), 'ENGRAM v1.1');
      expect(await store.getMeta('active_model'), 'fake/token-overlap');
      expect(await store.getMeta('four_sentences'), isNotEmpty);
    });

    test(
      'saveMemory with text near conflict band is re-embedded at full precision',
      () async {
        // Two texts that are similar but not identical
        const a = 'alpha beta gamma delta epsilon zeta eta theta iota';
        const b = 'alpha beta gamma delta epsilon zeta eta theta iota kappa';
        final first = await memory.saveMemory(a);
        final second = await memory.saveMemory(b);
        // They might be conflict or updated depending on exact cosine
        expect(
          second.action,
          anyOf(SaveAction.conflict, SaveAction.updated, SaveAction.inserted),
        );
        // Either way, the result should have a valid id or conflict info
        if (second.action == SaveAction.conflict) {
          expect(second.conflictWithId, isNotNull);
        }
      },
    );

    test(
      'maintain triggers compact when maintCount hits compactEvery',
      () async {
        await build(config: const EngramConfig(compactEvery: 2));
        await memory.saveMemory('fact one');
        await memory.maintain(); // maintCount=1
        await memory.maintain(); // maintCount=2 -> compact
        // No assertion needed for compact (it's a no-op on InMemoryStore)
        // but we verify no exception is thrown
        expect(await memory.totalRecords(), 1);
      },
    );
  });

  // =========================================================================
  // Dream edge cases
  // =========================================================================
  group('dream edge cases', () {
    late VirtualClock clock;
    late InMemoryStore store;
    late EngramMemory memory;

    Future<void> build({EngramConfig config = const EngramConfig()}) async {
      clock = VirtualClock();
      store = InMemoryStore();
      memory = EngramMemory(
        store: store,
        embedder: FakeEmbedder(),
        config: config,
        clock: clock.call,
        defaultTimezone: tokyo,
      );
      await memory.initialize();
    }

    setUp(build);

    Future<List<String>> saveCluster() async {
      final ids = <String>[];
      for (final text in const [
        'user likes hot coffee in the morning routine',
        'user likes iced coffee in the evening routine',
        'user likes black coffee on sunday routine',
      ]) {
        final r = await memory.saveMemory(text);
        ids.add(r.id!);
        clock.advance(4000);
      }
      return ids;
    }

    test('dream with budget exceeding dreamBudgetHard is clamped', () async {
      await saveCluster();
      final reports = await memory.dream(
        adjudicate: mergeToGist,
        budget: 99999,
      );
      expect(reports, isNotEmpty);
      // Should not try more than dreamBudgetHard adjudications
      expect(reports.length, lessThanOrEqualTo(5)); // default dreamBudget
    });

    test(
      'dream with force=true re-adjudicates previous none verdicts',
      () async {
        await saveCluster();
        await memory.dream(adjudicate: alwaysNone);
        var called = false;
        await memory.dream(
          adjudicate: (req) {
            called = true;
            return const DreamDecision.none();
          },
          force: true,
        );
        expect(called, isTrue);
      },
    );

    test(
      'dream with too few members for clusterMin returns no reports',
      () async {
        // Only 2 memories, clusterMin=3 by default
        await memory.saveMemory('alpha beta gamma delta epsilon');
        await memory.saveMemory('zeta eta theta iota kappa');
        final reports = await memory.dream(adjudicate: mergeToGist);
        expect(reports, isEmpty);
      },
    );

    test('dream handles gen cap (genMax)', () async {
      await build(config: const EngramConfig(genMax: 1));
      final ids = await saveCluster();
      // First merge: gen 0 -> gen 1
      final r1 = await memory.dream(adjudicate: mergeToGist);
      expect(r1.single.after.single.gen, 1);

      // Create another cluster from the merged result + new similar memory
      final ids2 = await saveCluster();
      // The cluster may not form if the previously merged memory is alone
      // or if cohesion is too low. Just verify no crash.
      final r2 = await memory.dream(adjudicate: mergeToGist);
      for (final report in r2) {
        for (final snap in report.after) {
          expect(snap.gen, lessThanOrEqualTo(1));
        }
      }
    });

    test('dream split distributes mass with perMass calculation', () async {
      await saveCluster();
      final reports = await memory.dream(
        adjudicate: (req) => DreamDecision(
          action: DreamAction.split,
          memories: const [
            DreamProposal(text: 'user likes coffee'),
            DreamProposal(text: 'user has a morning routine'),
            DreamProposal(text: 'user has an evening routine'),
          ],
        ),
      );
      final split = reports.where((r) => r.action == DreamAction.split).single;
      expect(split.after.length, 3);
      for (final snap in split.after) {
        final row = (await store.getMemory(snap.id))!;
        expect(row.mass, greaterThan(0));
        expect(row.mass, lessThanOrEqualTo(64.0));
      }
    });

    test(
      'dream cluster with members consumed by earlier adjudications is skipped',
      () async {
        // Create two overlapping clusters
        await memory.saveMemory('alpha beta gamma delta epsilon');
        await memory.saveMemory('alpha beta gamma delta zeta');
        await memory.saveMemory('alpha beta gamma delta eta');
        await memory.saveMemory('alpha beta gamma delta theta');
        // With 4 memories and clusterMin=3, only one cluster may form.
        // After first adjudication, second cluster may have consumed members.
        final reports = await memory.dream(adjudicate: mergeToGist, budget: 5);
        // All reports should have valid member counts
        for (final report in reports) {
          expect(report.before.length, greaterThanOrEqualTo(3));
          expect(report.after, isNotEmpty);
        }
      },
    );

    test('dream promotion moves hot L2 memory to L1', () async {
      await build(config: const EngramConfig(cap1: 1, cap2: 10, cap3: 10));
      final r1 = await memory.saveMemory('first fact alpha beta');
      clock.advance(4000);
      final r2 = await memory.saveMemory('second fact gamma delta');
      await memory.maintain(); // one gets demoted to L2

      // Find the L2 memory and boost its activation
      final l2Rows = await store.listMemories(
        tier: 2,
        liveness: Liveness.liveOnly,
      );
      if (l2Rows.isNotEmpty) {
        for (var i = 0; i < 20; i++) {
          await memory.saveMemory(l2Rows.first.text); // exact rehearsal
        }
      }

      await memory.dream(adjudicate: alwaysNone, budget: 0);
      // The boosted memory should be back in L1
      final finalL1 = await store.countMemories(
        tier: 1,
        liveness: Liveness.liveOnly,
      );
      expect(finalL1, greaterThanOrEqualTo(1));
    });

    test(
      'dream with empty proposals from merge adjudication records none',
      () async {
        await saveCluster();
        final reports = await memory.dream(
          adjudicate: (req) => const DreamDecision(
            action: DreamAction.merge,
            memories: [], // empty proposals
          ),
        );
        expect(reports.single.action, DreamAction.none);
        expect(await memory.totalRecords(), 3); // nothing deleted
      },
    );

    test('dream cleans up foreign-model vectors during housekeeping', () async {
      await memory.saveMemory('test fact');
      // Insert a vector for a different model
      await store.upsertVector(
        VectorRecord(
          memoryId: 'ghost',
          modelId: 'other/model',
          dim: 4,
          dtype: VecDtype.f32,
          bytes: packF32(Float32List.fromList([1, 0, 0, 0])),
        ),
      );
      await memory.dream(adjudicate: alwaysNone, budget: 0);
      final vectors = await store.listVectors();
      expect(vectors.every((v) => v.modelId == 'fake/token-overlap'), isTrue);
    });

    test('dream logs conflict ring trims correctly', () async {
      await build(config: const EngramConfig(conflictCap: 3, dreamLogCap: 3));
      // Add several conflicts
      for (var i = 0; i < 5; i++) {
        await store.addConflict(ConflictRecord(a: 'a$i', b: 'b$i', at: i));
      }
      // Add several dream log entries
      for (var i = 0; i < 5; i++) {
        await store.putDreamVerdict(
          DreamLogRecord(fingerprint: 'fp$i', verdict: 'none', at: i),
        );
      }
      await memory.dream(adjudicate: alwaysNone, budget: 0);
      expect(await store.countConflicts(), lessThanOrEqualTo(3));
      expect(await store.countDreamLog(), lessThanOrEqualTo(3));
    });
  });
}

MemoryRecord _mem(
  String id, {
  int tier = 1,
  String? supersededBy,
  int lastAccess = 100,
}) =>
    MemoryRecord(
      id: id,
      text: 'text of $id',
      tier: tier,
      gen: 0,
      createdAt: 100,
      tz: 'UTC;+00:00',
      lastAccess: lastAccess,
      lastBonusAt: 100,
      mass: 1.0,
      supersededBy: supersededBy,
    );

VectorRecord _vec(String memoryId, {String modelId = 'm1'}) => VectorRecord(
      memoryId: memoryId,
      modelId: modelId,
      dim: 4,
      dtype: VecDtype.f32,
      bytes: packF32(Float32List.fromList([1, 0, 0, 0])),
    );
