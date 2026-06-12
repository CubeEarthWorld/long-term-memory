import 'dart:typed_data';

import 'package:long_term_memory/long_term_memory.dart';
import 'package:test/test.dart';

void main() {
  group('MemoryRecord', () {
    test('copyWith preserves all fields and applies changes', () {
      const original = MemoryRecord(
        id: '01ABC',
        text: 'ユーザーは京都に住んでいる',
        tier: 1,
        gen: 0,
        createdAt: 1700000000,
        tz: 'Asia/Tokyo;+09:00',
        lastAccess: 1700000100,
        lastBonusAt: 1700000100,
        mass: 2.5,
      );
      final updated = original.copyWith(
        tier: 2,
        mass: 5.0,
        lastAccess: 1700000200,
      );
      expect(updated.id, original.id);
      expect(updated.text, original.text);
      expect(updated.tier, 2);
      expect(updated.gen, 0);
      expect(updated.createdAt, original.createdAt);
      expect(updated.tz, original.tz);
      expect(updated.lastAccess, 1700000200);
      expect(updated.lastBonusAt, original.lastBonusAt);
      expect(updated.mass, 5.0);
      expect(updated.supersededBy, isNull);
    });

    test('copyWith clearSupersededBy resets tombstone', () {
      const original = MemoryRecord(
        id: '01XYZ',
        text: 'fact',
        tier: 1,
        gen: 0,
        createdAt: 100,
        tz: 'UTC;+00:00',
        lastAccess: 100,
        lastBonusAt: 100,
        mass: 1.0,
        supersededBy: '01OLD',
      );
      final cleared = original.copyWith(clearSupersededBy: true);
      expect(cleared.supersededBy, isNull);
      expect(cleared.isTombstone, false);
    });

    test(
      'copyWith can override supersededBy and clearSupersededBy together',
      () {
        const original = MemoryRecord(
          id: '01ABC',
          text: 'fact',
          tier: 1,
          gen: 0,
          createdAt: 100,
          tz: 'UTC;+00:00',
          lastAccess: 100,
          lastBonusAt: 100,
          mass: 1.0,
          supersededBy: 'old',
        );
        // clearSupersededBy takes priority
        final result = original.copyWith(
          supersededBy: 'new',
          clearSupersededBy: true,
        );
        expect(result.supersededBy, isNull);
      },
    );

    test('isTombstone true when supersededBy is non-null', () {
      const live = MemoryRecord(
        id: '01A',
        text: 'fact',
        tier: 1,
        gen: 0,
        createdAt: 100,
        tz: 'UTC;+00:00',
        lastAccess: 100,
        lastBonusAt: 100,
        mass: 1.0,
      );
      expect(live.isTombstone, false);
      const dead = MemoryRecord(
        id: '01B',
        text: 'fact',
        tier: 1,
        gen: 0,
        createdAt: 100,
        tz: 'UTC;+00:00',
        lastAccess: 100,
        lastBonusAt: 100,
        mass: 1.0,
        supersededBy: '01B',
      );
      expect(dead.isTombstone, true);
      // self-referential tombstone
      expect(dead.supersededBy, dead.id);
    });

    test('toJson/fromJson round-trip preserves all fields', () {
      const original = MemoryRecord(
        id: '01TEST',
        text: 'テスト命題',
        tier: 2,
        gen: 3,
        createdAt: 1700000000,
        tz: 'Asia/Tokyo;+09:00',
        lastAccess: 1700000100,
        lastBonusAt: 1700000050,
        mass: 15.5,
        supersededBy: '01OLD',
      );
      final json = original.toJson();
      final restored = MemoryRecord.fromJson(json);
      expect(restored.id, original.id);
      expect(restored.text, original.text);
      expect(restored.tier, original.tier);
      expect(restored.gen, original.gen);
      expect(restored.createdAt, original.createdAt);
      expect(restored.tz, original.tz);
      expect(restored.lastAccess, original.lastAccess);
      expect(restored.lastBonusAt, original.lastBonusAt);
      expect(restored.mass, original.mass);
      expect(restored.supersededBy, original.supersededBy);
    });

    test('toString includes tombstone marker and formats mass', () {
      const live = MemoryRecord(
        id: '01Z',
        text: 'hello world',
        tier: 1,
        gen: 0,
        createdAt: 100,
        tz: 'UTC;+00:00',
        lastAccess: 100,
        lastBonusAt: 100,
        mass: 3.5,
      );
      expect(live.toString(), contains('L1'));
      expect(live.toString(), contains('3.50'));
      expect(live.toString(), contains('hello world'));
      expect(live.toString(), isNot(contains('tombstone')));

      const dead = MemoryRecord(
        id: '01D',
        text: 'bye',
        tier: 2,
        gen: 1,
        createdAt: 100,
        tz: 'UTC;+00:00',
        lastAccess: 100,
        lastBonusAt: 100,
        mass: 0.5,
        supersededBy: '01D',
      );
      expect(dead.toString(), contains('tombstone'));
    });
  });

  group('VectorRecord', () {
    test('f32 toJson/fromJson round-trip', () {
      final original = VectorRecord(
        memoryId: '01M',
        modelId: 'test-model',
        dim: 768,
        dtype: VecDtype.f32,
        bytes: Uint8List.fromList([0, 0, 128, 63]), // 1.0f in LE
      );
      final json = original.toJson();
      expect(json['dtype'], 'f32');
      expect(json['scale'], isNull);
      expect(json['bytes'], isA<String>());
      final restored = VectorRecord.fromJson(json);
      expect(restored.memoryId, original.memoryId);
      expect(restored.modelId, original.modelId);
      expect(restored.dim, original.dim);
      expect(restored.dtype, original.dtype);
      expect(restored.scale, isNull);
      expect(restored.bytes, original.bytes);
    });

    test('int8 toJson/fromJson round-trip with scale', () {
      final original = VectorRecord(
        memoryId: '01N',
        modelId: 'test-model',
        dim: 128,
        dtype: VecDtype.int8,
        scale: 0.0078,
        bytes: Uint8List.fromList([10, 20, 30, 40]),
      );
      final json = original.toJson();
      expect(json['dtype'], 'int8');
      expect(json['scale'], 0.0078);
      final restored = VectorRecord.fromJson(json);
      expect(restored.dtype, VecDtype.int8);
      expect(restored.scale, 0.0078);
      expect(restored.bytes, original.bytes);
    });
  });

  group('ConflictRecord', () {
    test('toJson/fromJson round-trip', () {
      const original = ConflictRecord(
        a: '01FIRST',
        b: '01SECOND',
        at: 1700000000,
      );
      final json = original.toJson();
      final restored = ConflictRecord.fromJson(json);
      expect(restored.a, original.a);
      expect(restored.b, original.b);
      expect(restored.at, original.at);
    });
  });

  group('DreamLogRecord', () {
    test('toJson/fromJson round-trip', () {
      const original = DreamLogRecord(
        fingerprint: 'abc123def456',
        verdict: 'merge',
        at: 1700000000,
      );
      final json = original.toJson();
      final restored = DreamLogRecord.fromJson(json);
      expect(restored.fingerprint, original.fingerprint);
      expect(restored.verdict, original.verdict);
      expect(restored.at, original.at);
    });
  });

  group('RecalledMemory', () {
    test('all fields are accessible', () {
      const rm = RecalledMemory(
        id: '01R',
        text: 'test memory',
        score: 0.85,
        cosine: 0.9,
        activation: 12.0,
        mass: 15.0,
        tier: 1,
        gen: 0,
        createdAt: 1700000000,
        tz: 'Asia/Tokyo;+09:00',
      );
      expect(rm.id, '01R');
      expect(rm.score, 0.85);
      expect(rm.cosine, 0.9);
      expect(rm.activation, 12.0);
      expect(rm.mass, 15.0);
      expect(rm.tier, 1);
      expect(rm.gen, 0);
      expect(rm.createdAt, 1700000000);
      expect(rm.tz, 'Asia/Tokyo;+09:00');
    });
  });

  group('RetrieveResult', () {
    test('empty constant has empty pack and list', () {
      expect(RetrieveResult.empty.packText, '');
      expect(RetrieveResult.empty.recalled, isEmpty);
    });
  });

  group('SaveResult', () {
    test('all fields are populated for inserted action', () {
      const sr = SaveResult(
        action: SaveAction.inserted,
        id: '01INS',
        text: 'new fact',
        tier: 1,
        gen: 0,
        mass: 1.0,
        activation: 1.0,
        cosine: 0.05,
      );
      expect(sr.action, SaveAction.inserted);
      expect(sr.id, '01INS');
      expect(sr.error, isNull);
      expect(sr.supersededId, isNull);
      expect(sr.conflictWithId, isNull);
    });

    test('updated action carries supersededId', () {
      const sr = SaveResult(
        action: SaveAction.updated,
        id: '01UPD',
        text: 'updated fact',
        tier: 1,
        gen: 0,
        mass: 2.0,
        activation: 2.0,
        cosine: 0.98,
        supersededId: '01OLD',
      );
      expect(sr.supersededId, '01OLD');
      expect(sr.conflictWithId, isNull);
    });

    test('conflict action carries conflictWithId', () {
      const sr = SaveResult(
        action: SaveAction.conflict,
        id: '01CFL',
        text: 'conflicting fact',
        tier: 1,
        gen: 0,
        mass: 1.0,
        activation: 1.0,
        cosine: 0.88,
        conflictWithId: '01EXISTING',
      );
      expect(sr.conflictWithId, '01EXISTING');
      expect(sr.supersededId, isNull);
    });

    test('rejected action carries error', () {
      const sr = SaveResult(
        action: SaveAction.rejected,
        error: 'text too long',
        text: 'abc',
      );
      expect(sr.id, isNull);
      expect(sr.error, 'text too long');
    });

    test('rateLimited action carries error', () {
      const sr = SaveResult(
        action: SaveAction.rateLimited,
        error: 'daily limit reached',
        text: 'fact',
      );
      expect(sr.action, SaveAction.rateLimited);
      expect(sr.id, isNull);
    });

    test('toString includes action name and error when present', () {
      expect(
        const SaveResult(
          action: SaveAction.rejected,
          error: 'bad',
          text: 'x',
        ).toString(),
        contains('rejected'),
      );
      expect(
        const SaveResult(
          action: SaveAction.rejected,
          error: 'bad',
          text: 'x',
        ).toString(),
        contains('bad'),
      );
      expect(
        const SaveResult(
          action: SaveAction.inserted,
          id: '01X',
          text: 'hi',
        ).toString(),
        contains('01X'),
      );
    });

    test('reinforced action has no supersededId', () {
      const sr = SaveResult(
        action: SaveAction.reinforced,
        id: '01REI',
        text: 'reinforced fact',
        tier: 1,
        gen: 0,
        mass: 3.0,
        activation: 3.0,
        cosine: 1.0,
      );
      expect(sr.action, SaveAction.reinforced);
      expect(sr.supersededId, isNull);
      expect(sr.conflictWithId, isNull);
    });
  });

  group('DeleteResult', () {
    test('tombstoned action', () {
      const dr = DeleteResult(
        action: DeleteAction.tombstoned,
        id: '01T',
        text: 'deleted text',
      );
      expect(dr.action, DeleteAction.tombstoned);
      expect(dr.id, '01T');
      expect(dr.text, 'deleted text');
    });

    test('deleted action', () {
      const dr = DeleteResult(
        action: DeleteAction.deleted,
        id: '01T',
        text: 'physically deleted',
      );
      expect(dr.action, DeleteAction.deleted);
    });

    test('notFound action has no text', () {
      const dr = DeleteResult(action: DeleteAction.notFound, id: 'nonexist');
      expect(dr.action, DeleteAction.notFound);
      expect(dr.text, isNull);
    });
  });

  group('DreamReport', () {
    test('success report has all fields', () {
      const report = DreamReport(
        clusterFingerprint: 'sha256...',
        action: DreamAction.merge,
        priority: 12.5,
        before: [
          MemorySnapshot(id: '01A', text: 'a', tier: 1, gen: 0),
          MemorySnapshot(id: '01B', text: 'b', tier: 1, gen: 0),
        ],
        after: [MemorySnapshot(id: '01C', text: 'merged', tier: 1, gen: 1)],
        deletedIds: ['01A', '01B'],
      );
      expect(report.action, DreamAction.merge);
      expect(report.before.length, 2);
      expect(report.after.length, 1);
      expect(report.deletedIds.length, 2);
      expect(report.error, isNull);
    });

    test('error report carries error string', () {
      const report = DreamReport(
        clusterFingerprint: 'fp',
        action: DreamAction.none,
        priority: 0,
        before: [],
        after: [],
        deletedIds: [],
        error: 'LLM timeout',
      );
      expect(report.error, 'LLM timeout');
      expect(report.action, DreamAction.none);
    });

    test('split action report', () {
      const report = DreamReport(
        clusterFingerprint: 'fp2',
        action: DreamAction.split,
        priority: 8.0,
        before: [MemorySnapshot(id: '01X', text: 'packed', tier: 1, gen: 0)],
        after: [
          MemorySnapshot(id: '01Y', text: 'unpacked 1', tier: 1, gen: 0),
          MemorySnapshot(id: '01Z', text: 'unpacked 2', tier: 1, gen: 0),
        ],
        deletedIds: ['01X'],
      );
      expect(report.action, DreamAction.split);
      expect(report.after.length, 2);
      expect(report.after.every((s) => s.gen == 0), isTrue);
    });
  });

  group('MemorySnapshot', () {
    test('holds id, text, tier, gen', () {
      const snap = MemorySnapshot(
        id: '01S',
        text: 'snapshot text',
        tier: 2,
        gen: 3,
      );
      expect(snap.id, '01S');
      expect(snap.text, 'snapshot text');
      expect(snap.tier, 2);
      expect(snap.gen, 3);
    });
  });

  group('DreamProposal', () {
    test('holds text and optional timezone', () {
      const withTz = DreamProposal(text: 'fact', timezone: 'Asia/Tokyo');
      expect(withTz.text, 'fact');
      expect(withTz.timezone, 'Asia/Tokyo');

      const withoutTz = DreamProposal(text: 'fact only');
      expect(withoutTz.text, 'fact only');
      expect(withoutTz.timezone, isNull);
    });
  });

  group('DreamDecision', () {
    test('none constant', () {
      const d = DreamDecision.none();
      expect(d.action, DreamAction.none);
      expect(d.memories, isEmpty);
    });

    test('merge action with memories', () {
      const d = DreamDecision(
        action: DreamAction.merge,
        memories: [DreamProposal(text: 'merged fact')],
      );
      expect(d.action, DreamAction.merge);
      expect(d.memories.length, 1);
    });

    test('split action with multiple memories', () {
      const d = DreamDecision(
        action: DreamAction.split,
        memories: [
          DreamProposal(text: 'fact a'),
          DreamProposal(text: 'fact b'),
        ],
      );
      expect(d.action, DreamAction.split);
      expect(d.memories.length, 2);
    });
  });

  group('EngramStats', () {
    test('total sums all tiers', () {
      const stats = EngramStats(
        l1: 10,
        l2: 20,
        l3: 5,
        tombstones: 3,
        conflicts: 4,
        dreamLog: 7,
      );
      expect(stats.total, 35);
    });

    test('toString format', () {
      const stats = EngramStats(
        l1: 1,
        l2: 2,
        l3: 3,
        tombstones: 1,
        conflicts: 1,
        dreamLog: 1,
      );
      final s = stats.toString();
      expect(s, contains('L1=1'));
      expect(s, contains('L2=2'));
      expect(s, contains('L3=3'));
      expect(s, contains('tombstones=1'));
      expect(s, contains('conflicts=1'));
      expect(s, contains('dreamLog=1'));
    });
  });

  group('VecDtype', () {
    test('parse returns correct enum', () {
      expect(VecDtype.parse('f32'), VecDtype.f32);
      expect(VecDtype.parse('int8'), VecDtype.int8);
      expect(VecDtype.parse('unknown'), VecDtype.int8); // fallback
      expect(VecDtype.parse(''), VecDtype.int8);
    });

    test('label returns correct string', () {
      expect(VecDtype.f32.label, 'f32');
      expect(VecDtype.int8.label, 'int8');
    });
  });
}
