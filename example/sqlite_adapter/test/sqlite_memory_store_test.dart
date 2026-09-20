import 'dart:io';
import 'dart:typed_data';

import 'package:long_term_memory/long_term_memory.dart';
import 'package:sqlite_memory_store/sqlite_memory_store.dart';
import 'package:test/test.dart';

Memory mem(String id, {bool consolidated = false}) => Memory(
      id: id,
      text: 'text of $id',
      createdAt: 1700000000,
      tz: 'Asia/Tokyo;+09:00',
      lastRecall: 1700000000,
      stability: 86400,
      consolidated: consolidated,
      modelId: 'test',
      vector: Float32List.fromList([1, 0, 0]),
      cue: 'cue of $id',
    );

void main() {
  late Directory tmp;
  late SqliteMemoryStore store;

  setUp(() async {
    tmp = Directory.systemTemp.createTempSync('engram_sqlite_test');
    store = SqliteMemoryStore('${tmp.path}/memory.db');
    await store.open();
  });

  tearDown(() async {
    await store.close();
    tmp.deleteSync(recursive: true);
  });

  test('put / loadAll / remove round-trip', () async {
    await store.put(mem('a'));
    await store.put(mem('b', consolidated: true));
    final rows = await store.loadAll();
    expect(rows.map((m) => m.id), ['a', 'b']);
    expect(rows[1].consolidated, isTrue);
    expect(rows[0].vector, [1, 0, 0]);
    expect(rows[0].cue, 'cue of a');   // a dropped cue column went unnoticed
    await store.put(mem('a').copyWith(stability: 99));
    expect(
        (await store.loadAll()).firstWhere((m) => m.id == 'a').stability, 99);
    await store.remove('a');
    expect((await store.loadAll()).map((m) => m.id), ['b']);
  });

  test('transaction rolls back on error', () async {
    await store.put(mem('a'));
    await expectLater(
      store.transaction(() async {
        await store.remove('a');
        throw StateError('boom');
      }),
      throwsStateError,
    );
    expect((await store.loadAll()).length, 1);
  });

  test('backup keeps a rotating ring and survives reopen', () async {
    await store.put(mem('a'));
    for (var i = 0; i < 3; i++) {
      await store.backup();
    }
    final snaps = Directory('${tmp.path}/memory.db.snapshots').listSync();
    expect(snaps.length, 3);
    await store.close();
    store = SqliteMemoryStore('${tmp.path}/memory.db');
    await store.open();
    expect((await store.loadAll()).single.id, 'a');
  });
}
