/// Example [MemoryStore] adapter over `package:sqlite3`: one table, WAL
/// mode, real transactions (one per engine operation) and a rotating
/// snapshot ring for [backup]. On Flutter, add `sqlite3_flutter_libs` and this file
/// works unchanged.
library;

import 'dart:io';
import 'dart:typed_data';

import 'package:long_term_memory/long_term_memory.dart';
import 'package:sqlite3/sqlite3.dart';

const _schema = '''
CREATE TABLE IF NOT EXISTS memory (
  id TEXT PRIMARY KEY,
  text TEXT NOT NULL,
  created_at INTEGER NOT NULL,
  tz TEXT NOT NULL,
  last_recall INTEGER NOT NULL,
  stability REAL NOT NULL,
  consolidated INTEGER NOT NULL,
  model_id TEXT NOT NULL,
  vector BLOB NOT NULL
);
''';

/// [MemoryStore] backed by a single SQLite file (or `:memory:`).
class SqliteMemoryStore extends MemoryStore {
  /// Creates a store at [path] (`':memory:'` for an in-memory database).
  /// [snapshotGenerations] is the size of the rotating [backup] ring.
  SqliteMemoryStore(this.path, {this.snapshotGenerations = 8});

  /// Database file path.
  final String path;

  /// Number of rotating snapshot files kept by [backup].
  final int snapshotGenerations;

  late Database _db;
  var _open = false;

  @override
  Future<void> open() async {
    if (_open) return;
    _db = sqlite3.open(path);
    _db.execute('PRAGMA journal_mode=WAL;');
    _db.execute(_schema);
    _open = true;
  }

  @override
  Future<void> close() async {
    if (!_open) return;
    _db.dispose();
    _open = false;
  }

  @override
  Future<List<Memory>> loadAll() async => [
        for (final r in _db.select('SELECT * FROM memory ORDER BY rowid'))
          Memory(
            id: r['id'] as String,
            text: r['text'] as String,
            createdAt: r['created_at'] as int,
            tz: r['tz'] as String,
            lastRecall: r['last_recall'] as int,
            stability: (r['stability'] as num).toDouble(),
            consolidated: (r['consolidated'] as int) != 0,
            modelId: r['model_id'] as String,
            vector: unpackF32(r['vector'] as Uint8List),
          ),
      ];

  @override
  Future<void> put(Memory m) async => _db.execute(
        'INSERT OR REPLACE INTO memory(id,text,created_at,tz,last_recall,'
        'stability,consolidated,model_id,vector) VALUES(?,?,?,?,?,?,?,?,?)',
        [
          m.id, m.text, m.createdAt, m.tz, m.lastRecall, m.stability, //
          m.consolidated ? 1 : 0, m.modelId, packF32(m.vector),
        ],
      );

  @override
  Future<void> remove(String id) async =>
      _db.execute('DELETE FROM memory WHERE id=?', [id]);

  @override
  Future<void> clear() async => _db.execute('DELETE FROM memory');

  @override
  Future<T> transaction<T>(Future<T> Function() action) async {
    _db.execute('BEGIN IMMEDIATE');
    try {
      final result = await action();
      _db.execute('COMMIT');
      return result;
    } catch (_) {
      _db.execute('ROLLBACK');
      rethrow;
    }
  }

  @override
  Future<void> backup() async {
    if (path == ':memory:') return;
    final dir = Directory('$path.snapshots')..createSync(recursive: true);
    final last = File('${dir.path}/snap-${snapshotGenerations - 1}.db');
    if (last.existsSync()) last.deleteSync();
    for (var i = snapshotGenerations - 2; i >= 0; i--) {
      final f = File('${dir.path}/snap-$i.db');
      if (f.existsSync()) f.renameSync('${dir.path}/snap-${i + 1}.db');
    }
    _db.execute('VACUUM INTO ?', ['${dir.path}/snap-0.db']);
  }
}
