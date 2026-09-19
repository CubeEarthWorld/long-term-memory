import '../models.dart';
import 'memory_store.dart';

/// Reference [MemoryStore] backed by a map — zero dependencies.
///
/// Data lives in RAM only; use [toJson] / [InMemoryStore.fromJson] for crude
/// persistence, or implement a real adapter (see `example/sqlite_adapter`).
class InMemoryStore extends MemoryStore {
  /// Creates an empty store.
  InMemoryStore();

  /// Restores a store serialised with [toJson].
  factory InMemoryStore.fromJson(Map<String, Object?> json) {
    final store = InMemoryStore();
    for (final m in (json['memories'] as List<Object?>? ?? const [])) {
      final rec = Memory.fromJson((m as Map).cast<String, Object?>());
      store._rows[rec.id] = rec;
    }
    return store;
  }

  final Map<String, Memory> _rows = {};

  @override
  Future<List<Memory>> loadAll() async => _rows.values.toList();

  @override
  Future<void> put(Memory memory) async => _rows[memory.id] = memory;

  @override
  Future<void> remove(String id) async => _rows.remove(id);

  @override
  Future<void> clear() async => _rows.clear();

  /// Serialises the whole store to JSON (vectors base64-encoded).
  Map<String, Object?> toJson() =>
      {'memories': _rows.values.map((m) => m.toJson()).toList()};
}
