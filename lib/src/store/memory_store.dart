import '../models.dart';

/// The storage interface — implemented by the application over any database.
///
/// The engine keeps every trace in RAM and treats the store purely as the
/// durable substrate, so an adapter is six operations: load everything once,
/// put / remove single rows, clear, and two optional hooks. No SQL or query
/// semantics leak through; all memory logic lives in the engine.
///
/// **Transactionality**: the engine wraps every operation's writes in
/// [transaction] — a remember (insert + eviction), a recall's strength
/// updates, and a dream replacement (delete a cluster + insert its gists). On
/// a non-transactional store a crash inside that window can lose members
/// without their replacement; implement [transaction] and [backup] if your
/// data matters.
abstract class MemoryStore {
  /// Opens / migrates the backing storage. Called once by `initialize()`.
  Future<void> open() async {}

  /// Releases resources.
  Future<void> close() async {}

  /// Returns every stored trace (called once at start-up), in insertion
  /// order — the engine tie-breaks its dream seeds and recall candidates on
  /// it, so a store that reorders on rewrite changes behaviour after a
  /// restart. Ordering by id works: ULIDs are lexicographically time-ordered.
  Future<List<Memory>> loadAll();

  /// Inserts or replaces the row with `memory.id`.
  Future<void> put(Memory memory);

  /// Physically deletes the row with [id] (no-op if absent).
  Future<void> remove(String id);

  /// Deletes every row.
  Future<void> clear();

  /// Runs [action] atomically if the backend supports transactions.
  Future<T> transaction<T>(Future<T> Function() action) => action();

  /// Takes a backup before destructive dream work (e.g. a rotating file
  /// ring for SQLite). Defaults to a no-op.
  Future<void> backup() async {}
}
