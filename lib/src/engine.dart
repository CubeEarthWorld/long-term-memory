/// The ENGRAM v2.1 engine: traces with a forgetting curve, wake-phase verbs
/// (remember / recall / forget) and a sleep-phase dream.
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'config.dart';
import 'dream/adjudicator.dart';
import 'embedder.dart';
import 'models.dart';
import 'store/memory_store.dart';
import 'text.dart';
import 'timezone.dart';
import 'ulid.dart';
import 'vector_index.dart';
import 'vector_math.dart';

/// Seeds examined per dream, as a multiple of the LLM budget (bounds the
/// cost of a dream to O(budget · capacity · dim)).
const int _seedsPerBudget = 8;

/// A long-term memory for LLM applications (ENGRAM v2.1, see SPEC.md).
///
/// Every trace is held in RAM; the injected [MemoryStore] is the durable
/// substrate, the [Embedder] turns text into vectors, the [VectorIndex] scores
/// them (pure Dart by default; an application may plug in a native one) and —
/// for [dream] only — a [DreamAdjudicator] backed by any LLM. Every entry
/// point accepts an explicit Unix time (`nowUnix`, seconds) and a
/// [MemoryTimezone]; omitted values fall back to the injected [clock] /
/// `defaultTimezone`, then to the system clock / UTC. Public methods are serialized, so an instance is
/// safe to call from interleaving async code. One process owns a store.
class EngramMemory {
  /// Creates an engine over [store] and [embedder]. [index] holds the vectors
  /// of the indexed traces (default [DartVectorIndex]); the engine owns it and
  /// keeps it in step with its traces, so it must not be shared with another
  /// engine ([initialize] clears it).
  EngramMemory({
    required this.store,
    required this.embedder,
    this.config = const EngramConfig(),
    int Function()? clock,
    MemoryTimezone? defaultTimezone,
    VectorIndex? index,
  })  : _clock = clock,
        _defaultTz = defaultTimezone ?? MemoryTimezone.utc,
        _index = index ?? DartVectorIndex();

  /// The injected storage backend.
  final MemoryStore store;

  /// The injected embedding model.
  final Embedder embedder;

  /// Active parameters.
  final EngramConfig config;

  final int Function()? _clock;
  final MemoryTimezone _defaultTz;

  /// Every trace, in insertion order (which breaks every tie). Mutated only by
  /// [_setTrace], [_dropTrace] and [_clearTraces], which keep [_index] holding
  /// exactly the traces that are [_isIndexed].
  final Map<String, Memory> _traces = {};
  final VectorIndex _index;
  final List<int> _writeTimes = [];
  final Map<String, Recalled> _lastRecall = {};
  Future<void> _chain = Future<void>.value();

  /// An L2-normalised freshly embedded vector, or `null` when the embedder
  /// contradicted its own [Embedder.dimension]. A wrong-length vector is an
  /// embedder fault, not data: returning null routes it through the same path
  /// as an offline embedder, so the trace is indexed on a later attempt
  /// instead of reaching [dot], which throws.
  Float32List? _vector(List<double> v) {
    final out = l2Normalized(v);
    return out.length == embedder.dimension ? out : null;
  }

  Future<T> _serialize<T>(Future<T> Function() action) {
    final run = _chain.then((_) => action());
    _chain = run.then<void>((_) {}, onError: (Object _) {});
    return run;
  }

  // ================================================================== //
  // clock
  // ================================================================== //

  /// Current Unix time in seconds (injected clock → wall clock).
  int nowUnix() =>
      _clock?.call() ?? DateTime.now().millisecondsSinceEpoch ~/ 1000;

  /// Current local datetime string (e.g. `'2026-06-12 09:30 +09:00'`) — the
  /// format expected by the bundled prompt templates.
  String nowLocal() => formatLocal(nowUnix(), _defaultTz.storageField);

  // ================================================================== //
  // the forgetting curve (SPEC §3)
  // ================================================================== //

  /// Retrievability `R = 2^(−max(0, now − lastRecall) / stability)`.
  double retrievability(Memory m, int now) =>
      math.pow(2.0, -_elapsed(m, now) / m.stability).toDouble();

  /// Strength `S·R` — proportional to the trace's total remaining
  /// retrievability; eviction takes the lowest.
  double strength(Memory m, int now) => m.stability * retrievability(m, now);

  /// `log2(strength)`: the same ordering as [strength] but free of the
  /// floating-point underflow that would tie every dormant trace at 0.
  double _rank(Memory m, int now) =>
      math.log(m.stability) / math.ln2 - _elapsed(m, now) / m.stability;

  /// Cue activation: the cosine rescaled above the model's baseline.
  double _activation(double cos) =>
      math.max(0.0, (cos - config.cosineFloor) / (1 - config.cosineFloor));

  double _elapsed(Memory m, int now) =>
      math.max(0, now - m.lastRecall).toDouble();

  // NaN would survive min/max and then never compare below weakestRank,
  // i.e. an immortal trace (SPEC §7 forbids one).
  double _clampStability(double s) =>
      math.min(math.max(s.isNaN ? 1.0 : s, 1.0), config.maxStability);

  /// The retrieval update: stability grows with the spacing achieved
  /// (`1 − R`) and the cue activation [a]; `lastRecall` moves to [now].
  Memory _retrieved(Memory m, int now, double a) => m.copyWith(
        stability: _clampStability(m.stability *
            (1 + config.spacingGain * a * (1 - retrievability(m, now)))),
        lastRecall: now,
      );

  // ================================================================== //
  // lifecycle
  // ================================================================== //

  /// Opens the store, loads every trace, clamps out-of-range values and
  /// re-embeds traces indexed under another model. Call once first.
  Future<void> initialize() => _serialize(() async {
        await store.open();
        final now = nowUnix();
        _clearTraces();
        for (final raw in await store.loadAll()) {
          // Every row is validated (SPEC §2). A store can hand back a row an
          // older version or a hand-edited database wrote; an unusable one is
          // skipped rather than allowed to break search for the whole store.
          if (raw.id.isEmpty || raw.text.isEmpty) continue;
          final m = raw.copyWith(
            stability: _clampStability(raw.stability),
            lastRecall: math.min(raw.lastRecall, now),
            createdAt: math.min(raw.createdAt, now),
          );
          _setTrace(m);
          if (m.stability != raw.stability ||
              m.lastRecall != raw.lastRecall ||
              m.createdAt != raw.createdAt) {
            await store.put(m);
          }
        }
        await _reindex();
      });

  /// Re-embeds traces whose vector belongs to another model, in batches. A
  /// failing embedder leaves the remaining traces stale (invisible to
  /// search) until the next attempt.
  Future<void> _reindex() async {
    final stale = _stale;
    for (var i = 0; i < stale.length; i += 64) {
      final batch = stale.sublist(i, math.min(i + 64, stale.length));
      List<Float32List> vectors;
      try {
        vectors =
            await embedder.embedDocuments([for (final m in batch) m.text]);
      } catch (_) {
        return;
      }
      await store.transaction(() async {
        for (var j = 0; j < batch.length; j++) {
          final v = _vector(vectors[j]);
          if (v == null) continue; // stays stale, retried later
          await _put(batch[j].copyWith(modelId: embedder.modelId, vector: v));
        }
      });
    }
  }

  /// Whether the current embedder can score this trace. One `modelId` means one
  /// dimension (SPEC §7), so a vector of any other length — a provider glitch,
  /// or a model file swapped behind an unchanged `modelId` — counts as stale and
  /// is re-embedded by the next [_reindex] instead of reaching [dot].
  bool _isIndexed(Memory m) =>
      m.modelId == embedder.modelId && m.vector.length == embedder.dimension;

  Iterable<Memory> get _indexed => _traces.values.where(_isIndexed);

  List<Memory> get _stale =>
      _traces.values.where((m) => !_isIndexed(m)).toList(growable: false);

  bool get _anyStale => _traces.values.any((m) => !_isIndexed(m));

  /// Inserts or replaces [m] in [_traces] and keeps [_index] in step: an
  /// indexed trace is (re)put unless the index already holds this very
  /// vector — the common case, a strength update — and a trace that stopped
  /// being indexed is removed.
  void _setTrace(Memory m) {
    final old = _traces[m.id];
    _traces[m.id] = m;
    final wasIndexed = old != null && _isIndexed(old);
    if (_isIndexed(m)) {
      if (!wasIndexed || !identical(old.vector, m.vector)) {
        _index.put(m.id, m.vector);
      }
    } else if (wasIndexed) {
      _index.remove(m.id);
    }
  }

  /// Removes [id] from [_traces] and, if it was indexed, from [_index].
  void _dropTrace(String id) {
    final old = _traces.remove(id);
    if (old != null && _isIndexed(old)) _index.remove(id);
  }

  void _clearTraces() {
    _traces.clear();
    _index.clear();
  }

  /// Scores every indexed trace against [queries] in one pass over the index.
  /// Returns the traces in insertion order and the row-major scores
  /// (`sims[i * queries.length + j]` = cos(trace i, query j)).
  (List<Memory>, Float64List) _scan(List<Float32List> queries) {
    final rows = _indexed.toList(growable: false);
    if (rows.isEmpty || queries.isEmpty) return (rows, Float64List(0));
    return (rows, _scores(rows, queries));
  }

  /// [VectorIndex.scores] for [rows], with its shape checked: a native index
  /// that returned the wrong shape would otherwise corrupt every ranking.
  Float64List _scores(List<Memory> rows, List<Float32List> queries) {
    final out = _index.scores([for (final m in rows) m.id], queries);
    if (out.length != rows.length * queries.length) {
      throw StateError('VectorIndex.scores returned ${out.length} scores for '
          '${rows.length} ids × ${queries.length} queries');
    }
    return out;
  }

  /// Erases everything.
  Future<void> reset() => _serialize(() async {
        await store.clear();
        _clearTraces();
        _writeTimes.clear();
        _lastRecall.clear();
      });

  /// All traces (a snapshot, in insertion order).
  Future<List<Memory>> memories() =>
      _serialize(() async => _traces.values.toList(growable: false));

  /// The trace with [id], or `null`.
  Future<Memory?> memory(String id) => _serialize(() async => _traces[id]);

  // ================================================================== //
  // wake phase (SPEC §4)
  // ================================================================== //

  /// Stores one self-contained proposition. Exact duplicates are rehearsed
  /// instead of stored; everything else is inserted (never overwritten —
  /// paraphrases and contradictions are resolved by the dream). Beyond
  /// [EngramConfig.capacity] the weakest trace past the grace period is
  /// forgotten. [salience] (0–10) scales the initial stability.
  Future<RememberResult> remember(
    String text, {
    double salience = 1.0,
    String cue = '',
    int? nowUnix,
    MemoryTimezone? timezone,
  }) =>
      _serialize(() async {
        final now = nowUnix ?? this.nowUnix();
        text = cleanText(text, config.textMax);
        cue = cleanText(cue, config.textMax);
        if (text.isEmpty) return const RememberResult(RememberAction.rejected);
        for (final m in _traces.values) {
          if (m.text == text) {
            final r = _retrieved(m, now, 1.0);
            await _put(r);
            return RememberResult(RememberAction.reinforced, memory: r);
          }
        }
        final dayAgo = now - 86400;
        _writeTimes.removeWhere((t) => t < dayAgo);
        if (_writeTimes.length >= config.writesPerDay) {
          return const RememberResult(RememberAction.rateLimited);
        }
        _writeTimes.add(now);
        Float32List? v;
        try {
          v = _vector((await embedder.embedDocuments([text]))[0]);
        } catch (_) {
          v = null; // embedder offline: keep the text, index it later
        }
        final q = v == null ? null : (await _cueVectors([(cue, v)])).single;
        final candidates =
            q == null ? const <(double, Memory)>[] : _candidates(q, null);
        final best = candidates.isEmpty ? 0.0 : candidates.first.$1;
        final m = Memory(
          id: ulid(now * 1000),
          text: text,
          createdAt: now,
          tz: (timezone ?? _defaultTz).storageField,
          lastRecall: now,
          // clamp() orders NaN above every double, so NaN would read as 10.
          stability: _clampStability(config.initialStability *
              (salience.isNaN ? 1.0 : salience.clamp(0.0, 10.0))),
          consolidated: v != null && candidates.isEmpty && !_anyStale,
          modelId: v == null ? '' : embedder.modelId,
          vector: v ?? Float32List(0),
          cue: cue,
        );
        // One operation = one commit.
        final evicted = await store.transaction(() async {
          await _put(m);
          return _enforceCapacity(now);
        });
        return RememberResult(RememberAction.inserted,
            memory: m, cosine: best, evicted: evicted);
      });

  /// Retrieves up to [EngramConfig.injectN] traces relevant to [query] and
  /// packs them for prompt injection. Injection is exposure, not use: each
  /// injected trace receives the retrieval update at half its cue
  /// activation; [cite] completes it for the traces the reply actually used.
  Future<RecallResult> recall(String query, {int? nowUnix}) =>
      _serialize(() async {
        final now = nowUnix ?? this.nowUnix();
        _lastRecall
            .clear(); // a recall always ends the previous citation window
        final parts = cues(query, config.maxCues);
        // Nothing indexed = nothing to score, so do not pay for an embedding.
        if (parts.isEmpty || _indexed.isEmpty) return RecallResult.empty;
        List<Float32List> qs;
        try {
          qs = [
            for (final q in await embedder.embedQueries(parts))
              _vector(q) ?? (throw StateError('embedder dimension mismatch')),
          ];
        } catch (_) {
          return RecallResult.empty; // embedder offline: nothing to cue with
        }

        // One pass over the index for every cue, into flat arrays; a Recalled
        // is built only for the pool.
        final (rows, sims) = _scan(qs);
        final k = qs.length;
        final cosines = Float64List(rows.length);
        final rs = Float64List(rows.length);
        final scores = Float64List(rows.length);
        var top = 0.0;
        for (var i = 0; i < rows.length; i++) {
          var cos = double.negativeInfinity;
          for (var j = 0; j < k; j++) {
            cos = math.max(cos, sims[i * k + j]);
          }
          final r = retrievability(rows[i], now);
          final score =
              _activation(cos) * (config.alpha + (1 - config.alpha) * r);
          top = math.max(top, score);
          cosines[i] = cos;
          rs[i] = r;
          scores[i] = score;
        }
        final floor = math.max(config.minScore, config.relativeScore * top);
        // Insertion order; MMR ties keep it.
        final pool = [
          for (var i = 0; i < rows.length; i++)
            if (scores[i] >= floor)
              Recalled(rows[i],
                  score: scores[i], cosine: cosines[i], retrievability: rs[i]),
        ];
        final lines = <String>[];
        final packed = <Recalled>[];
        var used = 0;
        // One operation = one commit.
        await store.transaction(() async {
          for (final c in _mmr(pool)) {
            final m = c.memory;
            final line = '[${m.createdAt} ${m.tz}] ${m.text}　《id:${m.id}》\n';
            final cost = line.runes.length; // code points, like Python's len()
            if (lines.isNotEmpty && used + cost > config.budgetChars) {
              continue;
            }
            lines.add(line);
            used += cost;
            packed.add(c);
            _lastRecall[m.id] = c;
            await _put(_retrieved(m, now, 0.5 * _activation(c.cosine)));
          }
        });
        return RecallResult(packText: lines.join(), recalled: packed);
      });

  /// Marks the traces cited in [replyText] (as `《id:…》`) from the last
  /// [recall] as actually used: their stability is raised to what a full
  /// activation (`a = 1`) would have given. Returns the ids affected.
  Future<List<String>> cite(String replyText) => _serialize(() async {
        final ids = <String>[];
        for (final match in RegExp('《id:([^》]+)》').allMatches(replyText)) {
          final c = _lastRecall.remove(match.group(1));
          if (c == null) continue;
          final m = _traces[c.memory.id];
          if (m == null) continue;
          final partial = 1 +
              config.spacingGain *
                  0.5 *
                  _activation(c.cosine) *
                  (1 - c.retrievability);
          final full = 1 + config.spacingGain * (1 - c.retrievability);
          await _put(m.copyWith(
              stability: _clampStability(m.stability * full / partial)));
          ids.add(m.id);
        }
        return ids;
      });

  /// MMR: `next = argmax[score − λ·max cos(m, selected)]`.
  List<Recalled> _mmr(List<Recalled> pool) {
    final maxSim = List<double>.filled(pool.length, 0.0);
    final alive = List<bool>.filled(pool.length, true);
    final selected = <Recalled>[];
    while (selected.length < config.injectN) {
      var bestI = -1;
      var bestVal = double.negativeInfinity;
      for (var i = 0; i < pool.length; i++) {
        if (!alive[i]) continue;
        final val = pool[i].score - config.mmrLambda * maxSim[i];
        if (val > bestVal) {
          bestVal = val;
          bestI = i;
        }
      }
      if (bestI < 0) break;
      selected.add(pool[bestI]);
      alive[bestI] = false;
      if (selected.length == config.injectN) break;
      final rest = [
        for (var i = 0; i < pool.length; i++)
          if (alive[i]) i,
      ];
      if (rest.isEmpty) break;
      final sims = _scores(
          [for (final i in rest) pool[i].memory], [pool[bestI].memory.vector]);
      for (var r = 0; r < rest.length; r++) {
        maxSim[rest[r]] = math.max(maxSim[rest[r]], sims[r]);
      }
    }
    return selected;
  }

  /// Deletes the trace with [id]. Returns whether it existed.
  Future<bool> forget(String id) => _serialize(() async {
        if (!_traces.containsKey(id)) return false;
        await _remove(id);
        return true;
      });

  Future<void> _put(Memory m) async {
    _setTrace(m);
    await store.put(m);
  }

  Future<void> _remove(String id) async {
    _dropTrace(id);
    await store.remove(id);
  }

  /// Forgets the lowest-ranked trace while over capacity. Traces inside the
  /// grace period are protected unless they make up more than a tenth of
  /// the capacity (so a flood evicts its own members) or nothing older
  /// exists. Returns the last victim.
  Future<Memory?> _enforceCapacity(int now) async {
    final excess = _traces.length - config.capacity;
    if (excess <= 0) return null;
    final young = <(double, int, Memory)>[];
    final old = <(double, int, Memory)>[];
    var index = 0;
    for (final m in _traces.values) {
      final age = now - m.createdAt;
      (age >= 0 && age < config.gracePeriod ? young : old)
          .add((_rank(m, now), index++, m));
    }
    // Ranks cannot change while evicting: order each age group once (weakest
    // first, ties by insertion order) instead of rescanning per victim.
    int compare((double, int, Memory) a, (double, int, Memory) b) =>
        a.$1 == b.$1 ? a.$2.compareTo(b.$2) : a.$1.compareTo(b.$1);
    young.sort(compare);
    old.sort(compare);
    var yi = 0;
    var oi = 0;
    Memory? victim;
    for (var i = 0; i < excess; i++) {
      if (oi == old.length ||
          (young.length - yi > config.capacity ~/ 10 &&
              young[yi].$1 < old[oi].$1)) {
        victim = young[yi++].$3;
      } else {
        victim = old[oi++].$3;
      }
      await _remove(victim.id);
    }
    return victim;
  }

  // ================================================================== //
  // sleep phase (SPEC §5)
  // ================================================================== //

  /// The clusters the next dream would hand to the LLM (seed first),
  /// without calling it — for inspection UIs.
  /// Clusters may overlap: a settled trace is a candidate for every later
  /// piece of evidence. [budget] defaults to [EngramConfig.dreamBudget], as
  /// in [dream].
  ///
  /// Nothing changes while previewing, so every seed's cue is embedded in one
  /// [Embedder.embedQueries] call and all of them are scored in one pass over
  /// the index.
  Future<List<List<Memory>>> clusters({int? budget}) => _serialize(() async {
        final seeds = _seeds(budget ?? config.dreamBudget);
        if (seeds.isEmpty) return <List<Memory>>[];
        final qs =
            await _cueVectors([for (final s in seeds) (s.cue, s.vector)]);
        final (rows, sims) = _scan(qs);
        final out = <List<Memory>>[];
        for (var j = 0; j < seeds.length; j++) {
          final cluster = [
            seeds[j],
            for (final (_, i) in _near(rows, sims, qs.length, j, seeds[j]))
              rows[i],
          ];
          if (cluster.length >= 2) out.add(cluster);
        }
        return out;
      });

  /// The labile traces a dream of [budget] scans: first in, first out
  /// (insertion order), at most 8·budget. FIFO needs no ranking and never
  /// starves a seed. Materialised: the dream mutates [_traces] as it goes.
  List<Memory> _seeds(int budget) => _indexed
      .where((m) => !m.consolidated)
      .take(_seedsPerBudget * math.max(budget, 0))
      .toList(growable: false);

  /// Each (cue, own vector) pair's query vector: the cue's, or the trace's own
  /// vector when it has no cue. The distinct non-empty cues are embedded in a
  /// single [Embedder.embedQueries] call. When that call fails (embedder
  /// offline) every pair falls back to its own vector; a missing or
  /// wrong-dimension result falls back for its own pairs only.
  Future<List<Float32List>> _cueVectors(
      List<(String, Float32List)> pairs) async {
    final slot = <String, int>{};
    for (final (cue, _) in pairs) {
      if (cue.isNotEmpty) slot.putIfAbsent(cue, () => slot.length);
    }
    var embedded = const <Float32List?>[];
    if (slot.isNotEmpty) {
      try {
        final raw = await embedder.embedQueries(slot.keys.toList());
        embedded = [
          for (var i = 0; i < slot.length; i++)
            i < raw.length ? _vector(raw[i]) : null,
        ];
      } catch (_) {
        // embedder offline: every cue falls back to its trace's own vector
      }
    }
    return [
      for (final (cue, own) in pairs)
        switch (slot[cue]) {
          final int s when s < embedded.length => embedded[s] ?? own,
          _ => own,
        },
    ];
  }

  /// Traces the cue [q] reactivates (see [_near]), with their cosines.
  List<(double, Memory)> _candidates(Float32List q, Memory? seed) {
    final (rows, sims) = _scan([q]);
    return [
      for (final (cos, i) in _near(rows, sims, 1, 0, seed)) (cos, rows[i]),
    ];
  }

  /// The traces query [column] of a [_scan] reactivates, strongest first, as
  /// (cos, row): cos >= thetaRelated and not written after the seed, so
  /// evidence only ever rewrites its own past (SPEC §5). Ties break on
  /// insertion order — ULID tails are random, so ids would not agree across
  /// languages. At most dreamMaxMembers - 1.
  List<(double, int)> _near(List<Memory> rows, Float64List sims, int stride,
      int column, Memory? seed) {
    final near = <(double, int)>[
      for (var i = 0; i < rows.length; i++)
        if (seed == null ||
            (rows[i].id != seed.id && rows[i].createdAt <= seed.createdAt))
          for (final cos in [sims[i * stride + column]])
            if (cos >= config.thetaRelated) (cos, i),
    ]..sort(
        (a, b) => a.$1 == b.$1 ? a.$2.compareTo(b.$2) : b.$1.compareTo(a.$1));
    return near.take(config.dreamMaxMembers - 1).toList(growable: false);
  }

  /// Offline consolidation — the only place traces are rewritten. Runs:
  /// backup → re-index stale traces → up to [budget] LLM adjudications via
  /// [adjudicate] (keep / replace per cluster, each in one transaction).
  Future<List<DreamReport>> dream({
    required DreamAdjudicator adjudicate,
    int? budget,
    int? nowUnix,
    MemoryTimezone? timezone,
  }) =>
      _serialize(() async {
        final now = nowUnix ?? this.nowUnix();
        final tz = (timezone ?? _defaultTz).storageField;
        await store.backup();
        await _reindex();
        var left = budget ?? config.dreamBudget;
        final reports = <DreamReport>[];
        final seeds = _seeds(left);
        // A seed's cue and vector never change during a dream (a seed is only
        // settled or absorbed), so every cue is embedded up front in one call.
        // The candidates are still searched per seed: each verdict rewrites
        // the store the next seed searches.
        final qs =
            await _cueVectors([for (final s in seeds) (s.cue, s.vector)]);
        for (var j = 0; j < seeds.length; j++) {
          final seed = _traces[seeds[j].id];
          if (seed == null || seed.consolidated) {
            continue;
          }
          if (left <= 0) break;
          // The seed (newest evidence), then the older traces its cue reactivates.
          final cluster = [
            seed,
            for (final (_, m) in _candidates(qs[j], seed)) m,
          ];
          if (cluster.length < 2) {
            await _put(seed.copyWith(consolidated: true));
            continue;
          }
          left -= 1;
          // An error leaves the seed labile for the next dream.
          reports.add(await _adjudicate(adjudicate, cluster, now, tz));
        }
        return reports;
      });

  Future<DreamReport> _adjudicate(
    DreamAdjudicator adjudicate,
    List<Memory> cluster,
    int now,
    String tz,
  ) async {
    final request = DreamRequest(
      currentLocalTime: formatLocal(now, tz),
      members: [
        for (final m in cluster)
          DreamMember(
            id: m.id,
            text: m.text,
            localTime: formatLocal(m.createdAt, m.tz),
            timezone: m.tz,
            retrievability:
                (retrievability(m, now) * 100).roundToDouble() / 100,
          ),
      ],
    );
    // The embedder call inside _gists is as fallible as the LLM call, and both
    // must leave the seed labile for the next dream rather than escape and
    // discard the reports already collected.
    List<Memory> absorbed;
    List<Memory> gists;
    try {
      final decision = await Future.sync(() => adjudicate(request));
      final texts = <String>[];
      for (final t in decision.memories) {
        final c = cleanText(t, config.textMax);
        if (c.isNotEmpty && !texts.contains(c)) texts.add(c);
      }
      // The verdict names the older candidates the seed supersedes; the ones it
      // does not name were merely offered and are left untouched.
      final picked = decision.absorbedIds.toSet();
      absorbed = [
        cluster.first,
        ...cluster.skip(1).where((m) => picked.contains(m.id)),
      ];
      gists = texts.length > absorbed.length
          ? const <Memory>[]
          : await _gists(texts, absorbed, now);
    } catch (e) {
      return DreamReport(
          action: DreamAction.error,
          before: cluster,
          after: const [],
          error: '$e');
    }
    if (gists.isEmpty) {
      await _put(cluster.first.copyWith(consolidated: true));
      return DreamReport(
          action: DreamAction.keep, before: cluster, after: const []);
    }
    await store.transaction(() async {
      for (final m in absorbed) {
        await store.remove(m.id);
      }
      for (final g in gists) {
        await store.put(g);
      }
    });
    for (final m in absorbed) {
      _dropTrace(m.id);
    }
    for (final g in gists) {
      _setTrace(g);
    }
    return DreamReport(
        action: DreamAction.replace, before: cluster, after: gists);
  }

  /// Builds the replacement traces, or an empty list when [texts] is empty
  /// or any text is unrelated to every member (confabulation guard). A gist
  /// inherits the strongest member's stability plus the *live* evidence of
  /// the others (`max S_i + Σ strength_i`, capped) and conserves the
  /// cluster's total strength (`R = Σ strength_i / S`, encoded back into
  /// `lastRecall`), so consolidation neither refreshes nor inflates.
  Future<List<Memory>> _gists(
      List<String> texts, List<Memory> cluster, int now) async {
    if (texts.isEmpty) return const [];
    final vectors = [
      for (final v in await embedder.embedDocuments(texts))
        // Throwing leaves the seed labile, so the next dream retries it.
        _vector(v) ?? (throw StateError('embedder dimension mismatch')),
    ];
    final sims = _scores(cluster, vectors);
    for (var j = 0; j < vectors.length; j++) {
      var best = double
          .negativeInfinity; // a true max, so gistMinCosine <= 0 still rejects
      for (var i = 0; i < cluster.length; i++) {
        best = math.max(best, sims[i * vectors.length + j]);
      }
      if (best < config.gistMinCosine) return const [];
    }
    final strongest =
        cluster.reduce((a, b) => a.stability >= b.stability ? a : b);
    var sumSR = 0.0;
    for (final m in cluster) {
      sumSR += strength(m, now);
    }
    final stability =
        _clampStability(strongest.stability + sumSR - strength(strongest, now));
    final r = math.min(1.0, sumSR / stability);
    // Half-lives already elapsed for the gist (64 ⇒ R underflows to 0).
    final elapsed = r > 0 ? math.min(64.0, -math.log(r) / math.ln2) : 64.0;
    final lastRecall = now - (stability * elapsed).round();
    // The gist was stated when its newest member was, not when the dream ran,
    // so a later update can still reach it as an older candidate.
    // = the seed; candidates are no newer, and ties keep the first.
    final newest = cluster.reduce((a, b) => b.createdAt > a.createdAt ? b : a);
    return [
      for (var i = 0; i < texts.length; i++)
        Memory(
          id: ulid(now * 1000),
          text: texts[i],
          createdAt: newest.createdAt,
          tz: newest.tz,
          lastRecall: lastRecall,
          stability: stability,
          consolidated: true,
          modelId: embedder.modelId,
          vector: vectors[i],
          cue: newest.cue,
        ),
    ];
  }
}
