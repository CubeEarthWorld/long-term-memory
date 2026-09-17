/// The ENGRAM v2 engine: traces with a forgetting curve, wake-phase verbs
/// (remember / recall / forget) and a sleep-phase dream.
library;

import 'dart:async';
import 'dart:math' as math;
import 'dart:typed_data';

import '../config.dart';
import '../dream/adjudicator.dart';
import '../embedder.dart';
import '../models.dart';
import '../store/memory_store.dart';
import '../text.dart';
import '../timezone.dart';
import '../ulid.dart';
import '../vector_math.dart';

/// Seeds examined per dream, as a multiple of the LLM budget (bounds the
/// cost of a dream to O(budget · capacity · dim)).
const int _seedsPerBudget = 8;

/// A long-term memory for LLM applications (ENGRAM v2, see SPEC.md).
///
/// Every trace is held in RAM; the injected [MemoryStore] is the durable
/// substrate, the [Embedder] the index, and — for [dream] only — a
/// [DreamAdjudicator] backed by any LLM. Every entry point accepts an
/// explicit Unix time (`nowUnix`, seconds) and a [MemoryTimezone]; omitted
/// values fall back to the injected [clock] / `defaultTimezone`, then to
/// the system clock / UTC. Public methods are serialized, so an instance is
/// safe to call from interleaving async code. One process owns a store.
class EngramMemory {
  /// Creates an engine over [store] and [embedder].
  EngramMemory({
    required this.store,
    required this.embedder,
    this.config = const EngramConfig(),
    int Function()? clock,
    MemoryTimezone? defaultTimezone,
  })  : _clock = clock,
        _defaultTz = defaultTimezone ?? MemoryTimezone.utc {
    _ulid = UlidGenerator(millis: () => nowUnix() * 1000);
  }

  /// The injected storage backend.
  final MemoryStore store;

  /// The injected embedding model.
  final Embedder embedder;

  /// Active parameters.
  final EngramConfig config;

  final int Function()? _clock;
  final MemoryTimezone _defaultTz;
  late final UlidGenerator _ulid;
  final Map<String, Memory> _traces = {};
  final List<int> _writeTimes = [];
  final Map<String, Recalled> _lastRecall = {};
  Future<void> _chain = Future<void>.value();

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
  String nowLocal({int? nowUnix, MemoryTimezone? timezone}) => formatLocal(
      nowUnix ?? this.nowUnix(), (timezone ?? _defaultTz).storageField);

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

  double _clampStability(double s) =>
      math.min(math.max(s, 1.0), config.maxStability);

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
        _traces.clear();
        for (final raw in await store.loadAll()) {
          final m = raw.copyWith(
            stability: _clampStability(raw.stability),
            lastRecall: math.min(raw.lastRecall, now),
            createdAt: math.min(raw.createdAt, now),
          );
          _traces[m.id] = m;
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
      for (var j = 0; j < batch.length; j++) {
        await _put(batch[j].copyWith(
            modelId: embedder.modelId, vector: l2Normalized(vectors[j])));
      }
    }
  }

  Iterable<Memory> get _indexed =>
      _traces.values.where((m) => m.modelId == embedder.modelId);

  List<Memory> get _stale => _traces.values
      .where((m) => m.modelId != embedder.modelId)
      .toList(growable: false);

  /// Erases everything.
  Future<void> reset() => _serialize(() async {
        await store.clear();
        _traces.clear();
        _writeTimes.clear();
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
    int? nowUnix,
    MemoryTimezone? timezone,
  }) =>
      _serialize(() async {
        final now = nowUnix ?? this.nowUnix();
        text = cleanText(text, config.textMax);
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
          v = l2Normalized((await embedder.embedDocuments([text]))[0]);
        } catch (_) {
          v = null; // embedder offline: keep the text, index it later
        }
        var best = 0.0;
        Memory? nearest;
        if (v != null) {
          for (final m in _indexed) {
            final c = dot(m.vector, v);
            if (c > best) {
              best = c;
              nearest = m;
            }
          }
        }
        final related = best >= config.thetaRelated;
        final m = Memory(
          id: _ulid.next(),
          text: text,
          createdAt: now,
          tz: (timezone ?? _defaultTz).storageField,
          lastRecall: now,
          stability: _clampStability(
              config.initialStability * salience.clamp(0.0, 10.0)),
          consolidated: v != null && !related && _stale.isEmpty,
          modelId: v == null ? '' : embedder.modelId,
          vector: v ?? Float32List(0),
        );
        // One operation = one commit.
        final evicted = await store.transaction(() async {
          await _put(m);
          if (related && nearest!.consolidated) {
            // Reconsolidation: the reactivated old trace becomes labile too,
            // so the pair is adjudicated at the old trace's (higher) priority.
            await _put(nearest.copyWith(consolidated: false));
          }
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
        final parts = cues(query, config.maxCues);
        if (parts.isEmpty || _traces.isEmpty) return RecallResult.empty;
        List<Float32List> qs;
        try {
          qs = [
            for (final q in await embedder.embedQueries(parts)) l2Normalized(q),
          ];
        } catch (_) {
          return RecallResult.empty; // embedder offline: nothing to cue with
        }
        final scored = <Recalled>[];
        var top = 0.0;
        for (final m in _indexed) {
          var cos = double.negativeInfinity;
          for (final q in qs) {
            cos = math.max(cos, dot(m.vector, q));
          }
          final r = retrievability(m, now);
          final score =
              _activation(cos) * (config.alpha + (1 - config.alpha) * r);
          top = math.max(top, score);
          scored.add(Recalled(m, score: score, cosine: cos, retrievability: r));
        }
        final floor = math.max(config.minScore, config.relativeScore * top);
        final pool = scored.where((c) => c.score >= floor).toList()
          ..sort((a, b) => b.score.compareTo(a.score));
        final lines = <String>[];
        final packed = <Recalled>[];
        var used = 0;
        _lastRecall.clear();
        // One operation = one commit.
        await store.transaction(() async {
          for (final c in _mmr(pool)) {
            final m = c.memory;
            final line = '[${m.createdAt} ${m.tz}] ${m.text}　《id:${m.id}》\n';
            if (lines.isNotEmpty && used + line.length > config.budgetChars) {
              continue;
            }
            lines.add(line);
            used += line.length;
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
          final m = c == null ? null : _traces[c.memory.id];
          if (m == null) continue;
          final partial = 1 +
              config.spacingGain *
                  0.5 *
                  _activation(c!.cosine) *
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
      for (var i = 0; i < pool.length; i++) {
        if (!alive[i]) continue;
        maxSim[i] = math.max(
            maxSim[i], dot(pool[i].memory.vector, pool[bestI].memory.vector));
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
    _traces[m.id] = m;
    await store.put(m);
  }

  Future<void> _remove(String id) async {
    _traces.remove(id);
    await store.remove(id);
  }

  /// Forgets the lowest-ranked trace while over capacity. Traces inside the
  /// grace period are protected unless they make up more than a tenth of
  /// the capacity (so a flood evicts its own members) or nothing older
  /// exists. Returns the last victim.
  Future<Memory?> _enforceCapacity(int now) async {
    Memory? victim;
    while (_traces.length > config.capacity) {
      final young = <Memory>[];
      Memory? weakest;
      var weakestRank = double.infinity;
      void consider(Memory m) {
        final r = _rank(m, now);
        if (r < weakestRank) {
          weakest = m;
          weakestRank = r;
        }
      }

      for (final m in _traces.values) {
        final age = now - m.createdAt;
        if (age >= 0 && age < config.gracePeriod) {
          young.add(m);
        } else {
          consider(m);
        }
      }
      if (weakest == null || young.length > config.capacity ~/ 10) {
        young.forEach(consider);
      }
      await _remove(weakest!.id);
      victim = weakest;
    }
    return victim;
  }

  // ================================================================== //
  // sleep phase (SPEC §5)
  // ================================================================== //

  /// The clusters the next dream would hand to the LLM (seed first),
  /// without calling it — for inspection UIs.
  Future<List<List<Memory>>> clusters() => _serialize(() async {
        final out = <List<Memory>>[];
        final taken = <String>{};
        for (final seed in _seeds) {
          if (taken.contains(seed.id)) continue;
          final cluster = _cluster(seed);
          if (cluster.length < 2) continue;
          out.add(cluster);
          taken.addAll(cluster.map((m) => m.id));
        }
        return out;
      });

  /// Labile traces, most stable first: what carries the most accumulated
  /// evidence is integrated first (a correction of an important fact is
  /// seeded by that fact, ahead of fresh junk pairs).
  List<Memory> get _seeds => _indexed.where((m) => !m.consolidated).toList()
    ..sort((a, b) => b.stability.compareTo(a.stability));

  List<Memory> _cluster(Memory seed) {
    final near = <(double, Memory)>[
      for (final m in _indexed)
        if (m.id != seed.id &&
            dot(m.vector, seed.vector) >= config.thetaRelated)
          (dot(m.vector, seed.vector), m),
    ]..sort((a, b) => b.$1.compareTo(a.$1));
    return [seed, ...near.take(config.dreamMaxMembers - 1).map((e) => e.$2)];
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
        final failed = <String>{}; // members of clusters whose LLM call threw
        for (final s in _seeds.take(_seedsPerBudget * left)) {
          final seed = _traces[s.id];
          if (seed == null || seed.consolidated || failed.contains(s.id)) {
            continue;
          }
          if (left <= 0) break;
          final cluster = _cluster(seed);
          if (cluster.length < 2) {
            await _put(seed.copyWith(consolidated: true));
            continue;
          }
          left -= 1;
          final report = await _adjudicate(adjudicate, cluster, now, tz);
          if (report.action == DreamAction.error) {
            failed.addAll(cluster.map((m) => m.id));
          }
          reports.add(report);
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
    DreamDecision decision;
    try {
      decision = await Future.sync(() => adjudicate(request));
    } catch (e) {
      return DreamReport(
          action: DreamAction.error,
          before: cluster,
          after: const [],
          error: '$e');
    }
    final texts = <String>[];
    for (final t in decision.memories) {
      final c = cleanText(t, config.textMax);
      if (c.isNotEmpty && !texts.contains(c)) texts.add(c);
    }
    final gists = texts.length > cluster.length
        ? const <Memory>[]
        : await _gists(texts, cluster, now, tz);
    if (gists.isEmpty) {
      for (final m in cluster) {
        await _put(m.copyWith(consolidated: true));
      }
      return DreamReport(
          action: DreamAction.keep, before: cluster, after: const []);
    }
    await store.transaction(() async {
      for (final m in cluster) {
        await store.remove(m.id);
      }
      for (final g in gists) {
        await store.put(g);
      }
    });
    for (final m in cluster) {
      _traces.remove(m.id);
    }
    for (final g in gists) {
      _traces[g.id] = g;
    }
    await _enforceCapacity(now);
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
      List<String> texts, List<Memory> cluster, int now, String tz) async {
    if (texts.isEmpty) return const [];
    final vectors = [
      for (final v in await embedder.embedDocuments(texts)) l2Normalized(v),
    ];
    for (final v in vectors) {
      var best = 0.0;
      for (final m in cluster) {
        best = math.max(best, dot(m.vector, v));
      }
      if (best < config.gistMinCosine) return const [];
    }
    var maxS = 0.0;
    var sumSR = 0.0;
    for (final m in cluster) {
      maxS = math.max(maxS, m.stability);
      sumSR += strength(m, now);
    }
    final strongest =
        cluster.reduce((a, b) => a.stability >= b.stability ? a : b);
    final stability = _clampStability(maxS + sumSR - strength(strongest, now));
    final r = math.min(1.0, sumSR / stability);
    // Half-lives already elapsed for the gist (64 ⇒ R underflows to 0).
    final elapsed = r > 0 ? math.min(64.0, -math.log(r) / math.ln2) : 64.0;
    final lastRecall = now - (stability * elapsed).round();
    return [
      for (var i = 0; i < texts.length; i++)
        Memory(
          id: _ulid.next(),
          text: texts[i],
          createdAt: now,
          tz: tz,
          lastRecall: lastRecall,
          stability: stability,
          consolidated: true,
          modelId: embedder.modelId,
          vector: vectors[i],
        ),
    ];
  }
}
