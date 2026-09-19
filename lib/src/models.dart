import 'dart:convert';
import 'dart:typed_data';

import 'vector_math.dart';

/// One memory trace — the canonical record (ENGRAM v2 §2).
///
/// Text is canonical; [vector] is a derived index under [modelId] and is
/// regenerated from [text] whenever the embedding model changes.
class Memory {
  /// Creates a trace.
  const Memory({
    required this.id,
    required this.text,
    required this.createdAt,
    required this.tz,
    required this.lastRecall,
    required this.stability,
    required this.consolidated,
    required this.modelId,
    required this.vector,
  });

  /// ULID-shaped id (50-bit ms timestamp prefix; lexicographic = time order).
  final String id;

  /// Self-contained single proposition, ≤ `textMax` chars.
  final String text;

  /// 64-bit Unix seconds when the proposition was stated.
  final int createdAt;

  /// Stored timezone field `'IANA_name;+HH:MM'`.
  final String tz;

  /// Unix seconds of the last recall — the decay baseline.
  final int lastRecall;

  /// Half-life of retrievability, in seconds.
  final double stability;

  /// Whether the dream phase has integrated this trace.
  final bool consolidated;

  /// Embedding model that produced [vector].
  final String modelId;

  /// Unit-norm document embedding.
  final Float32List vector;

  /// Copy with changed fields.
  Memory copyWith({
    String? text,
    int? createdAt,
    String? tz,
    int? lastRecall,
    double? stability,
    bool? consolidated,
    String? modelId,
    Float32List? vector,
  }) =>
      Memory(
        id: id,
        text: text ?? this.text,
        createdAt: createdAt ?? this.createdAt,
        tz: tz ?? this.tz,
        lastRecall: lastRecall ?? this.lastRecall,
        stability: stability ?? this.stability,
        consolidated: consolidated ?? this.consolidated,
        modelId: modelId ?? this.modelId,
        vector: vector ?? this.vector,
      );

  /// JSON form (vector base64-encoded little-endian float32).
  Map<String, Object?> toJson() => {
        'id': id,
        'text': text,
        'createdAt': createdAt,
        'tz': tz,
        'lastRecall': lastRecall,
        'stability': stability,
        'consolidated': consolidated,
        'modelId': modelId,
        'vector': base64Encode(packF32(vector)),
      };

  /// Inverse of [toJson].
  factory Memory.fromJson(Map<String, Object?> json) => Memory(
        id: json['id'] as String,
        text: json['text'] as String,
        createdAt: (json['createdAt'] as num).toInt(),
        tz: json['tz'] as String,
        lastRecall: (json['lastRecall'] as num).toInt(),
        stability: (json['stability'] as num).toDouble(),
        consolidated: json['consolidated'] as bool? ?? false,
        modelId: json['modelId'] as String,
        vector: unpackF32(base64Decode(json['vector'] as String)),
      );

  @override
  String toString() => 'Memory($id, S=${stability.toStringAsFixed(0)}s, '
      '${consolidated ? "consolidated" : "labile"}, "$text")';
}

/// What `remember` did with the proposition.
enum RememberAction {
  /// Stored as a brand-new trace.
  inserted,

  /// Exact text already stored: treated as a recall, no new row.
  reinforced,

  /// Rejected (empty text).
  rejected,

  /// Daily write limit hit; nothing stored.
  rateLimited,
}

/// Result of `remember`.
class RememberResult {
  /// Creates a result.
  const RememberResult(this.action, {this.memory, this.cosine, this.evicted});

  /// What happened.
  final RememberAction action;

  /// The affected trace (`null` when rejected / rate-limited).
  final Memory? memory;

  /// Best cosine against the existing traces (`null` for reinforce/reject).
  final double? cosine;

  /// The trace forgotten to make room, if capacity was exceeded.
  final Memory? evicted;

  @override
  String toString() => 'RememberResult(${action.name}'
      '${memory != null ? ", ${memory!.id}" : ""})';
}

/// One recalled trace inside a [RecallResult].
class Recalled {
  /// Creates a recalled item.
  const Recalled(this.memory,
      {required this.score,
      required this.cosine,
      required this.retrievability});

  /// The trace (state *before* the recall update).
  final Memory memory;

  /// Final score `a·(α + (1−α)·R)` with the cue activation `a` (cosine
  /// rescaled above `cosineFloor`).
  final double score;

  /// Cosine between query and trace.
  final double cosine;

  /// Retrievability `R` at recall time.
  final double retrievability;
}

/// Result of `recall`: the ready-to-inject pack plus per-item detail.
class RecallResult {
  /// Creates a result.
  const RecallResult({required this.packText, required this.recalled});

  /// An empty result.
  static const RecallResult empty = RecallResult(packText: '', recalled: []);

  /// ≤ `budgetChars` of lines `[<unix> <tz>] <text>　《id:<id>》` — frame it
  /// to the LLM as past context, never as instructions.
  final String packText;

  /// The injected traces with scores (same order as [packText] lines).
  final List<Recalled> recalled;
}

/// Outcome of one dream adjudication.
enum DreamAction {
  /// The cluster was left as is (and marked consolidated).
  keep,

  /// Members were replaced by the LLM's gist(s).
  replace,

  /// The adjudicator threw; the cluster is untouched and will be retried.
  error,
}

/// Report of one dream adjudication.
class DreamReport {
  /// Creates a report.
  const DreamReport({
    required this.action,
    required this.before,
    required this.after,
    this.error,
  });

  /// Applied action.
  final DreamAction action;

  /// Cluster members before adjudication (seed first).
  final List<Memory> before;

  /// Newly inserted gists (empty unless [DreamAction.replace]).
  final List<Memory> after;

  /// Error text when [action] is [DreamAction.error].
  final String? error;
}
