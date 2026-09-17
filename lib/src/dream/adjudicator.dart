import 'dart:async';
import 'dart:convert';

import 'prompts.dart';

/// The dream-phase LLM callback — implemented by the application.
///
/// The engine gathers a cluster of related traces and asks the callback to
/// either keep them or replace them with gist propositions:
///
/// ```dart
/// Future<DreamDecision> adjudicate(DreamRequest request) async {
///   final raw = await myLlm.generateJson(request.buildPrompt());
///   return DreamDecision.parseJson(raw);
/// }
/// ```
///
/// If the callback throws, the cluster is left untouched and retried in the
/// next dream. Text shortening, stability inheritance and the transactional
/// delete-and-insert are enforced by the engine, not the callback.
typedef DreamAdjudicator = FutureOr<DreamDecision> Function(
    DreamRequest request);

/// One trace inside a [DreamRequest].
class DreamMember {
  /// Creates a member.
  const DreamMember({
    required this.id,
    required this.text,
    required this.localTime,
    required this.timezone,
    required this.retrievability,
  });

  /// Trace id.
  final String id;

  /// Proposition text.
  final String text;

  /// Local datetime the proposition was stated, e.g.
  /// `'2026-05-20 14:00 +09:00'` — the reference point for resolving any
  /// relative date words in [text].
  final String localTime;

  /// Stored timezone field `'IANA;+HH:MM'`.
  final String timezone;

  /// Retrievability `R` at dream time (rounded).
  final double retrievability;

  /// JSON form handed to the LLM.
  Map<String, Object?> toJson() => {
        'id': id,
        'text': text,
        'local_time': localTime,
        'timezone': timezone,
        'R': retrievability,
      };
}

/// One cluster of semantically close traces awaiting an LLM verdict.
class DreamRequest {
  /// Creates a request.
  const DreamRequest({required this.currentLocalTime, required this.members});

  /// Current local datetime string, e.g. `'2026-06-12 09:30 +09:00'`.
  final String currentLocalTime;

  /// The cluster members (seed first, then neighbours by cosine).
  final List<DreamMember> members;

  /// The members as pretty-printed JSON.
  String membersAsJson() => const JsonEncoder.withIndent('  ')
      .convert(members.map((m) => m.toJson()).toList());

  /// Builds the complete consolidation prompt, ready for any LLM in JSON
  /// mode. See [EngramPrompts.buildDreamInstruction].
  String buildPrompt({EngramLocale locale = EngramLocale.ja}) =>
      EngramPrompts.buildDreamInstruction(
        currentTime: currentLocalTime,
        listing: membersAsJson(),
        locale: locale,
      );
}

/// The LLM's verdict for one cluster: keep it, or replace it with
/// [memories] (one or more self-contained propositions).
class DreamDecision {
  /// Replace the cluster with [memories]; an empty list means keep.
  const DreamDecision(this.memories);

  /// "Leave the cluster untouched."
  const DreamDecision.keep() : memories = const [];

  /// Parses a raw LLM response leniently: strips code fences, extracts the
  /// first JSON object from surrounding prose, accepts `memories` as strings
  /// or `{text: ...}` objects, and treats `action: keep` or an empty list as
  /// keep. Never throws — malformed input yields [DreamDecision.keep].
  factory DreamDecision.parseJson(String? raw) {
    final obj = relaxedJsonDecode(raw);
    if (obj is! Map) return const DreamDecision.keep();
    final action = (obj['action'] ?? '').toString().trim().toLowerCase();
    final memsRaw = obj['memories'];
    if (action == 'keep' || memsRaw is! List) {
      return const DreamDecision.keep();
    }
    return DreamDecision([
      for (final item in memsRaw)
        if (_textOf(item).isNotEmpty) _textOf(item),
    ]);
  }

  static String _textOf(Object? item) =>
      (item is Map ? item['text'] : item)?.toString().trim() ?? '';

  /// Replacement propositions (empty = keep).
  final List<String> memories;

  /// Whether the verdict is "keep".
  bool get isKeep => memories.isEmpty;
}

/// Best-effort JSON parse tolerant of code fences and surrounding prose.
/// Returns the decoded object, or `null` when nothing parseable is found.
Object? relaxedJsonDecode(String? text) {
  if (text == null || text.trim().isEmpty) return null;
  final t = text
      .trim()
      .replaceAll(RegExp(r'^```(?:json)?', multiLine: true), '')
      .replaceAll(RegExp(r'```$', multiLine: true), '')
      .trim();
  try {
    return jsonDecode(t);
  } on FormatException {
    final m = RegExp(r'\{.*\}|\[.*\]', dotAll: true).firstMatch(t);
    if (m == null) return null;
    try {
      return jsonDecode(m.group(0)!);
    } on FormatException {
      return null;
    }
  }
}
