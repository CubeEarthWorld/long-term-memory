import 'dart:async';
import 'dart:convert';

import '../json.dart';
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
/// If the callback throws — including the [FormatException] that
/// [DreamDecision.parseJson] raises on an unparsable answer — the cluster is
/// left untouched and retried in the next dream. Text shortening, stability
/// inheritance and the transactional delete-and-insert are enforced by the
/// engine, not the callback.
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
  /// Replace the seed and [absorbedIds] with [memories]; an empty list of
  /// memories means keep. Candidates not named in [absorbedIds] are untouched.
  const DreamDecision(this.memories, {this.absorbedIds = const []});

  /// "Leave the cluster untouched."
  const DreamDecision.keep()
      : memories = const [],
        absorbedIds = const [];

  /// Parses a raw LLM response leniently: strips code fences, extracts the
  /// first JSON object from surrounding prose, accepts `memories` as strings
  /// or `{text: ...}` objects, and treats `action: keep` or an empty list as
  /// keep. Returns null for an empty or unparsable answer so the caller can
  /// throw and let the next dream retry - a truncated reasoning model must
  /// not be read as "keep".
  static DreamDecision? parse(String? raw) {
    final obj = relaxedJsonDecode(raw);
    if (obj is! Map) return null;
    final action = (obj['action'] ?? '').toString().trim().toLowerCase();
    if (action == 'keep') return const DreamDecision.keep();
    // A missing or mistyped `memories` is a broken answer, not an empty one.
    if (obj['memories'] is! List) return null;
    final texts = jsonTexts(obj['memories']);
    if (texts.isEmpty) return const DreamDecision.keep();
    final ids = obj['ids'];
    return DreamDecision(texts, absorbedIds: [
      if (ids is List)
        for (final i in ids)
          if ('$i'.isNotEmpty) '$i',
    ]);
  }

  /// Lenient parse that throws a [FormatException] on an empty or unparsable
  /// answer, so the engine leaves the cluster labile and the next dream retries
  /// it. Silence must never be read as "keep" (SPEC §5) — a truncated
  /// reasoning model would otherwise settle the trace permanently.
  factory DreamDecision.parseJson(String? raw) =>
      parse(raw) ?? (throw const FormatException('unparsable dream verdict'));

  /// Replacement propositions (empty = keep).
  final List<String> memories;

  /// Ids of the older candidates the seed supersedes.
  final List<String> absorbedIds;

  /// Whether the verdict is "keep".
  bool get isKeep => memories.isEmpty;
}
