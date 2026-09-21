/// Text helpers shared by the write and dream paths.
library;

/// The whitespace class, spelled out so it is identical to Python's `\s`:
/// Dart's `\s` omits U+001C-U+001F and U+0085, Python's omits U+FEFF.
const String whitespaceClass = r'[\s-﻿]';

/// Normalises a proposition for storage: strips the pack delimiters `《》`
/// (so a stored text can never spoof an injected id), collapses whitespace
/// and newlines to single spaces, trims, then shortens to [maxChars].
String cleanText(String text, int maxChars) => shorten(
      text
          .replaceAll(RegExp('[《》]'), '')
          .replaceAll(RegExp('$whitespaceClass+'), ' ')
          .trim(),
      maxChars,
    ).trim(); // a boundary cut keeps its separator, which may be a space

/// Truncates [text] to at most [maxChars], preferring a sentence/clause
/// boundary in the second half of the cut.
///
/// Counts code points, not UTF-16 code units, so the limit means the same
/// thing as Python's `len()` and a cut can never split a surrogate pair.
String shorten(String text, int maxChars) {
  final runes = text.runes.toList();
  if (runes.length <= maxChars) return text;
  final cut = runes.sublist(0, maxChars);
  for (final sep in const ['。', '．', '.', '、', ' ']) {
    final idx = cut.lastIndexOf(sep.codeUnitAt(0));
    if (idx > maxChars * 0.5) {
      return String.fromCharCodes(cut.sublist(0, idx + 1));
    }
  }
  return String.fromCharCodes(cut);
}

final RegExp _cueBreak = RegExp('\\n+|(?<=[。！？])|(?<=[.!?])$whitespaceClass+');

/// Splits a query into cues (lines, then sentences) so a trace relevant to
/// any part of a long multi-topic turn can surface. More than [maxCues]
/// pieces are coalesced into [maxCues] contiguous buckets (nothing is
/// dropped).
List<String> cues(String text, int maxCues) {
  final parts = text
      .split(_cueBreak)
      .map((s) => s.trim())
      .where((s) => s.isNotEmpty)
      .toList();
  final n = parts.length;
  if (n <= maxCues) return parts;
  return [
    for (var i = 0; i < maxCues; i++)
      parts.sublist(i * n ~/ maxCues, (i + 1) * n ~/ maxCues).join(' '),
  ];
}
