/// Text helpers shared by the write and dream paths.
library;

/// Normalises a proposition for storage: strips the pack delimiters `《》`
/// (so a stored text can never spoof an injected id), collapses whitespace
/// and newlines to single spaces, trims, then shortens to [maxChars].
String cleanText(String text, int maxChars) => shorten(
      text
          .replaceAll(RegExp('[《》]'), '')
          .replaceAll(RegExp(r'\s+'), ' ')
          .trim(),
      maxChars,
    );

/// Truncates [text] to at most [maxChars], preferring a sentence/clause
/// boundary in the second half of the cut.
String shorten(String text, int maxChars) {
  if (text.length <= maxChars) return text;
  final cut = text.substring(0, maxChars);
  for (final sep in const ['。', '．', '.', '、', ' ']) {
    final idx = cut.lastIndexOf(sep);
    if (idx > maxChars * 0.5) return cut.substring(0, idx + 1);
  }
  return cut;
}

final RegExp _cueBreak = RegExp(r'\n+|(?<=[。！？])|(?<=[.!?])\s+');

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
