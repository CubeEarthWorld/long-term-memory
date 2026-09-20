/// Lenient JSON helpers shared by the dream verdict and the extraction
/// fallback parsers.
library;

import 'dart:convert';

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

/// The non-empty trimmed texts of a decoded list whose items are strings or
/// `{text: ...}` objects; anything else yields an empty list.
List<String> jsonTexts(Object? list) {
  if (list is! List) return const [];
  final out = <String>[];
  for (final item in list) {
    final text = _text(item);
    if (text.isNotEmpty) out.add(text);
  }
  return out;
}

String _text(Object? item) =>
    (item is Map ? item['text'] : item)?.toString().trim() ?? '';
