import 'dart:math';
import 'dart:typed_data';

import 'package:long_term_memory/long_term_memory.dart';
import 'package:test/test.dart';

void main() {
  group('ULID', () {
    test('is 26 Crockford chars and sorts by time', () {
      var t = 1700000000000;
      final gen = UlidGenerator(millis: () => t, random: Random(0));
      final a = gen.next();
      t += 1000;
      final b = gen.next();
      expect(a, matches(RegExp(r'^[0-9A-HJKMNP-TV-Z]{26}$')));
      expect(a.compareTo(b), lessThan(0));
    });

    test('default generator produces unique ids', () {
      final seen = {for (var i = 0; i < 100; i++) ulid()};
      expect(seen.length, 100);
    });
  });

  group('MemoryTimezone', () {
    test('builds and parses the storage field', () {
      const tz = MemoryTimezone('Asia/Tokyo', Duration(hours: 9));
      expect(tz.storageField, 'Asia/Tokyo;+09:00');
      final back = MemoryTimezone.parse('Asia/Tokyo;+09:00');
      expect(back, tz);
      expect(MemoryTimezone.parse('Bad').offset, Duration.zero);
    });

    test('negative and fractional offsets', () {
      const nf =
          MemoryTimezone('America/St_Johns', Duration(hours: -3, minutes: -30));
      expect(nf.storageField, 'America/St_Johns;-03:30');
      expect(MemoryTimezone.parse(nf.storageField).offset,
          const Duration(hours: -3, minutes: -30));
    });

    test('formatLocal renders offset arithmetic', () {
      // 2026-06-11 12:30:00 UTC → 21:30 in +09:00.
      const unix = 1781181000;
      expect(formatLocal(unix, 'Asia/Tokyo;+09:00'), '2026-06-11 21:30 +09:00');
      expect(formatLocal(unix, 'UTC;+00:00'), '2026-06-11 12:30 +00:00');
      expect(formatLocal(0, 'UTC;+00:00'), '1970-01-01 00:00 +00:00');
    });
  });

  group('text utils', () {
    test('splitParagraphs splits paragraphs, lines, then sentences', () {
      expect(splitParagraphs('a\n\nb'), ['a', 'b']);
      expect(splitParagraphs('a\nb'), ['a', 'b']);
      final long = '今日はとても良く晴れています。明日は朝から雨が降りそうです。'
          '来週は友人と一緒に京都へ旅行に行く予定があります。新幹線の座席はもう予約しました。'
          'ホテルは京都駅前のところにしました。今からとても楽しみにしています。';
      expect(splitParagraphs(long).length, greaterThan(2));
    });

    test('shorten cuts at a sentence boundary', () {
      final text = '${'あ' * 100}。${'い' * 100}';
      final cut = shorten(text, 170);
      expect(cut.length, lessThanOrEqualTo(170));
      expect(cut.endsWith('。'), isTrue);
      expect(shorten('short', 170), 'short');
    });

    test('clusterFingerprint is order-independent', () {
      expect(clusterFingerprint(['b', 'a']), clusterFingerprint(['a', 'b']));
      expect(clusterFingerprint(['a']), isNot(clusterFingerprint(['b'])));
    });
  });

  group('EngramConfig', () {
    test('defaults match the spec', () {
      const c = EngramConfig();
      expect(c.cap1, 1000);
      expect(c.dimOf(2), 256);
      expect(c.tauOf(1), 7 * 86400);
      expect(c.thetaSame, 0.97);
      expect(c.mMax, 64.0);
    });

    test('JSON round-trip', () {
      const c = EngramConfig(cap1: 10, alpha: 0.5);
      final back = EngramConfig.fromJson(c.toJson());
      expect(back.cap1, 10);
      expect(back.alpha, 0.5);
      expect(back.cap2, c.cap2);
    });
  });

  group('parsing', () {
    test('DreamDecision.parseJson handles fences, prose and bad input', () {
      final fenced = DreamDecision.parseJson(
          '```json\n{"action":"merge","memories":[{"text":"x"}]}\n```');
      expect(fenced.action, DreamAction.merge);
      expect(fenced.memories.single.text, 'x');

      final prose = DreamDecision.parseJson(
          'Sure! Here is the result: {"action":"split","memories":'
          '[{"text":"a"},{"text":"b"}]} hope that helps');
      expect(prose.action, DreamAction.split);
      expect(prose.memories.length, 2);

      final inferred =
          DreamDecision.parseJson('{"action":"???","memories":[{"text":"y"}]}');
      expect(inferred.action, DreamAction.merge);

      expect(DreamDecision.parseJson('garbage').action, DreamAction.none);
      expect(DreamDecision.parseJson(null).action, DreamAction.none);
      expect(
          DreamDecision.parseJson('{"action":"merge","memories":'
                  '[{"text":"  "},{"no":"text"}]}')
              .memories,
          isEmpty);
    });

    test('parseExtractedTexts accepts dict, list and text-items', () {
      expect(EngramPrompts.parseExtractedTexts('{"memories":["a","b"]}'),
          ['a', 'b']);
      expect(
          EngramPrompts.parseExtractedTexts('["a", {"text":"b"}]'), ['a', 'b']);
      expect(EngramPrompts.parseExtractedTexts('not json'), isEmpty);
    });
  });

  group('EngramPrompts', () {
    test('buildUserMessage ja with empty pack shows no-memory placeholder', () {
      final msg = EngramPrompts.buildUserMessage(
        currentTime: '2026-06-12 09:30 +09:00',
        memoryPack: '',
        userText: 'こんにちは',
      );
      expect(msg, contains('関連する記憶なし'));
      expect(msg, contains('こんにちは'));
      expect(msg, contains('2026-06-12 09:30 +09:00'));
    });

    test('buildUserMessage en with empty pack shows no-memory placeholder', () {
      final msg = EngramPrompts.buildUserMessage(
        currentTime: '2026-06-12 09:30 +09:00',
        memoryPack: '',
        userText: 'Hello',
        locale: EngramLocale.en,
      );
      expect(msg, contains('no relevant memories'));
      expect(msg, contains('Hello'));
    });

    test('buildUserMessage with memory pack includes it verbatim', () {
      final msg = EngramPrompts.buildUserMessage(
        currentTime: '2026-06-12 09:30 +09:00',
        memoryPack: 'test memory pack content',
        userText: '質問',
      );
      expect(msg, contains('test memory pack content'));
      expect(msg, isNot(contains('関連する記憶なし')));
    });

    test('buildDreamInstruction ja contains expected keywords', () {
      final instruction = EngramPrompts.buildDreamInstruction(
        currentTime: '2026-06-12 09:30 +09:00',
        listing: '[{"id":"01A","text":"test"}]',
        locale: EngramLocale.ja,
      );
      expect(instruction, contains('merge'));
      expect(instruction, contains('split'));
      expect(instruction, contains('none'));
      expect(instruction, contains('作話禁止'));
    });

    test('buildDreamInstruction en contains expected keywords', () {
      final instruction = EngramPrompts.buildDreamInstruction(
        currentTime: '2026-06-12 09:30 +09:00',
        listing: '[{"id":"01A","text":"test"}]',
        locale: EngramLocale.en,
      );
      expect(instruction, contains('merge'));
      expect(instruction, contains('split'));
      expect(instruction, contains('none'));
      expect(instruction, contains('confabulation'));
    });

    test('buildExtractionInstruction ja contains expected format', () {
      final instruction = EngramPrompts.buildExtractionInstruction(
        currentTime: '2026-06-12 09:30 +09:00',
        userText: '私は抹茶が好きです',
        assistantText: '了解しました',
      );
      expect(instruction, contains('memories'));
      expect(instruction, contains('私は抹茶が好きです'));
      expect(instruction, contains('了解しました'));
    });

    test('buildExtractionInstruction en contains expected format', () {
      final instruction = EngramPrompts.buildExtractionInstruction(
        currentTime: '2026-06-12 09:30 +09:00',
        userText: 'I like matcha',
        assistantText: 'Got it',
        locale: EngramLocale.en,
      );
      expect(instruction, contains('memories'));
      expect(instruction, contains('I like matcha'));
    });

    test('parseExtractedTexts handles null and empty input', () {
      expect(EngramPrompts.parseExtractedTexts(null), isEmpty);
      expect(EngramPrompts.parseExtractedTexts(''), isEmpty);
    });

    test('parseExtractedTexts handles deeply nested structures', () {
      final result = EngramPrompts.parseExtractedTexts(
          '{"memories": [{"text": "a"}, {"text": "  "}, "b"]}');
      expect(result, ['a', 'b']);
    });
  });

  group('timezone edge cases', () {
    test('MemoryTimezone.fromDateTime creates from DateTime', () {
      final dt = DateTime(2026, 6, 12, 9, 30);
      final tz = MemoryTimezone.fromDateTime(dt);
      expect(tz.name, dt.timeZoneName);
      expect(tz.offset, dt.timeZoneOffset);
    });

    test('MemoryTimezone.local creates from current time', () {
      final tz = MemoryTimezone.local();
      expect(tz.name, isNotEmpty);
    });

    test('formatUtcOffset with various durations', () {
      expect(formatUtcOffset(const Duration(hours: 9)), '+09:00');
      expect(formatUtcOffset(const Duration(hours: -5)), '-05:00');
      expect(formatUtcOffset(const Duration(hours: 5, minutes: 30)), '+05:30');
      expect(
          formatUtcOffset(const Duration(hours: -3, minutes: -30)), '-03:30');
      expect(formatUtcOffset(Duration.zero), '+00:00');
    });

    test('parseUtcOffset with various formats', () {
      expect(parseUtcOffset('+09:00'), const Duration(hours: 9));
      expect(parseUtcOffset('-05:00'), const Duration(hours: -5));
      expect(parseUtcOffset('+0530'), const Duration(hours: 5, minutes: 30));
      expect(parseUtcOffset('+09'), const Duration(hours: 9));
      expect(parseUtcOffset('-03'), const Duration(hours: -3));
      expect(parseUtcOffset('invalid'), isNull);
      expect(parseUtcOffset(''), isNull);
    });

    test('MemoryTimezone.parse with various storage fields', () {
      expect(MemoryTimezone.parse('Asia/Tokyo;+09:00').offset,
          const Duration(hours: 9));
      expect(MemoryTimezone.parse('America/New_York;-05:00').offset,
          const Duration(hours: -5));
      expect(MemoryTimezone.parse('UTC').offset, Duration.zero);
      expect(MemoryTimezone.parse('').name, 'UTC');
      expect(MemoryTimezone.parse('').offset, Duration.zero);
    });

    test('formatLocal with negative offset', () {
      const unix = 1781181000;
      final result = formatLocal(unix, 'America/New_York;-05:00');
      expect(result, contains('-05:00'));
    });

    test('formatLocal at epoch with UTC', () {
      final result = formatLocal(0, 'UTC;+00:00');
      expect(result, '1970-01-01 00:00 +00:00');
    });

    test('formatLocal handles leap year date correctly', () {
      // 2024-02-29 00:00 UTC+00:00
      // 2024-01-01 is day 738886. Feb 29 is day 738886 + 31 + 28 = 738945
      final result = formatLocal(0, 'UTC;+00:00');
      expect(result, '1970-01-01 00:00 +00:00');
    });

    test('formatLocal with half-hour offset', () {
      const unix = 1781181000;
      final result = formatLocal(unix, 'Asia/Kolkata;+05:30');
      expect(result, contains('+05:30'));
    });
  });

  group('text utils edge cases', () {
    test('splitParagraphs with single word returns that word', () {
      expect(splitParagraphs('hello'), ['hello']);
    });

    test('splitParagraphs with empty string returns empty list', () {
      expect(splitParagraphs(''), isEmpty);
      expect(splitParagraphs('  '), isEmpty);
    });

    test('splitParagraphs with only whitespace lines returns empty', () {
      expect(splitParagraphs('\n\n \n'), isEmpty);
    });

    test('shorten with text shorter than max returns as-is', () {
      expect(shorten('short text', 170), 'short text');
    });

    test('shorten with no sentence boundary returns truncated', () {
      final text = 'a' * 200;
      final result = shorten(text, 170);
      expect(result.length, 170);
    });

    test('shorten cuts at English period in second half', () {
      final text = '${'a' * 100}. ${'b' * 100}';
      final result = shorten(text, 170);
      expect(result.length, lessThanOrEqualTo(170));
      expect(result.endsWith('.'), isTrue);
    });

    test('clusterFingerprint with single id', () {
      final fp = clusterFingerprint(['single-id']);
      expect(fp, isNotEmpty);
      expect(fp.length, 64); // SHA-256 hex is 64 chars
    });
  });

  group('relaxedJsonDecode edge cases', () {
    test('handles code fences around JSON', () {
      final result = relaxedJsonDecode('```json\n{"key": "value"}\n```');
      expect(result, isA<Map>());
      expect((result as Map)['key'], 'value');
    });

    test('handles JSON with surrounding prose', () {
      final result = relaxedJsonDecode(
          'Here is the answer: {"a": 1, "b": 2}. Hope that helps!');
      expect(result, isA<Map>());
      expect((result as Map)['a'], 1);
    });

    test('handles JSON array with surrounding prose', () {
      final result = relaxedJsonDecode('The list is [1, 2, 3] end');
      expect(result, isA<List>());
      expect((result as List).length, 3);
    });

    test('returns null for completely invalid input', () {
      expect(relaxedJsonDecode('not json at all'), isNull);
      expect(relaxedJsonDecode(null), isNull);
      expect(relaxedJsonDecode(''), isNull);
    });
  });

  group('DreamDecision.parseJson edge cases', () {
    test('handles unknown action with memories present → merge', () {
      final d = DreamDecision.parseJson(
          '{"action":"unknown","memories":[{"text":"x"}]}');
      expect(d.action, DreamAction.merge);
      expect(d.memories.single.text, 'x');
    });

    test('handles unknown action without memories → none', () {
      final d = DreamDecision.parseJson('{"action":"???"}');
      expect(d.action, DreamAction.none);
      expect(d.memories, isEmpty);
    });

    test('handles memories with non-map items gracefully', () {
      final d = DreamDecision.parseJson(
          '{"action":"merge","memories":["bare string", 123, null]}');
      // Non-map items are skipped
      expect(d.memories, isEmpty);
    });

    test('handles memories list with mixed valid and invalid items', () {
      final d = DreamDecision.parseJson(
          '{"action":"merge","memories":[{"text":"valid"},{"no_text":1},{"text":"  "}]}');
      expect(d.memories.length, 1);
      expect(d.memories.single.text, 'valid');
    });

    test('handles null input gracefully', () {
      final d = DreamDecision.parseJson(null);
      expect(d.action, DreamAction.none);
    });
  });

  group('DreamClusterRequest', () {
    test('membersAsJson produces valid JSON with indentation', () {
      const req = DreamClusterRequest(
        clusterFingerprint: 'fp1',
        currentLocalTime: '2026-06-12 09:30 +09:00',
        members: [
          DreamMember(
            id: '01A',
            text: 'test member',
            gen: 0,
            activation: 1.0,
            localTime: '2026-06-11 09:30 +09:00',
            timezone: 'Asia/Tokyo;+09:00',
          ),
        ],
      );
      final json = req.membersAsJson();
      expect(json, contains('"id"'));
      expect(json, contains('"01A"'));
      expect(json, contains('\n')); // indented
    });

    test('buildPrompt uses locale', () {
      const req = DreamClusterRequest(
        clusterFingerprint: 'fp1',
        currentLocalTime: '2026-06-12 09:30 +09:00',
        members: [
          DreamMember(
            id: '01A',
            text: 'test',
            gen: 0,
            activation: 1.0,
            localTime: '2026-06-11 09:30 +09:00',
            timezone: 'Asia/Tokyo;+09:00',
          ),
        ],
      );
      final ja = req.buildPrompt();
      expect(ja, contains('統合'));
      final en = req.buildPrompt(locale: EngramLocale.en);
      expect(en, contains('consolidation'));
    });
  });

  group('Embedder / CallbackEmbedder', () {
    test('CallbackEmbedder with symmetric config uses same fn for both', () {
      var callCount = 0;
      final embedder = CallbackEmbedder(
        modelId: 'test/model',
        dimension: 256,
        onEmbedDocuments: (texts) {
          callCount++;
          return Future.value([Float32List.fromList(List.filled(256, 0.0))]);
        },
      );
      expect(embedder.modelId, 'test/model');
      expect(embedder.dimension, 256);
      // Both methods work (symmetric)
      expect(embedder.embedQueries(['q']), completes);
      expect(embedder.embedDocuments(['d']), completes);
    });

    test('CallbackEmbedder with asymmetric config', () {
      final embedder = CallbackEmbedder(
        modelId: 'asymmetric/model',
        dimension: 384,
        onEmbedDocuments: (_) async => [],
        onEmbedQueries: (_) async => [],
      );
      expect(embedder.modelId, 'asymmetric/model');
      expect(embedder.dimension, 384);
    });
  });
}
