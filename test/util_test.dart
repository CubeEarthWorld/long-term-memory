import 'dart:math';
import 'dart:typed_data';

import 'package:long_term_memory/long_term_memory.dart';
import 'package:long_term_memory/src/text.dart';
import 'package:long_term_memory/src/timezone.dart';
import 'package:long_term_memory/src/ulid.dart';
import 'package:long_term_memory/src/vector_math.dart';
import 'package:test/test.dart';

void main() {
  test('cues split lines and sentences, coalesce beyond max', () {
    expect(cues('a。b！c', 8), ['a。', 'b！', 'c']);
    expect(
        cues('one. two? three!\nfour', 8), ['one.', 'two?', 'three!', 'four']);
    expect(cues('1.5 kg is fine', 8), ['1.5 kg is fine']);
    expect(cues('a\nb\nc\nd\ne', 2), ['a b', 'c d e']);
    expect(cues('  \n ', 8), isEmpty);
  });

  test('shorten prefers a boundary in the second half', () {
    expect(shorten('abcdefghij', 20), 'abcdefghij');
    expect(shorten('これは文です。これは長い文です。おわり', 11), 'これは文です。');
    expect(shorten('a' * 30, 10), 'a' * 10);
    expect(cleanText(' x 《y》\t z ', 100), 'x y z');
  });

  test('formatLocal works past year 9999 and for negative offsets', () {
    expect(formatLocal(1749641400, 'Asia/Tokyo;+09:00'),
        '2025-06-11 20:30 +09:00');
    expect(formatLocal(0, 'UTC;-03:30'), '1969-12-31 20:30 -03:30');
    expect(formatLocal(253402300800, 'UTC;+00:00'),
        startsWith('10000-01-01 00:00'));
    expect(formatLocal(100000000000, 'X;+00:00'), startsWith('5138-11-16'));
    expect(MemoryTimezone.parse('Asia/Tokyo;+09:00').offset,
        const Duration(hours: 9));
    expect(MemoryTimezone.parse('garbage').offset, Duration.zero);
    expect(parseUtcOffset('-0530'), const Duration(hours: -5, minutes: -30));
  });

  test('ids sort by time and survive a 50-bit millisecond clock', () {
    final rng = Random(1);
    final a = ulid(1700000000000, random: rng);
    final b = ulid(1700000000001, random: rng);
    expect(a.length, 26);
    expect(a.compareTo(b), lessThan(0));
    final far = ulid(1 << 49, random: rng);
    expect(far.compareTo(b), greaterThan(0));
    expect(far.substring(0, 10), isNot(startsWith('0')));
  });

  test('vector helpers', () {
    final v = l2Normalized([3, 4, 0].map((e) => e.toDouble()).toList());
    expect(v[0], closeTo(0.6, 1e-6));
    expect(v[1], closeTo(0.8, 1e-6));
    expect(dot(v, v), closeTo(1, 1e-6));
    expect(unpackF32(packF32(v)), v);
    expect(l2Normalized(Float32List(3)), [0, 0, 0]);
  });

  test('config lenient parse', () {
    final c = EngramConfig.fromJson(
        {'capacity': 42, 'alpha': '0.5', 'minScore': 'bad'});
    expect((c.capacity, c.alpha, c.minScore), (42, 0.5, 0.1));
  });

  test('InMemoryStore json round-trip', () async {
    final s = InMemoryStore();
    await s.put(Memory(
      id: 'a',
      text: 't',
      createdAt: 1,
      tz: 'UTC;+00:00',
      lastRecall: 1,
      stability: 2,
      consolidated: true,
      modelId: 'm',
      vector: Float32List.fromList([1, 0]),
    ));
    final copy = InMemoryStore.fromJson(s.toJson());
    final m = (await copy.loadAll()).single;
    expect(m.text, 't');
    expect(m.vector, [1, 0]);
    expect(m.consolidated, isTrue);
  });

  test('fromJson skips an unreadable row but keeps the rest', () async {
    final s = InMemoryStore();
    await s.put(Memory(
      id: 'a',
      text: 't',
      createdAt: 1,
      tz: 'UTC;+00:00',
      lastRecall: 1,
      stability: 2,
      consolidated: true,
      modelId: 'm',
      vector: Float32List.fromList([1, 0]),
    ));
    final json = s.toJson();
    final rows = (json['memories'] as List).toList()
      ..add({'id': 'bad', 'vector': 42}); // not a memory at all
    final copy = InMemoryStore.fromJson({'memories': rows});
    expect((await copy.loadAll()).map((m) => m.id), ['a']);
    // A payload that is not a list of memories still fails loudly.
    expect(() => InMemoryStore.fromJson({'memories': 'nope'}),
        throwsFormatException);
  });
}
