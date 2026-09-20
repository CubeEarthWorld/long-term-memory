@Tags(['slow'])
library;

import 'dart:math';

import 'package:long_term_memory/long_term_memory.dart';
import 'package:test/test.dart';

import 'support/fakes.dart';

/// Millennia of virtual time: writes every few days, weekly recalls of a
/// handful of key facts, monthly dreams, occasional decade-long silences
/// and a clock fault. Asserts every invariant of SPEC §7 and that the
/// rehearsed facts remain retrievable while noise is forgotten.
void main() {
  test('3000 virtual years keep invariants and recall quality', () async {
    // thetaRelated is calibrated per embedding model; the token-overlap fake
    // scores unrelated text far higher than a real model, so the reactivation
    // band is raised here to keep the same selectivity.
    const config =
        EngramConfig(capacity: 300, writesPerDay: 50, thetaRelated: 0.75);
    final (memory, clock, store) = await build(
      config: config,
      embedder: FakeEmbedder(dimension: 32),
    );
    final rng = Random(7);
    const facts = [
      'user likes matcha ice cream',
      'user was born in kyoto japan',
      'user has a cat named tama',
    ];
    for (final f in facts) {
      await memory.remember(f, salience: 2);
    }
    const day = 86400;
    var writes = 0, recalls = 0, dreams = 0;
    final start = clock.t;
    for (var d = 0; d < 3000 * 365; d += 3) {
      clock.advance(3 * day);
      if (rng.nextInt(1000) == 0) clock.advance(10 * 365 * day); // silence
      await memory.remember('note ${rng.nextInt(1 << 30)} topic${d % 97}');
      writes++;
      if (d % 7 < 3) {
        final f = facts[rng.nextInt(facts.length)];
        final r = await memory.recall(f);
        expect(r.recalled.map((c) => c.memory.text), contains(f),
            reason: 'key fact must stay retrievable on day $d');
        recalls++;
      }
      if (d % 30 < 3) {
        await memory.dream(adjudicate: mergeToGist, budget: 2);
        dreams++;
      }
      if (d == 1500 * 365) {
        // Clock fault: jump 200 years ahead, then back.
        final t = clock.t;
        clock.t += 200 * 365 * day;
        await memory.recall(facts[0]);
        clock.t = t;
      }
    }
    final rows = await store.loadAll();
    expect(rows.length, lessThanOrEqualTo(config.capacity));
    final now = clock.t;
    for (final m in rows) {
      expect(m.stability, inInclusiveRange(1, config.maxStability));
      expect(m.text.length, inInclusiveRange(1, config.textMax));
      expect(m.vector.length, 32);
      final r = memory.retrievability(m, now);
      expect(r, inInclusiveRange(0, 1));   // a NaN already fails the range
    }
    for (final f in facts) {
      final r = await memory.recall(f);
      expect(r.recalled.first.memory.text, f);
      expect(r.recalled.first.memory.stability, greaterThan(365 * day));
    }
    expect(int.parse(formatLocal(now, 'UTC;+00:00').substring(0, 4)),
        greaterThan(5000));
    expect((now - start) ~/ (365 * day), greaterThanOrEqualTo(3000));
    expect(writes, greaterThan(300000));
    expect(recalls, greaterThan(100000));
    expect(dreams, greaterThan(30000));
  }, timeout: const Timeout(Duration(minutes: 10)));
}
