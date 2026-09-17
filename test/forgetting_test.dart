import 'package:long_term_memory/long_term_memory.dart';
import 'package:test/test.dart';

import 'support/fakes.dart';

void main() {
  test('retrievability halves every stability', () async {
    final (memory, _, _) = await build();
    final m = (await memory.remember('half life')).memory!;
    final t0 = m.lastRecall;
    expect(memory.retrievability(m, t0), 1.0);
    expect(memory.retrievability(m, t0 + 86400), closeTo(0.5, 1e-9));
    expect(memory.retrievability(m, t0 + 2 * 86400), closeTo(0.25, 1e-9));
    expect(memory.retrievability(m, t0 - 1000), 1.0, reason: 'clock rollback');
    expect(memory.retrievability(m, t0 + 86400 * 100000), 0.0);
  });

  test('spaced recall strengthens more than massed recall', () async {
    final (memory, clock, _) = await build();
    final massed = (await memory.remember('massed alpha')).memory!;
    final spaced = (await memory.remember('spaced beta')).memory!;
    for (var i = 0; i < 3; i++) {
      clock.advance(60);
      await memory.recall('massed alpha');
    }
    clock.advanceDays(3);
    await memory.recall('spaced beta');
    final sm = (await memory.memory(massed.id))!.stability;
    final ss = (await memory.memory(spaced.id))!.stability;
    expect(sm, lessThan(86400 * 1.05), reason: 'R≈1 ⇒ ~no gain');
    expect(ss, greaterThan(86400 * 2), reason: 'R≈0.125 ⇒ large gain');
  });

  test('stability is capped: no immortal memory', () async {
    final (memory, clock, _) = await build();
    final m = (await memory.remember('rehearsed daily')).memory!;
    for (var i = 0; i < 400; i++) {
      clock.advanceDays(30);
      await memory.recall('rehearsed daily');
    }
    final s = (await memory.memory(m.id))!.stability;
    expect(s, const EngramConfig().maxStability);
    final silent = clock.t + (100 * 365 * 86400);
    expect(memory.retrievability((await memory.memory(m.id))!, silent),
        lessThan(0.001));
  });

  test('strength orders old stable facts above fresh junk', () async {
    final (memory, clock, _) = await build();
    final fact = (await memory.remember('stable fact', salience: 10)).memory!;
    clock.advanceDays(5);
    final junk = (await memory.remember('junk')).memory!;
    final now = clock.t;
    expect(memory.strength((await memory.memory(fact.id))!, now),
        greaterThan(memory.strength(junk, now)));
  });

  test('a dormant but relevant trace still competes (alpha floor)', () async {
    final (memory, clock, _) = await build();
    await memory.remember('rust database project');
    clock.advanceDays(3650);
    final r = await memory.recall('rust database');
    expect(r.recalled.single.retrievability, lessThan(1e-6));
    expect(r.recalled.single.score, greaterThan(0.25));
  });
}
