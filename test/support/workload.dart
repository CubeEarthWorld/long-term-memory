/// The seeded workload behind `test/equivalence/golden.json` (1.0.1's outcome).
library;

import 'package:long_term_memory/long_term_memory.dart';

import 'fakes.dart';

/// Topic-clustered token texts so cues reactivate real candidates.
const _topics = [
  ['kyoto', 'trip', 'hotel', 'train', 'temple', 'autumn'],
  ['doctor', 'clinic', 'allergy', 'pollen', 'medicine', 'spring'],
  ['piano', 'lesson', 'teacher', 'recital', 'practice', 'chopin'],
  ['salary', 'bank', 'rent', 'budget', 'savings', 'loan'],
  ['cat', 'vet', 'food', 'litter', 'toy', 'kitten'],
  ['python', 'dart', 'engine', 'memory', 'vector', 'test'],
];

/// Runs the workload and returns its JSON-able trace.
Future<List<Object?>> runWorkload({
  EngramMemory Function(Embedder embedder, EngramConfig config, int Function())?
      create,
  Future<void> Function(EngramMemory memory)? afterStep,
}) async {
  var seed = 0x5eed;
  int next(int n) {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    return seed % n;
  }

  final clock = VirtualClock();
  const config = EngramConfig(
      capacity: 48, gracePeriod: 86400, dreamBudget: 3, relativeScore: 0.3);
  final embedder = FakeEmbedder();
  final memory = create?.call(embedder, config, clock.call) ??
      EngramMemory(
        store: InMemoryStore(),
        embedder: embedder,
        config: config,
        clock: clock.call,
        defaultTimezone: const MemoryTimezone('Asia/Tokyo', Duration(hours: 9)),
      );
  await memory.initialize();
  String d(double v) => v.toString();
  List<String> texts(List<Memory> ms) => [for (final m in ms) m.text];
  final trace = <Object?>[];
  RecallResult? last;
  for (var step = 0; step < 220; step++) {
    clock.advance(600 + next(12 * 3600));
    final op = next(100);
    final topic = _topics[next(_topics.length)];
    String words(int n) =>
        [for (var i = 0; i < n; i++) topic[next(6)]].join(' ');
    if (op < 55) {
      final text = '${words(3)} ${_topics[next(6)][next(6)]} n${next(40)}';
      final cue = next(3) == 0 ? '' : words(2);
      final r = await memory.remember(text,
          cue: cue, salience: (1 + next(4)).toDouble());
      trace.add({
        'op': 'remember',
        'action': r.action.name,
        'cosine': r.cosine?.toString(),
        'evicted': r.evicted?.text,
      });
    } else if (op < 75) {
      last = await memory.recall('${words(2)}、${words(2)}');
      trace.add({
        'op': 'recall',
        'recalled': [
          for (final c in last.recalled)
            [c.memory.text, d(c.score), d(c.cosine), d(c.retrievability)],
        ],
        'pack': last.packText.replaceAll(RegExp('《id:[^》]+》'), '《id》'),
      });
    } else if (op < 80) {
      final ids = [for (final c in last?.recalled ?? <Recalled>[]) c.memory.id];
      trace.add({
        'op': 'cite',
        'cited': (await memory.cite(ids.map((i) => '《id:$i》').join())).length,
      });
    } else if (op < 90) {
      trace.add({
        'op': 'clusters',
        'clusters': [
          for (final c in await memory.clusters(budget: 1 + next(4))) texts(c),
        ],
      });
    } else {
      final reports = await memory.dream(
          adjudicate: next(2) == 0 ? mergeToGist : alwaysKeep,
          budget: 1 + next(3));
      trace.add({
        'op': 'dream',
        'reports': [
          for (final rep in reports)
            {
              'action': rep.action.name,
              'before': texts(rep.before),
              'after': [
                for (final g in rep.after)
                  [g.text, d(g.stability), g.lastRecall],
              ],
            },
        ],
      });
    }
    await afterStep?.call(memory);
  }
  trace.add({
    'op': 'final',
    'memories': [
      for (final m in await memory.memories())
        [m.text, d(m.stability), m.lastRecall, m.consolidated],
    ],
  });
  return trace;
}
