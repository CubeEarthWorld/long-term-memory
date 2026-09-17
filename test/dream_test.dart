import 'package:long_term_memory/long_term_memory.dart';
import 'package:test/test.dart';

import 'support/fakes.dart';

void main() {
  Future<EngramMemory> seeded(
      {EngramConfig config = const EngramConfig()}) async {
    final (memory, _, _) = await build(config: config);
    await memory.remember('trip kyoto plan schedule note');
    await memory.remember('trip kyoto hotel schedule note');
    await memory.remember('trip kyoto food schedule note');
    await memory.remember('completely different topic zebra');
    return memory;
  }

  test('replace merges a cluster into a gist that inherits live evidence',
      () async {
    final memory = await seeded();
    final before = await memory.memories();
    final reports = await memory.dream(adjudicate: mergeToGist);
    expect(reports.single.action, DreamAction.replace);
    expect(reports.single.before.length, 3);
    final gist = reports.single.after.single;
    expect(gist.consolidated, isTrue);
    expect(gist.stability, closeTo(3 * 86400, 1));
    expect(gist.lastRecall, before.first.lastRecall);
    final all = await memory.memories();
    expect(all.length, 2);
    expect(all.every((m) => m.consolidated), isTrue);
    expect(await memory.dream(adjudicate: mergeToGist), isEmpty,
        reason: 'settled store ⇒ no LLM calls');
  });

  test('keep marks the cluster consolidated without touching strength',
      () async {
    final memory = await seeded();
    final before = {for (final m in await memory.memories()) m.id: m};
    final reports = await memory.dream(adjudicate: alwaysKeep);
    expect(reports.single.action, DreamAction.keep);
    for (final m in await memory.memories()) {
      expect(m.consolidated, isTrue);
      expect(m.stability, before[m.id]!.stability);
      expect(m.lastRecall, before[m.id]!.lastRecall);
    }
  });

  test('a throwing adjudicator leaves the cluster labile for retry', () async {
    final memory = await seeded();
    final reports =
        await memory.dream(adjudicate: (_) => throw StateError('llm down'));
    expect(reports.single.action, DreamAction.error);
    expect((await memory.clusters()).length, 1);
    expect((await memory.dream(adjudicate: mergeToGist)).single.action,
        DreamAction.replace);
  });

  test('unrelated or excessive replacements are treated as keep', () async {
    final memory = await seeded();
    final r1 = await memory.dream(
        adjudicate: (_) =>
            const DreamDecision(['zzz qqq unrelated hallucination']));
    expect(r1.single.action, DreamAction.keep);
    final memory2 = await seeded();
    final r2 = await memory2.dream(
        adjudicate: (req) => DreamDecision([
              for (var i = 0; i < req.members.length + 1; i++) 'trip kyoto $i',
            ]));
    expect(r2.single.action, DreamAction.keep);
  });

  test('budget bounds LLM calls; strongest seeds first', () async {
    final (memory, _, _) = await build();
    await memory.remember('weak pair one alpha x');
    await memory.remember('weak pair one alpha y');
    await memory.remember('strong pair two beta x', salience: 5);
    await memory.remember('strong pair two beta y', salience: 5);
    final reports = await memory.dream(adjudicate: mergeToGist, budget: 1);
    expect(reports.length, 1);
    expect(reports.single.before.first.text, startsWith('strong'));
    expect((await memory.clusters()).length, 1);
  });

  test('a correction is adjudicated with the old fact in view', () async {
    final (memory, clock, _) = await build();
    final old = (await memory.remember('user lives in tokyo city', salience: 5))
        .memory!;
    clock.advanceDays(30);
    await memory.remember('user lives in osaka city');
    DreamRequest? seen;
    await memory.dream(adjudicate: (req) {
      seen = req;
      return const DreamDecision.keep();
    });
    expect(seen!.members.first.id, old.id, reason: 'old strong trace seeds');
    expect(
        seen!.members.map((m) => m.text), contains('user lives in osaka city'));
    expect(seen!.members.first.localTime, formatLocal(old.createdAt, old.tz));
  });

  test('model switch re-embeds from text at start-up', () async {
    final store = InMemoryStore();
    final (a, _, _) =
        await build(store: store, embedder: FakeEmbedder(modelId: 'm1'));
    await a.remember('persisted across models');
    final (b, _, _) =
        await build(store: store, embedder: FakeEmbedder(modelId: 'm2'));
    final m = (await b.memories()).single;
    expect(m.modelId, 'm2');
    expect((await b.recall('persisted across models')).recalled, hasLength(1));
  });

  test('dream tolerates an offline embedder', () async {
    final store = InMemoryStore();
    final (a, _, _) = await build(store: store);
    await a.remember('x1 y1');
    final (b, _, _) = await build(store: store, embedder: BrokenEmbedder());
    expect(await b.dream(adjudicate: mergeToGist), isEmpty);
    expect((await b.memories()).single.modelId, 'fake/token-overlap');
  });

  test('DreamDecision.parseJson is lenient', () {
    expect(DreamDecision.parseJson('```json\n{"action":"keep"}\n```').isKeep,
        isTrue);
    expect(
        DreamDecision.parseJson(
                'x {"action":"replace","memories":["a",{"text":"b"},""]} y')
            .memories,
        ['a', 'b']);
    expect(DreamDecision.parseJson('garbage').isKeep, isTrue);
    expect(DreamDecision.parseJson(null).isKeep, isTrue);
    expect(EngramPrompts.parseExtractedTexts('{"memories":["a"," ","b"]}'),
        ['a', 'b']);
  });
}
