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
    expect(reports.first.action, DreamAction.keep);
    expect(reports.every((r) => r.action == DreamAction.keep), isTrue);
    for (final m in await memory.memories()) {
      expect(m.consolidated, isTrue, reason: 'every seed settled');
      expect(m.stability, before[m.id]!.stability);
      expect(m.lastRecall, before[m.id]!.lastRecall);
    }
    expect(await memory.dream(adjudicate: alwaysKeep), isEmpty);
  });

  test('a throwing adjudicator leaves the cluster labile for retry', () async {
    final memory = await seeded();
    final reports =
        await memory.dream(adjudicate: (_) => throw StateError('llm down'));
    expect(reports.first.action, DreamAction.error);
    expect((await memory.clusters()), isNotEmpty);
    expect((await memory.dream(adjudicate: mergeToGist)).first.action,
        DreamAction.replace);
  });

  test('unrelated or excessive replacements are treated as keep', () async {
    final memory = await seeded();
    final r1 = await memory.dream(
        adjudicate: (_) =>
            const DreamDecision(['zzz qqq unrelated hallucination']));
    expect(r1.first.action, DreamAction.keep);
    final memory2 = await seeded();
    // Names every candidate, so `absorbed` is the whole cluster and the gist
    // count is the only thing left to reject the verdict.
    final r2 = await memory2.dream(
        adjudicate: (req) => DreamDecision([
              for (var i = 0; i < req.members.length + 1; i++) 'trip kyoto $i',
            ], absorbedIds: [for (final m in req.members.skip(1)) m.id]));
    expect(r2.first.action, DreamAction.keep);
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

  test('a correction is seeded by the new evidence, with the old fact in view',
      () async {
    final (memory, clock, _) = await build();
    final old = (await memory.remember('user lives in tokyo city', salience: 5))
        .memory!;
    clock.advanceDays(30);
    final fresh =
        (await memory.remember('user lives in osaka city')).memory!;
    DreamRequest? seen;
    await memory.dream(adjudicate: (req) {
      seen = req;
      return const DreamDecision.keep();
    });
    expect(seen!.members.first.id, fresh.id,
        reason: 'the newest evidence seeds');
    expect(seen!.members.map((m) => m.id), contains(old.id));
    expect(
        seen!.members.first.localTime, formatLocal(fresh.createdAt, fresh.tz));
    expect((await memory.memory(old.id))!.consolidated, isTrue,
        reason: 'a candidate that was only offered is untouched');
  });

  test('only the named candidates are absorbed', () async {
    final (memory, clock, _) = await build();
    await memory.remember('user lives in tokyo city');
    final bystander =
        (await memory.remember('user lives in tokyo city with a cat')).memory!;
    clock.advanceDays(30);
    await memory.remember('user lives in osaka city');
    final reports = await memory.dream(
        adjudicate: (req) => req.members.first.text != 'user lives in osaka city'
            ? const DreamDecision.keep()
            : DreamDecision(['user moved to osaka city'], absorbedIds: [
                for (final m in req.members.skip(1))
                  if (m.text == 'user lives in tokyo city') m.id,
              ]));
    expect(reports.map((r) => r.action), contains(DreamAction.replace));
    expect(await memory.memory(bystander.id), isNotNull,
        reason: 'not named ⇒ not rewritten');
    final texts = [for (final m in await memory.memories()) m.text];
    expect(texts, contains('user moved to osaka city'));
    expect(texts, isNot(contains('user lives in tokyo city')));
  });

  test('a cue reaches the version the update supersedes', () async {
    final (memory, clock, _) = await build();
    await memory.remember('user lives in tokyo city');
    clock.advanceDays(30);
    // Text far from the old fact (cos ≈ -0.11 < thetaRelated); only the cue
    // reaches back to it (cos ≈ 0.90).
    await memory.remember('resident of osaka prefecture now',
        cue: 'user lives in city');
    final clusters = await memory.clusters();
    expect(clusters, hasLength(1));
    expect(clusters.single.map((m) => m.text).toSet(), {
      'user lives in tokyo city',
      'resident of osaka prefecture now',
    });
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

  test('DreamDecision parsing is lenient but never reads silence as keep', () {
    expect(DreamDecision.parseJson('```json\n{"action":"keep"}\n```').isKeep,
        isTrue);
    expect(
        DreamDecision.parseJson(
                'x {"action":"replace","memories":["a",{"text":"b"},""]} y')
            .memories,
        ['a', 'b']);
    // SPEC §5: an unparsable verdict must retry, never settle the trace.
    expect(() => DreamDecision.parseJson('garbage'), throwsFormatException);
    expect(() => DreamDecision.parseJson(null), throwsFormatException);
    expect(DreamDecision.parse('garbage'), isNull);
    expect(DreamDecision.parse(null), isNull);
    // A missing or mistyped `memories` is a broken answer, not an empty one.
    expect(DreamDecision.parse('{"action":"replace","ids":["x"]}'), isNull);
    expect(DreamDecision.parse('{"action":"replace","memories":"a string"}'),
        isNull);
    expect(DreamDecision.parse('{"action":"replace","memories":[]}')!.isKeep,
        isTrue);
    expect(EngramPrompts.parseExtractedTexts('{"memories":["a"," ","b"]}'),
        ['a', 'b']);
  });
}
