# long_term_memory

A portable, **zero-dependency** long-term memory engine for LLM applications, written in pure Dart and usable from any Flutter app (all platforms) or Dart server/CLI. It implements **ENGRAM v2.1** — a memory *trace* model derived from the principles of human memory — and is the Dart twin of [CubeEarthWorld/llm-long-term-memory](https://github.com/CubeEarthWorld/llm-long-term-memory) (the specification lives there as `SPEC.md`; a cross-language conformance test keeps both implementations identical).

> Generation only at the moment of verbalization. All judgement is distance.
> All forgetting is arithmetic. All consolidation happens inside the dream.

**The package contains only the algorithm.** The LLM, the embedding model and the database are injected through three small interfaces, so it composes with anything: [firebase_ai](https://pub.dev/packages/firebase_ai), [llamadart](https://pub.dev/documentation/llamadart/latest/), [sqlite3](https://pub.dev/documentation/sqlite3/latest/), drift, Isar, Hive, REST endpoints, on-device ONNX models, …

日本語版は [README.ja.md](README.ja.md) を参照してください。

---

## The model in one screen

A memory is a **trace** with two numbers — when it was last recalled and how stable it is (a half-life) — plus one flag (`consolidated`). No tiers, no counters, no rings, no maintenance call.

```text
R(now)   = 2^(−max(0, now − lastRecall) / stability)           retrievability ∈ [0,1]
strength = stability · R                                       total remaining retrievability
a        = max(0, (cos − cosineFloor) / (1 − cosineFloor))     cue activation
recall   : stability ← min(stability · (1 + gain·a·(1−R)), S_max);  lastRecall ← now
new      : stability = clamp(S0 · salience, 1 s, S_max)
```

| verb | what happens |
|---|---|
| `remember(text, salience:, cue:)` | exact duplicate → rehearsal; otherwise insert, **never overwrite**. A trace whose `cue` reaches a neighbour (cos ≥ θ_related) is born *labile*; no existing trace is touched. Over `capacity`, the lowest-strength trace outside the 3-day grace period is forgotten. |
| `recall(query)` | multi-cue cosine → `score = a·(α + (1−α)·R)` → absolute + relative cut → MMR → `[unix tz] text 《id》` pack ≤ 1024 chars. Injection is exposure: half-activation strengthening. |
| `cite(reply)` | the `《id》`s the LLM quoted are strengthened as *used* (full activation). |
| `forget(id)` | id-only physical delete. |
| `dream(adjudicate:)` | labile traces (first in, first out; ≤ 8·budget) each seed a cluster of the older traces their `cue` reactivates (cos ≥ θ_related, ≤ 8); your LLM answers **keep** or **replace** (the ids it supersedes + the gist texts). Gists inherit the strongest member's stability plus the *live* evidence of the others; unrelated outputs are rejected as confabulation. A settled store makes no LLM calls. |

Everything is bounded, so cost does not depend on elapsed time: a 3000-virtual-year simulation (hundreds of thousands of writes, decade-long silences, a clock fault) is part of the test suite.

## Installation

```yaml
dependencies:
  long_term_memory:
    git:
      url: https://github.com/CubeEarthWorld/long-term-memory
```

## Quick start

```dart
import 'package:long_term_memory/long_term_memory.dart';

Future<void> main() async {
  final memory = EngramMemory(
    store: InMemoryStore(),               // swap for your DB adapter
    embedder: myEmbedder,                 // your Embedder implementation
    defaultTimezone: const MemoryTimezone('Asia/Tokyo', Duration(hours: 9)),
  );
  await memory.initialize();

  // REMEMBER — store one self-contained fact (e.g. from an LLM tool call).
  // `cue` is the question this fact would later be asked with: it is what the
  // dream phase searches the past with, so an update reaches the version it
  // supersedes even when the two sentences are textually far apart.
  final saved = await memory.remember('ユーザーは京都に住んでいる',
      salience: 2, cue: 'ユーザーは今どこに住んでいるか？');
  print(saved.action); // RememberAction.inserted

  // RECALL — fetch relevant memories for the next prompt.
  final recall = await memory.recall('どこに住んでいるか覚えてる？');
  print(recall.packText); // [<unix> Asia/Tokyo;+09:00] ユーザーは京都に住んでいる　《id:...》

  // CITE — after the LLM replied, strengthen the memories it actually used.
  await memory.cite(llmReply);

  // DREAM — offline consolidation through YOUR LLM (run when the app is idle).
  await memory.dream(adjudicate: (request) async {
    final raw = await myLlm.generateJson(request.buildPrompt());
    return DreamDecision.parseJson(raw);   // {"action":"keep"} or {"action":"replace","ids":[...],"memories":[...]}
  });
}
```

A runnable zero-dependency version (toy embedder + in-memory store) is in [`example/main.dart`](example/main.dart): `dart run example/main.dart`.

## How it fits into a chat app

```
user utterance
   ├─► memory.recall(utterance)            → memory pack
   ├─► your LLM call
   │     system  = EngramPrompts.conversationSystemPromptJa (or En)
   │     user    = EngramPrompts.buildUserMessage(currentTime: memory.nowLocal(),
   │                 memoryPack: pack.packText, userText: utterance)
   │     tools   = [EngramPrompts.saveMemoryToolSpec, EngramPrompts.deleteMemoryToolSpec]
   ├─► on tool call "save_memory"   → memory.remember(args['text'], salience: args['salience'] ?? 1,
   │                                                   cue: args['cue'] ?? '')
   ├─► on tool call "delete_memory" → memory.forget(args['id'])
   └─► memory.cite(reply)                  → memories the reply quoted as 《id:…》 are strengthened
```

Optionally, when a turn saved nothing, run the extraction fallback: send `EngramPrompts.buildExtractionInstruction(...)` to your LLM, parse with `EngramPrompts.parseExtractedTexts(raw)`, and feed each string through `remember`.

## Time and timezone inputs

Every entry point accepts the current Unix time (seconds) and a timezone:

```dart
await memory.remember('ユーザーは2026-04-01にベルリンへ出張した',
    nowUnix: 1774000000,
    timezone: const MemoryTimezone('Europe/Berlin', Duration(hours: 2)));
await memory.recall('出張の予定', nowUnix: 1774100000);
```

Omitted values fall back to the injected `clock` / `defaultTimezone`, then to the system clock / UTC. Timezones are stored per memory as `'IANA_name;+HH:MM'`, so local-time formatting is pure arithmetic (no tz database) and works for any year. A `clock` callback makes the engine fully deterministic for tests; future timestamps are clamped at load so a clock rollback can never mint immortal memories.

## The three interfaces

### 1. `Embedder` — your embedding model

```dart
abstract class Embedder {
  String get modelId;       // stamped on vectors; switching models re-embeds every trace from text
  int get dimension;        // one modelId means one dimension
  Future<List<Float32List>> embedQueries(List<String> texts);
  Future<List<Float32List>> embedDocuments(List<String> texts);
}
```

Vectors need not be normalised. `CallbackEmbedder` wires an SDK without declaring a class. If the embedder throws, `remember` still keeps the text (indexed later), `recall` returns nothing, and `dream` re-indexes what it can — the engine never loses data because a model is unavailable. A stored vector of any length other than `dimension` — a model file swapped behind an unchanged `modelId`, or a provider that mis-sizes one response — is treated as stale and re-embedded rather than compared against vectors it cannot be compared with. Three parameters depend on the model's cosine distribution and should be calibrated once: `cosineFloor` (baseline cosine of unrelated text, ≈0.4 for EmbeddingGemma), `thetaRelated`, `gistMinCosine`.

### 2. `MemoryStore` — your database

Six operations, no query language: `loadAll`, `put`, `remove`, `clear`, `transaction`, `backup` (+ `open`/`close`). The engine holds every trace in RAM (≈35 MB at 10k traces × 768 dims); the store only persists. `InMemoryStore` is bundled (with `toJson`/`fromJson`), and a complete SQLite adapter with WAL, transactions and a rotating snapshot ring ships in [`example/sqlite_adapter`](example/sqlite_adapter).

### 3. `DreamAdjudicator` — your LLM (dream phase only)

```dart
typedef DreamAdjudicator = FutureOr<DreamDecision> Function(DreamRequest request);
```

`request.members` carry id / text / local time / timezone / R; `request.buildPrompt()` gives the consolidation prompt (ja/en); `DreamDecision.parseJson` parses leniently but throws a `FormatException` on an empty or unparsable answer — including a missing or mistyped `memories` field — so a truncated reasoning model is never read as "keep" (use `DreamDecision.parse`, which returns `null` instead, if you want to decide yourself). The engine enforces every guard: text hygiene, ≤ 8 members per verdict, no more replacements than members, a cosine check against the members (confabulation guard), strength-conserving inheritance, one transaction per cluster, and retry of clusters whose callback threw.

## API reference

| Method | Purpose |
|---|---|
| `initialize()` | Open the store, load and clamp every trace, re-embed stale vectors. |
| `remember(text, {salience, cue, nowUnix, timezone})` | Store a proposition → `RememberResult` (`inserted` / `reinforced` / `rejected` / `rateLimited`, with `evicted`). |
| `recall(query, {nowUnix})` | → `RecallResult(packText, recalled)`; injected traces get the half-activation update. |
| `cite(replyText)` | Complete the strengthening of the traces the reply quoted as `《id:…》`. |
| `forget(id)` | Physical delete by id → `bool`. |
| `dream({adjudicate, budget, nowUnix, timezone})` | Offline consolidation → `List<DreamReport>` (`keep` / `replace` / `error`). |
| `clusters({budget})` | The clusters the next dream would hand to the LLM (no LLM call); exactly the seeds `dream(budget:)` scans (default `dreamBudget`). |
| `memories()` / `memory(id)` / `retrievability(m, now)` / `strength(m, now)` | Introspection. |
| `nowUnix()` / `nowLocal()` / `reset()` | Clock helpers, erase everything. |

All methods are serialized internally; an `EngramMemory` instance is safe to call from interleaving async code. One process owns a store.

## Configuration (`EngramConfig`, 19 parameters)

| Parameter | Default | Meaning |
|---|---|---|
| `capacity` | 10000 | max traces; the weakest old trace is forgotten beyond it |
| `initialStability` | 1 d | S0 of a new trace |
| `spacingGain` | 3.0 | stability growth on recall |
| `maxStability` | 10 y | no immortal memory |
| `gracePeriod` | 3 d | new traces are protected from eviction (unless they exceed a tenth of capacity) |
| `cosineFloor` | 0.0 | baseline cosine of unrelated text under your model — the default is model-agnostic (the Python reference ships 0.4, pre-calibrated for EmbeddingGemma) |
| `alpha` | 0.35 | retrievability floor in the score |
| `injectN` / `mmrLambda` | 5 / 0.3 | traces injected per recall, MMR diversity |
| `minScore` / `relativeScore` | 0.1 / 0.6 | absolute and relative score cuts |
| `budgetChars` / `maxCues` | 1024 / 8 | pack budget, query cues |
| `thetaRelated` | 0.55 | neighbourhood (labile) threshold — how far a cue reaches into the past |
| `dreamBudget` / `dreamMaxMembers` | 5 / 8 | LLM calls per dream, members per cluster |
| `gistMinCosine` | 0.5 | replacement texts must relate to the cluster |
| `textMax` / `writesPerDay` | 170 / 1000 | text limit (code points, as in Python), soft write rate |

## Testing your integration

The test suite shows the patterns: a deterministic token-overlap `FakeEmbedder` (bit-identical to the Python reference's) and a `VirtualClock` in [`test/support/fakes.dart`](test/support/fakes.dart) make every scenario reproducible; `test/conformance/` holds the scripted scenario and its trace shared with the Python implementation.

```bash
dart test                                # package suite (incl. the 3000-year simulation)
cd example/sqlite_adapter && dart test   # adapter suite
```

## License

MIT — see [LICENSE](LICENSE).
