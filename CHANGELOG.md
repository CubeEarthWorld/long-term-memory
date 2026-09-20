# Changelog

## 1.0.0 — cues, and no silent failures

First stable release. Every remaining way for the engine to fail quietly is now
an error you can see: an unreadable dream verdict, a dropped `cue`, a mistyped
salience, a config value that only asserts in debug. A trace also carries the
question it answers, so an update finds the version it supersedes. The
conformance trace shared with the Python reference is regenerated; one small
breaking change to `Memory.copyWith`.

- **`cue` — the question a fact will later be asked with.** `remember` takes
  `cue:`, `Memory` carries it, and the dream searches the past with the cue's
  query vector instead of the text's own, so an update reaches the version it
  supersedes even when the two sentences are textually far apart. `cue` is now
  in the bundled `saveMemoryToolSpec` (required alongside `text`), in both
  conversation system prompts, in `Memory.toJson`/`fromJson`, and in the
  example SQLite adapter's schema — which used to drop it on every reload.
  Existing adapter databases are migrated with `ALTER TABLE` on open.
- **A dream verdict is never guessed.** `DreamDecision.parseJson` still parses
  leniently (code fences, surrounding prose, `memories` as strings or objects)
  but now throws a `FormatException` on an empty or unparsable answer, and
  treats a missing or mistyped `memories` field as a broken answer rather than
  an empty one. Silence must not be read as "keep" (SPEC §5): the cluster stays
  labile and the next dream retries it, instead of a truncated reasoning model
  settling the trace forever. `DreamDecision.parse` returns `null` for callers
  that would rather decide themselves.
- **The verdict names what it supersedes.** The consolidation prompt asks for
  `ids` alongside `memories`; candidates the verdict does not name are merely
  offered and left untouched, a `keep` settles only the seed, and a gist
  inherits its newest member's `createdAt` / `tz` / `cue` so a later update can
  still reach it.
- **Text limits count code points, not UTF-16 code units**, so a limit means
  the same thing as Python's `len()` and a cut can never split a surrogate
  pair. The injected pack's character budget is counted the same way, and the
  whitespace class used for cleaning and cue splitting now matches Python's
  `\s` exactly.
- **Ties are deterministic.** `List.sort` is not stable above ~32 elements, so
  the recall pool, the dream seeds and the candidate list all break score,
  stability and cosine ties on insertion order — the one order both languages
  agree on (ULID tails are random).
- **NaN no longer wins.** `clamp()` orders NaN above every double, so a NaN
  `salience` used to read as maximum salience and a NaN stability as an
  immortal trace; both now fall back to the default. The confabulation guard
  compares against a true maximum, so `gistMinCosine <= 0` still rejects.
- **`EngramConfig.fromJson` throws** `ArgumentError` on an out-of-range value
  instead of relying on the const constructor's asserts, which are stripped in
  release builds (`cosineFloor: 1` would have turned every score into NaN).
- **`dot` throws on a dimension mismatch** instead of silently comparing the
  shared prefix: one `modelId` must mean one dimension, as in Python.
- **`thetaRelated` default is 0.55** (was 0.75), measured on EmbeddingGemma.
- **Breaking:** `Memory.copyWith` no longer accepts `text` or `tz`, and the new
  `cue` is not copyable either — a trace's content is never rewritten in place
  (SPEC §7); the dream inserts a new trace instead.
- Smaller fixes: `recall` no longer pays for an embedding when nothing is
  indexed; `reset()` clears the citation window; the example adapter loads with
  `ORDER BY id` (a rehearsed row's `rowid` is its *last write* order, not its
  insertion order); `EngramPrompts.conversationSystemPrompt(locale)` picks the
  `Ja`/`En` constant for you.

## 0.3.0 — smaller surface, same engine

Structural clean-up; the conformance trace shared with the Python reference is
unchanged. Not backward compatible for callers of the removed API.

- **Public API narrowed** to what an integration needs: `EngramConfig.copyWith`
  / `toJson`, `MemoryTimezone.fromDateTime` / `local`, `UlidGenerator`,
  `Embedder.dimension` (never read by the engine), `l2Normalized(dim:)` and the
  internal helpers (`cleanText`, `cues`, `shorten`, `dot`, `formatUtcOffset`,
  `parseUtcOffset`) are gone. `relaxedJsonDecode` moved to `src/json.dart` and
  stays exported.
- Ids are generated from the operation's `nowUnix` (as in Python); backdated
  writes now get correctly ordered ids.
- `dream(budget: <0)` no longer throws; a `recall` that injects nothing still
  ends the previous citation window; the dream keep-path and re-indexing commit
  in one transaction (SPEC §5).
- `EngramPrompts.buildUserMessage` is one function (locale parameter);
  `src/engine/engine.dart` flattened to `src/engine.dart`; stale v1 wording in
  doc comments removed.

## 0.2.0 — ENGRAM v2 (trace model)

A redesign from first principles of human memory; not backward compatible.

- **One table, one formula.** A memory is a trace with `lastRecall` and
  `stability`; `R = 2^(−Δt/S)`, retrieval grows stability by
  `1 + gain·a·(1−R)` (spacing effect × cue activation). Tiers, mass, refractory
  timers, generation counters, conflict rings, dream logs, tombstones and the
  per-turn `maintain()` are gone.
- **Nothing is overwritten at write time.** Paraphrases, updates and
  contradictions become labile pairs (reconsolidation: the old trace is
  reactivated too) that the dream adjudicates with both timestamps in view.
- **Forgetting = eviction of the lowest strength (`S·R`)** past a 3-day grace
  period (hippocampal buffer); floods evict their own members.
- **Injection is exposure, not use**: half-activation strengthening on recall,
  completed by the new `cite(reply)` for the memories the LLM quoted.
- **Dream** seeds clusters from labile traces (most stable first), ≤ 8 members
  per verdict, `keep` / `replace`, gists inherit the strongest member plus live
  evidence with strength conservation, confabulation guard by cosine. A settled
  store makes no LLM calls.
- **Salience** (emotional weight, 0–10) scales initial stability; `cosineFloor`
  calibrates cue activation to the embedding model.
- `MemoryStore` shrinks to six operations; the store is loaded into RAM at
  start and stale-model vectors are re-embedded from text (failure-tolerant).
- Ids carry a 50-bit millisecond timestamp (valid past year 37,000); local-time
  formatting handles negative offsets and any year.
- `package:crypto` dependency removed — the package has no dependencies.
- Test suite rewritten, including a 3000-virtual-year simulation and a
  cross-language conformance scenario shared with the Python reference.

## 0.1.0

Initial release: pure-Dart port of the ENGRAM v1.1 engine.
