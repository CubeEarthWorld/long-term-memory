# long_term_memory

LLM アプリ向けの、移植性が高く**依存ゼロ**の長期記憶エンジン（純 Dart。Flutter 全プラットフォーム／Dart サーバー・CLI で利用可）。人間の記憶の原理から導いた**痕跡（trace）モデル ENGRAM v2** を実装しており、[CubeEarthWorld/llm-long-term-memory](https://github.com/CubeEarthWorld/llm-long-term-memory) の Dart 版です（仕様書 `SPEC.md` はそちらに置いてあります。言語間一致テストで両実装の同一性を保証しています）。

> 生成は言語化の瞬間だけ。判断はすべて距離。忘却はすべて算術。統合はすべて夢の中。

**パッケージにはアルゴリズムだけが入っています。** LLM・埋め込みモデル・DB は 3 つの小さなインタフェースで注入します。

## モデルの全体像

記憶は**痕跡**です。持つのは 2 つの数 — 最後に想起した時刻と安定度（半減期）— と 1 つのフラグ（`consolidated`）だけ。層・カウンタ・リング・保守呼び出しはありません。

```text
R(now)   = 2^(−max(0, now − lastRecall) / stability)           想起可能性 ∈ [0,1]
strength = stability · R                                       将来の想起可能性の総量
a        = max(0, (cos − cosineFloor) / (1 − cosineFloor))     手がかりの活性化
想起      : stability ← min(stability · (1 + gain·a·(1−R)), S_max);  lastRecall ← now
新規      : stability = clamp(S0 · salience, 1 s, S_max)
```

| 動詞 | 動作 |
|---|---|
| `remember(text, salience:)` | 同一テキストはリハーサル。それ以外は挿入し、**決して上書きしない**。近傍（cos ≥ θ_related）がある痕跡は*不安定*に生まれ、その近傍も不安定化する（再固定化）。容量超過時は猶予期間（3 日）外で strength 最小の痕跡を忘却。 |
| `recall(query)` | 複数手がかりのコサイン → `score = a·(α + (1−α)·R)` → 絶対・相対閾値 → MMR → `[unix tz] text 《id》` を ≤1024 字で注入。注入は「露出」なので半分だけ強化。 |
| `cite(reply)` | LLM が引用した《id》の記憶を「使用」として完全に強化。 |
| `forget(id)` | id 指定の物理削除。 |
| `dream(adjudicate:)` | 不安定な痕跡（安定度順）を種に cos ≥ θ_related の近傍クラスタ（≤8）を作り、あなたの LLM が **keep** か **replace [texts]** を判定。要旨は最強成員の安定度＋他の「生きた証拠」を継承。無関係な出力は作話として拒否。整理済みのストアでは LLM を呼ばない。 |

全状態が有界なので演算コストは経過時間に依存しません。3000 仮想年のシミュレーション（数十万回の書込み・10 年の沈黙・時計故障）がテストに含まれています。

## クイックスタート

```dart
final memory = EngramMemory(
  store: InMemoryStore(),               // 自作の DB アダプタに差し替え可
  embedder: myEmbedder,                 // あなたの Embedder 実装
  defaultTimezone: const MemoryTimezone('Asia/Tokyo', Duration(hours: 9)),
);
await memory.initialize();

await memory.remember('ユーザーは京都に住んでいる', salience: 2);   // 書込み
final recall = await memory.recall('どこに住んでいるか覚えてる？');  // 想起 → recall.packText
await memory.cite(llmReply);                                          // 応答が引用した《id》を強化
await memory.dream(adjudicate: (request) async {                      // 夢（オフライン統合）
  final raw = await myLlm.generateJson(request.buildPrompt());
  return DreamDecision.parseJson(raw);   // {"action":"keep"} または {"action":"replace","memories":[...]}
});
```

依存ゼロで動く例（トイ埋め込み＋インメモリストア）: `dart run example/main.dart`

## チャットアプリへの組み込み

```
発話 ─► memory.recall(発話) → 記憶パック
     ─► LLM 呼び出し（EngramPrompts.conversationSystemPromptJa / buildUserMessage / saveMemoryToolSpec, deleteMemoryToolSpec）
     ─► save_memory → memory.remember(text, salience:)   delete_memory → memory.forget(id)
     ─► memory.cite(応答)
```

保存が無かったターンは `EngramPrompts.buildExtractionInstruction` → `parseExtractedTexts` → `remember` で補えます。

## 3 つのインタフェース

- **`Embedder`** — `modelId` / `dimension` / `embedQueries` / `embedDocuments`。正規化は不要。埋め込みが失敗しても `remember` は本文を保持し（後で索引化）、`recall` は空を返し、`dream` は可能な範囲で再索引します。モデル依存の 3 パラメータ（`cosineFloor`≈EmbeddingGemma で 0.4、`thetaRelated`、`gistMinCosine`）は一度較正してください。
- **`MemoryStore`** — `loadAll / put / remove / clear / transaction / backup`（＋ open/close）の 6 操作。全件は RAM に保持され（1 万件 × 768 次元で約 35 MB）、ストアは永続化のみ。`InMemoryStore` 同梱、SQLite アダプタ（WAL・トランザクション・スナップショットリング）は [`example/sqlite_adapter`](example/sqlite_adapter)。
- **`DreamAdjudicator`** — `DreamRequest` を受けて `DreamDecision` を返すコールバック。本文の清浄化・1 裁定 ≤ 8 行・成員数を超える置換の拒否・成員とのコサイン検査（作話ガード）・強度保存の継承・1 クラスタ = 1 トランザクション・失敗クラスタの再試行はエンジンが保証します。

## 時刻とタイムゾーン

全エントリポイントは `nowUnix`（Unix 秒）と `MemoryTimezone` を受け取ります。省略時は注入した `clock` / `defaultTimezone` → システム時計 / UTC。タイムゾーンは `'IANA名;+HH:MM'` で記憶ごとに保存され、ローカル時刻整形は tz データベース不要の純粋な算術で任意の年に対応します。未来のタイムスタンプは読込時にクランプされ、時計逆行で不死の記憶が生まれることはありません。

## API

| メソッド | 目的 |
|---|---|
| `initialize()` | ストアを開き、全件を読込・クランプし、古いモデルのベクトルを再埋め込み |
| `remember(text, {salience, nowUnix, timezone})` | → `RememberResult`（`inserted` / `reinforced` / `rejected` / `rateLimited`、`evicted`） |
| `recall(query, {nowUnix})` | → `RecallResult(packText, recalled)` |
| `cite(replyText)` | 応答が引用した《id》の記憶を完全に強化 |
| `forget(id)` | id 指定の物理削除 → `bool` |
| `dream({adjudicate, budget, nowUnix, timezone})` | オフライン統合 → `List<DreamReport>`（`keep` / `replace` / `error`） |
| `clusters()` | 次の夢が LLM に渡すクラスタ（LLM 呼び出しなし） |
| `memories()` / `memory(id)` / `retrievability` / `strength` / `nowUnix()` / `nowLocal()` / `reset()` | 内省・時計・全消去 |

## 設定（`EngramConfig`、19 個）

`capacity` 10000 / `initialStability` 1 日 / `spacingGain` 3 / `maxStability` 10 年 / `gracePeriod` 3 日 / `cosineFloor` 0 / `alpha` 0.35 / `injectN` 5 / `mmrLambda` 0.3 / `minScore` 0.1 / `relativeScore` 0.6 / `budgetChars` 1024 / `maxCues` 8 / `thetaRelated` 0.75 / `dreamBudget` 5 / `dreamMaxMembers` 8 / `gistMinCosine` 0.5 / `textMax` 170 / `writesPerDay` 1000。意味は SPEC §6 を参照。

## テスト

```bash
dart test                                # パッケージ（3000 年シミュレーション含む）
cd example/sqlite_adapter && dart test   # アダプタ
```

`test/support/fakes.dart` の決定論的 `FakeEmbedder`（Python 版とビット一致）と `VirtualClock`、`test/conformance/` の共有シナリオ／トレースが統合テストの雛形です。

## ライセンス

MIT — [LICENSE](LICENSE)
