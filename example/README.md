# Examples

## `main.dart` — zero-dependency quick start

Bundled `InMemoryStore` + a toy hash embedder; shows remember → recall →
prompt building → dream end to end.

```bash
dart run example/main.dart
```

## `sqlite_adapter/` — production-style database adapter

A complete `MemoryStore` over [`package:sqlite3`](https://pub.dev/packages/sqlite3):
one table, WAL, a real transaction for dream replacements, and a rotating
snapshot ring (`backup()`).

```bash
cd example/sqlite_adapter
dart pub get
dart test
```

```dart
final store = SqliteMemoryStore('/path/to/memory.db');
final memory = EngramMemory(store: store, embedder: myEmbedder);
await memory.initialize();
```

On Flutter, add [`sqlite3_flutter_libs`](https://pub.dev/packages/sqlite3_flutter_libs);
the adapter works unchanged. The same six operations map directly onto drift /
Isar / Hive / ObjectBox.

## Wiring real models

### firebase_ai (Gemini embeddings + Gemini dream LLM)

```dart
final embeddingModel =
    FirebaseAI.googleAI().generativeModel(model: 'gemini-embedding-001');

final embedder = CallbackEmbedder(
  modelId: 'gemini-embedding-001',
  onEmbedDocuments: (texts) async {
    final res = await embeddingModel.batchEmbedContents([
      for (final t in texts)
        EmbedContentRequest(Content.text(t), taskType: TaskType.retrievalDocument),
    ]);
    return [for (final e in res.embeddings) Float32List.fromList(e.values)];
  },
  onEmbedQueries: (texts) async {
    final res = await embeddingModel.batchEmbedContents([
      for (final t in texts)
        EmbedContentRequest(Content.text(t), taskType: TaskType.retrievalQuery),
    ]);
    return [for (final e in res.embeddings) Float32List.fromList(e.values)];
  },
);

final chat = FirebaseAI.googleAI().generativeModel(
  model: 'gemini-3.5-flash',
  generationConfig: GenerationConfig(responseMimeType: 'application/json'),
);
await memory.dream(adjudicate: (request) async {
  final res = await chat.generateContent([Content.text(request.buildPrompt())]);
  return DreamDecision.parseJson(res.text);
});
```

### llamadart (fully local: EmbeddingGemma GGUF + GGUF LLM)

```dart
final embedLlama = Llama(modelPath: 'embeddinggemma-300m-qat-Q4_0.gguf', embedding: true);
final embedder = CallbackEmbedder(
  modelId: 'embeddinggemma-300m',
  onEmbedDocuments: (texts) async => [
    for (final t in texts) Float32List.fromList(await embedLlama.embed('title: none | text: $t')),
  ],
  onEmbedQueries: (texts) async => [
    for (final t in texts) Float32List.fromList(await embedLlama.embed('task: search result | query: $t')),
  ],
);
// EmbeddingGemma: unrelated texts sit around cos ≈ 0.4 → EngramConfig(cosineFloor: 0.4)

final chatLlama = Llama(modelPath: 'qwen2.5-3b-instruct.gguf');
await memory.dream(adjudicate: (request) async {
  final raw = await chatLlama.generate(request.buildPrompt());
  return DreamDecision.parseJson(raw);
});
```

### Conversation loop (any tool-calling LLM)

```dart
final pack = await memory.recall(userText);
final response = await yourLlm.chat(
  system: EngramPrompts.conversationSystemPromptJa,
  user: EngramPrompts.buildUserMessage(
    currentTime: memory.nowLocal(),
    memoryPack: pack.packText,
    userText: userText,
  ),
  tools: [EngramPrompts.saveMemoryToolSpec, EngramPrompts.deleteMemoryToolSpec],
);
for (final call in response.toolCalls) {
  switch (call.name) {
    case 'save_memory':
      await memory.remember(call.args['text'] as String,
          salience: (call.args['salience'] as num?)?.toDouble() ?? 1);
    case 'delete_memory':
      await memory.forget(call.args['id'] as String);
  }
}
await memory.cite(response.text);
```
