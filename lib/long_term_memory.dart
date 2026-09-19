/// A portable long-term memory engine for LLM applications (ENGRAM v2).
///
/// Bring your own LLM, embedding model and database — the package contains
/// only the algorithm:
///
/// ```dart
/// final memory = EngramMemory(
///   store: InMemoryStore(),          // or your own MemoryStore adapter
///   embedder: myEmbedder,            // your Embedder implementation
///   defaultTimezone: const MemoryTimezone('Asia/Tokyo', Duration(hours: 9)),
/// );
/// await memory.initialize();
///
/// await memory.remember('ユーザーは京都に住んでいる');
/// final recall = await memory.recall('どこに住んでいる？');
/// await memory.dream(adjudicate: myLlmCallback);
/// ```
///
/// See the README for the full architecture and integration recipes.
library;

export 'src/config.dart' show EngramConfig;
export 'src/dream/adjudicator.dart'
    show DreamAdjudicator, DreamDecision, DreamMember, DreamRequest;
export 'src/dream/prompts.dart' show EngramLocale, EngramPrompts;
export 'src/embedder.dart' show CallbackEmbedder, Embedder;
export 'src/engine.dart' show EngramMemory;
export 'src/json.dart' show relaxedJsonDecode;
export 'src/models.dart'
    show
        DreamAction,
        DreamReport,
        Memory,
        Recalled,
        RecallResult,
        RememberAction,
        RememberResult;
export 'src/store/in_memory_store.dart' show InMemoryStore;
export 'src/store/memory_store.dart' show MemoryStore;
export 'src/timezone.dart' show MemoryTimezone, formatLocal;
export 'src/vector_math.dart' show l2Normalized, packF32, unpackF32;
