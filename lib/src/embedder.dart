import 'dart:typed_data';

/// The embedding model interface — implemented by the application.
///
/// The package never loads a model itself: wrap whatever you use
/// (firebase_ai, llamadart, an ONNX runtime, a REST endpoint, ...) in this
/// interface. Asymmetric retrieval prompts are supported by having separate
/// query/document methods; if your model is symmetric, implement both with
/// the same call.
///
/// Returned vectors do **not** need to be normalised — the engine
/// L2-normalises every vector it receives. Any dimension works, as long as
/// every vector of one model has the same length.
abstract class Embedder {
  /// Stable identifier of the model (e.g. `'google/embeddinggemma-300m'`).
  /// Stamped on every stored vector; after a model switch, traces indexed
  /// under another model are re-embedded from their text.
  String get modelId;

  /// Length of the vectors this model returns.
  ///
  /// One [modelId] means one dimension: the engine treats a stored vector of
  /// any other length as stale and re-embeds it, and a returned vector of any
  /// other length as an embedder fault. Declaring it lets both happen before a
  /// mixed-dimension vector reaches a dot product.
  int get dimension;

  /// Embeds search queries (the "query" side of an asymmetric model).
  Future<List<Float32List>> embedQueries(List<String> texts);

  /// Embeds stored propositions (the "document" side).
  Future<List<Float32List>> embedDocuments(List<String> texts);
}

/// Convenience [Embedder] built from callbacks — handy when wiring an SDK
/// without declaring a class:
///
/// ```dart
/// final embedder = CallbackEmbedder(
///   modelId: 'text-embedding-004',
///   dimension: 768,
///   onEmbedQueries: (texts) => myApi.embed(texts, taskType: 'RETRIEVAL_QUERY'),
///   onEmbedDocuments: (texts) => myApi.embed(texts, taskType: 'RETRIEVAL_DOCUMENT'),
/// );
/// ```
class CallbackEmbedder implements Embedder {
  /// Creates an embedder from two callbacks. If [onEmbedQueries] is omitted,
  /// queries use [onEmbedDocuments] (symmetric model).
  CallbackEmbedder({
    required this.modelId,
    required this.dimension,
    required Future<List<Float32List>> Function(List<String> texts)
        onEmbedDocuments,
    Future<List<Float32List>> Function(List<String> texts)? onEmbedQueries,
  })  : _documents = onEmbedDocuments,
        _queries = onEmbedQueries ?? onEmbedDocuments;

  @override
  final String modelId;

  @override
  final int dimension;

  final Future<List<Float32List>> Function(List<String>) _queries;
  final Future<List<Float32List>> Function(List<String>) _documents;

  @override
  Future<List<Float32List>> embedQueries(List<String> texts) => _queries(texts);

  @override
  Future<List<Float32List>> embedDocuments(List<String> texts) =>
      _documents(texts);
}
