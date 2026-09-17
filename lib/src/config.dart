/// ENGRAM v2 parameters (SPEC §6). Defaults are the reference values; tune
/// only with a record of why.
class EngramConfig {
  /// Creates a configuration. Every parameter has the spec default.
  const EngramConfig({
    this.capacity = 10000,
    this.initialStability = _day,
    this.spacingGain = 3.0,
    this.maxStability = 10 * _year,
    this.gracePeriod = 3 * _day,
    this.cosineFloor = 0.0,
    this.alpha = 0.35,
    this.injectN = 5,
    this.mmrLambda = 0.3,
    this.minScore = 0.1,
    this.relativeScore = 0.6,
    this.budgetChars = 1024,
    this.maxCues = 8,
    this.thetaRelated = 0.75,
    this.dreamBudget = 5,
    this.dreamMaxMembers = 8,
    this.gistMinCosine = 0.5,
    this.textMax = 170,
    this.writesPerDay = 1000,
  })  : assert(capacity > 0, 'capacity must be positive'),
        assert(initialStability > 0 && initialStability <= maxStability,
            '0 < initialStability <= maxStability'),
        assert(spacingGain >= 0 && gracePeriod >= 0,
            'spacingGain and gracePeriod must be non-negative'),
        assert(alpha >= 0 && alpha <= 1, 'alpha must be in [0, 1]'),
        assert(cosineFloor >= 0 && cosineFloor < 1,
            'cosineFloor must be in [0, 1)'),
        assert(relativeScore >= 0 && relativeScore <= 1,
            'relativeScore must be in [0, 1]'),
        assert(thetaRelated > 0 && thetaRelated < 1,
            'thetaRelated must be in (0, 1)'),
        assert(dreamMaxMembers >= 2, 'dreamMaxMembers must be at least 2'),
        assert(gistMinCosine >= 0 && gistMinCosine <= 1,
            'gistMinCosine must be in [0, 1]'),
        assert(budgetChars > 0 && textMax > 0 && maxCues > 0 && injectN > 0,
            'budgetChars, textMax, maxCues and injectN must be positive');

  static const double _day = 24 * 60 * 60;
  static const double _year = 365 * _day;

  /// Maximum number of traces; the weakest is forgotten beyond this.
  final int capacity;

  /// Stability (half-life of retrievability, seconds) of a new trace.
  final double initialStability;

  /// Stability growth on recall: `S ← S·(1 + gain·(1−R))`.
  final double spacingGain;

  /// Upper bound of stability in seconds — guarantees no immortal memory.
  final double maxStability;

  /// Traces younger than this (seconds) are not eviction candidates while
  /// any older trace exists — the consolidation window in which a new
  /// memory gets its chance to be recalled.
  final double gracePeriod;

  /// Baseline cosine of unrelated text under the embedding model (e.g. ≈0.4
  /// for EmbeddingGemma, 0 for models centred at zero). Cosines are
  /// rescaled to `(cos − floor) / (1 − floor)` before scoring so that
  /// [minScore] and cue activation mean the same thing for every model.
  final double cosineFloor;

  /// Retrievability floor in the score: dormant but relevant traces still
  /// compete.
  final double alpha;

  /// Number of traces injected per recall.
  final int injectN;

  /// MMR diversity penalty λ.
  final double mmrLambda;

  /// Absolute minimum score for injection (nothing is injected below it —
  /// unrelated traces must not be spuriously strengthened).
  final double minScore;

  /// Relative cut: candidates below `relativeScore × best score` are dropped,
  /// so a strong hit is not accompanied by a noisy tail.
  final double relativeScore;

  /// Character budget of the injected pack.
  final int budgetChars;

  /// Maximum query cues (lines / sentences) embedded per recall.
  final int maxCues;

  /// Cosine at or above which traces are neighbours for dream clustering.
  final double thetaRelated;

  /// Default LLM adjudications per `dream()` call.
  final int dreamBudget;

  /// Maximum members per cluster (bounds the blast radius of one verdict).
  final int dreamMaxMembers;

  /// A replacement text must reach this cosine against at least one cluster
  /// member; otherwise the whole verdict is treated as keep (confabulation /
  /// injection guard).
  final double gistMinCosine;

  /// Maximum characters of one stored proposition (longer text is shortened
  /// at a sentence boundary).
  final int textMax;

  /// Soft write-rate limit per rolling 24 h. Together with [gracePeriod] it
  /// bounds how much of the store a write flood can displace.
  final int writesPerDay;

  /// Returns a copy with the given fields replaced.
  EngramConfig copyWith({
    int? capacity,
    double? initialStability,
    double? spacingGain,
    double? maxStability,
    double? gracePeriod,
    double? cosineFloor,
    double? alpha,
    int? injectN,
    double? mmrLambda,
    double? minScore,
    double? relativeScore,
    int? budgetChars,
    int? maxCues,
    double? thetaRelated,
    int? dreamBudget,
    int? dreamMaxMembers,
    double? gistMinCosine,
    int? textMax,
    int? writesPerDay,
  }) =>
      EngramConfig(
        capacity: capacity ?? this.capacity,
        initialStability: initialStability ?? this.initialStability,
        spacingGain: spacingGain ?? this.spacingGain,
        maxStability: maxStability ?? this.maxStability,
        gracePeriod: gracePeriod ?? this.gracePeriod,
        cosineFloor: cosineFloor ?? this.cosineFloor,
        alpha: alpha ?? this.alpha,
        injectN: injectN ?? this.injectN,
        mmrLambda: mmrLambda ?? this.mmrLambda,
        minScore: minScore ?? this.minScore,
        relativeScore: relativeScore ?? this.relativeScore,
        budgetChars: budgetChars ?? this.budgetChars,
        maxCues: maxCues ?? this.maxCues,
        thetaRelated: thetaRelated ?? this.thetaRelated,
        dreamBudget: dreamBudget ?? this.dreamBudget,
        dreamMaxMembers: dreamMaxMembers ?? this.dreamMaxMembers,
        gistMinCosine: gistMinCosine ?? this.gistMinCosine,
        textMax: textMax ?? this.textMax,
        writesPerDay: writesPerDay ?? this.writesPerDay,
      );

  /// JSON form (for persisting / displaying the active configuration).
  Map<String, Object?> toJson() => {
        'capacity': capacity,
        'initialStability': initialStability,
        'spacingGain': spacingGain,
        'maxStability': maxStability,
        'gracePeriod': gracePeriod,
        'cosineFloor': cosineFloor,
        'alpha': alpha,
        'injectN': injectN,
        'mmrLambda': mmrLambda,
        'minScore': minScore,
        'relativeScore': relativeScore,
        'budgetChars': budgetChars,
        'maxCues': maxCues,
        'thetaRelated': thetaRelated,
        'dreamBudget': dreamBudget,
        'dreamMaxMembers': dreamMaxMembers,
        'gistMinCosine': gistMinCosine,
        'textMax': textMax,
        'writesPerDay': writesPerDay,
      };

  /// Lenient inverse of [toJson]: missing or mistyped values fall back to
  /// the defaults.
  factory EngramConfig.fromJson(Map<String, Object?> json) {
    const d = EngramConfig();
    int i(String k, int def) => switch (json[k]) {
          final num v => v.toInt(),
          final String v => int.tryParse(v) ?? def,
          _ => def,
        };
    double f(String k, double def) => switch (json[k]) {
          final num v => v.toDouble(),
          final String v => double.tryParse(v) ?? def,
          _ => def,
        };
    return EngramConfig(
      capacity: i('capacity', d.capacity),
      initialStability: f('initialStability', d.initialStability),
      spacingGain: f('spacingGain', d.spacingGain),
      maxStability: f('maxStability', d.maxStability),
      gracePeriod: f('gracePeriod', d.gracePeriod),
      cosineFloor: f('cosineFloor', d.cosineFloor),
      alpha: f('alpha', d.alpha),
      injectN: i('injectN', d.injectN),
      mmrLambda: f('mmrLambda', d.mmrLambda),
      minScore: f('minScore', d.minScore),
      relativeScore: f('relativeScore', d.relativeScore),
      budgetChars: i('budgetChars', d.budgetChars),
      maxCues: i('maxCues', d.maxCues),
      thetaRelated: f('thetaRelated', d.thetaRelated),
      dreamBudget: i('dreamBudget', d.dreamBudget),
      dreamMaxMembers: i('dreamMaxMembers', d.dreamMaxMembers),
      gistMinCosine: f('gistMinCosine', d.gistMinCosine),
      textMax: i('textMax', d.textMax),
      writesPerDay: i('writesPerDay', d.writesPerDay),
    );
  }
}
