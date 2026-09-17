import 'dart:math';

/// Crockford base32 alphabet used by ULID (excludes I, L, O, U).
const String _crockford = '0123456789ABCDEFGHJKMNPQRSTVWXYZ';

/// Generates ULID-shaped ids: 26 Crockford-base32 characters, the first 10
/// encoding a **50-bit** millisecond Unix timestamp (valid past year 37,000;
/// standard ULIDs stop at 48 bits / year 10,889) and the remaining 16
/// encoding 80 random bits. Lexicographic order equals time order.
///
/// The clock and the random source are injectable so tests (and engines on a
/// virtual clock) produce deterministic, time-consistent ids.
class UlidGenerator {
  /// Creates a generator. [millis] returns the current Unix time in
  /// milliseconds (defaults to the wall clock); [random] defaults to a
  /// cryptographically secure RNG.
  UlidGenerator({int Function()? millis, Random? random})
      : _millis = millis ?? (() => DateTime.now().millisecondsSinceEpoch),
        _random = random ?? Random.secure();

  final int Function() _millis;
  final Random _random;

  /// Returns a new 26-character id.
  String next() {
    var ts = max(0, _millis());
    final codes = List<int>.filled(26, 0);
    for (var i = 9; i >= 0; i--) {
      codes[i] = _crockford.codeUnitAt(ts % 32);
      ts ~/= 32;
    }
    for (var i = 10; i < 26; i++) {
      codes[i] = _crockford.codeUnitAt(_random.nextInt(32));
    }
    return String.fromCharCodes(codes);
  }
}
