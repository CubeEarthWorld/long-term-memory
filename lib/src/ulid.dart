import 'dart:math';

/// Crockford base32 alphabet used by ULID (excludes I, L, O, U).
const String _crockford = '0123456789ABCDEFGHJKMNPQRSTVWXYZ';

final Random _secure = Random.secure();

/// Returns a ULID-shaped id: 26 Crockford-base32 characters, the first 10
/// encoding [millis] as a **50-bit** millisecond Unix timestamp (valid past
/// year 37,000; standard ULIDs stop at 48 bits / year 10,889) and the
/// remaining 16 encoding 80 random bits. Lexicographic order equals time
/// order. [random] defaults to a cryptographically secure RNG.
String ulid(int millis, {Random? random}) {
  final rng = random ?? _secure;
  var ts = max(0, millis);
  final codes = List<int>.filled(26, 0);
  for (var i = 9; i >= 0; i--) {
    codes[i] = _crockford.codeUnitAt(ts % 32);
    ts ~/= 32;
  }
  for (var i = 10; i < 26; i++) {
    codes[i] = _crockford.codeUnitAt(rng.nextInt(32));
  }
  return String.fromCharCodes(codes);
}
