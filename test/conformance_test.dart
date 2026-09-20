import 'dart:convert';
import 'dart:io';

import 'package:long_term_memory/long_term_memory.dart';
import 'package:test/test.dart';

import 'support/fakes.dart';

/// Cross-language conformance: runs the scripted scenario in
/// `test/conformance/scenario.json` and compares the resulting trace with
/// `test/conformance/trace.json`, which the Python reference implementation
/// must reproduce (its test loads the same two files). Regenerate with
/// `CONFORMANCE_WRITE=1 dart test test/conformance_test.dart`.
void main() {
  test('scripted scenario matches the shared trace', () async {
    final dir = Directory('test/conformance');
    final scenario =
        jsonDecode(File('${dir.path}/scenario.json').readAsStringSync()) as Map;
    final config = EngramConfig.fromJson((scenario['config'] as Map).cast());
    final (memory, clock, _) = await build(
      config: config,
      embedder: FakeEmbedder(dimension: scenario['dimension'] as int),
    );
    final trace = <Map<String, Object?>>[];
    String r6(num v) => v.toStringAsFixed(6);
    for (final raw in scenario['steps'] as List) {
      final step = (raw as Map).cast<String, Object?>();
      final op = step['op'] as String;
      final row = <String, Object?>{'op': op};
      switch (op) {
        case 'advance':
          clock.advance(step['seconds'] as int);
        case 'remember':
          final r = await memory.remember(step['text'] as String,
              salience: (step['salience'] as num?)?.toDouble() ?? 1.0);
          row['action'] = r.action.name;
          row['text'] = r.memory?.text;
          row['stability'] = r.memory == null ? null : r6(r.memory!.stability);
          row['consolidated'] = r.memory?.consolidated;
          row['evicted'] = r.evicted?.text;
        case 'recall':
          final r = await memory.recall(step['query'] as String);
          row['texts'] = [for (final c in r.recalled) c.memory.text];
          row['scores'] = [for (final c in r.recalled) r6(c.score)];
          row['pack'] = r.packText.replaceAll(RegExp('《id:[^》]+》'), '《id》');
        case 'cite':
          final texts = (step['texts'] as List).cast<String>();
          final ids = [
            for (final m in await memory.memories())
              if (texts.contains(m.text)) '《id:${m.id}》',
          ];
          row['cited'] = (await memory.cite(ids.join(' '))).length;
        case 'dream':
          final reports = await memory.dream(
              adjudicate: mergeToGist, budget: step['budget'] as int?);
          row['reports'] = [
            for (final rep in reports)
              {
                'action': rep.action.name,
                'before': [for (final m in rep.before) m.text],
                'after': [for (final m in rep.after) m.text],
                'stability': [for (final m in rep.after) r6(m.stability)],
              },
          ];
        case 'forget':
          final all = await memory.memories();
          final i = all.indexWhere((m) => m.text == step['text']);
          row['forgot'] = i >= 0 && await memory.forget(all[i].id);
      }
      final snapshot = [
        for (final m in await memory.memories())
          {
            'text': m.text,
            'S': r6(m.stability),
            'R': r6(memory.retrievability(m, clock.t)),
            'c': m.consolidated,
          },
      ];
      row['state'] = snapshot;
      trace.add(row);
    }
    final file = File('${dir.path}/trace.json');
    final encoded = const JsonEncoder.withIndent(' ').convert(trace);
    // No `!file.existsSync()` fallback: a missing baseline must make the read
    // throw, not let the test write the trace and compare it to itself.
    if (Platform.environment['CONFORMANCE_WRITE'] == '1') {
      file.writeAsStringSync('$encoded\n');
    }
    expect(jsonDecode(file.readAsStringSync()), jsonDecode(encoded));
  });
}
