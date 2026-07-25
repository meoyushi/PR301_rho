// Headless runner for the vendored ats-screener scoring rules.
// Reads a ScoringInput JSON object on stdin, writes ScoreResult[] on stdout.
// Deterministic: pure rule-based scoring, no LLM, no network.
import { scoreResume } from './scorer/engine.ts';

const chunks = [];
for await (const chunk of process.stdin) chunks.push(chunk);

try {
	const input = JSON.parse(Buffer.concat(chunks).toString('utf8'));
	const results = scoreResume(input);
	process.stdout.write(JSON.stringify(results));
} catch (err) {
	process.stderr.write(String(err && err.stack ? err.stack : err));
	process.exit(1);
}
