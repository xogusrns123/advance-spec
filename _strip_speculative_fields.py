"""Stream agent_results_eagle3.json and write a slim clone with all
speculative-decoding internals stripped (logits, draft tokens, oracle
entries, full pool). Keeps text content, tool calls, timing, and metrics.

Usage:
    python3 _strip_speculative_fields.py <input.json> <output.json>
"""
import ijson, json, os, sys

HEAVY_KEYS_DROP = {
    # All draft / oracle / pool internals — these carry logits and tokens.
    'spec_decode',
    'oracle_vanilla_entries',
    'oracle_vanilla',
    'eagle3_pool_full',
    'eagle3_pool',
    'eagle3_entries',
    'eagle3',
    'draft_logits',
    'draft_tokens',
    'draft_token_ids',
    'mtp_logits',
    'mtp_tokens',
    'logits',
    'logprobs',
    'top_logprobs',
}

def clean(obj):
    """Recursively strip heavy keys."""
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items() if k not in HEAVY_KEYS_DROP}
    if isinstance(obj, list):
        return [clean(x) for x in obj]
    return obj

def main(in_path, out_path):
    if not os.path.exists(in_path):
        print(f'MISSING: {in_path}', file=sys.stderr)
        return 1
    n_q = 0
    in_size = os.path.getsize(in_path)
    print(f'IN  {in_path}  ({in_size/1e9:.2f} GB)', flush=True)
    # ijson streams questions.item. Top-level may have other keys
    # (e.g. metadata). We re-emit a minimal envelope: {questions: [...]}.
    with open(in_path, 'rb') as fin, open(out_path, 'w') as fout:
        fout.write('{\n  "questions": [\n')
        first = True
        for q in ijson.items(fin, 'questions.item'):
            slim = clean(q)
            if not first:
                fout.write(',\n')
            first = False
            chunk = json.dumps(slim, ensure_ascii=False, default=str, indent=2)
            # indent the whole record by 4 spaces so it sits inside the array
            fout.write('    ' + chunk.replace('\n', '\n    '))
            n_q += 1
            if n_q % 10 == 0:
                fout.flush()
                print(f'  ...{n_q} questions', flush=True)
        fout.write('\n  ]\n}\n')
    out_size = os.path.getsize(out_path)
    print(f'OUT {out_path}  ({out_size/1e6:.1f} MB)  questions={n_q}  shrink={in_size/max(out_size,1):.0f}x',
          flush=True)
    return 0

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    sys.exit(main(sys.argv[1], sys.argv[2]))
