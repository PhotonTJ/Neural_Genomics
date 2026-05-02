import json, os, glob

for f in glob.glob('generated_responses/*_responses.jsonl'):
    records = []
    with open(f) as fh:
        for line in fh:
            r = json.loads(line)
            r.pop('output_dpo', None)  # remove old DPO responses
            records.append(r)
    with open(f, 'w') as fh:
        for r in records:
            fh.write(json.dumps(r) + '\n')
    print(f'Cleaned DPO from {f}')