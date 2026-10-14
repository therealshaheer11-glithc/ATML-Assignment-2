"""Standard-library scheduling, immutable checkpoints and metric definitions."""
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import uuid

POLICIES = ('sft', 'rlvr', 'rlaif')
DOMAINS = ('gsm', 'transfer')
COUNTS = {'gsm': 300, 'transfer': 100}
LABELS = ('A', 'TIE', 'B')  # A is evaluated policy / clean; B is SFT / modified.


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False,
                       allow_nan=False, separators=(',', ':')) + '\n').encode()


def checksum(value):
    return hashlib.sha256(json_bytes(value)).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(8 * 2**20), b''):
            h.update(part)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_once(path, value):
    """Committed files are immutable; interrupted temporary files are retained."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists() and not path.is_symlink(), 'Refusing to replace committed evidence: ' + str(path))
    pending = path.parent / ('.pending_' + uuid.uuid4().hex)
    payload = json_bytes(value)
    with pending.open('xb') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    require(pending.read_bytes() == payload, 'Temporary evidence readback failed.')
    require(not path.exists(), 'Concurrent committed evidence appeared.')
    os.replace(pending, path)
    require(path.read_bytes() == payload, 'Committed evidence readback failed.')


@contextlib.contextmanager
def process_lock(store):
    """Use a local advisory lock, avoiding Drive filesystem locking assumptions."""
    import fcntl
    import tempfile
    locks = Path(tempfile.gettempdir()) / 'atml_task5_process_locks'
    locks.mkdir(exist_ok=True)
    name = hashlib.sha256(str(Path(store).resolve()).encode()).hexdigest()
    with (locks / name).open('a') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class Journal:
    """One atomic record stores the result AND RNG-after state as one committed unit."""
    def __init__(self, folder, binding):
        self.folder = Path(folder)
        require(not self.folder.is_symlink(), 'Linked result directory.')
        self.folder.mkdir(parents=True, exist_ok=True)
        identity = self.folder / 'BINDING.json'
        if identity.exists():
            require(read_json(identity) == binding, 'Resume identity differs.')
        else:
            require(not list(self.folder.iterdir()), 'Unknown unbound result directory; preserve and review.')
            write_once(identity, binding)
        self.binding_sha = checksum(binding)
        self.units = self.folder / 'units'
        self.units.mkdir(exist_ok=True)
        self._records = self._read_records()

    def _read_records(self):
        paths = sorted(self.units.glob('*.json'))
        require(all(p.name == f'{i:06d}.json' for i, p in enumerate(paths)),
                'Checkpoint sequence has a gap or unexpected filename.')
        previous = self.binding_sha
        records = []
        for index, path in enumerate(paths):
            require(not path.is_symlink(), 'Linked committed checkpoint.')
            record = read_json(path)
            contents = {k: v for k, v in record.items() if k != 'record_sha256'}
            require(record['index'] == index and record['binding_sha256'] == self.binding_sha
                    and record['previous_sha256'] == previous
                    and record.get('record_sha256') == checksum(contents),
                    'Checkpoint chain or frozen identity differs.')
            previous = checksum(record)
            records.append(record)
        return records

    def records(self):
        return list(self._records)

    def append(self, payload, rng_after=None):
        records = self._records
        index = len(records)
        previous = checksum(records[-1]) if records else self.binding_sha
        if records:
            require(checksum(read_json(self.units / f'{index-1:06d}.json')) == previous,
                    'Last committed checkpoint changed.')
        record = {'index': index, 'binding_sha256': self.binding_sha,
                  'previous_sha256': previous, 'payload': payload, 'rng_after': rng_after}
        record['record_sha256'] = checksum(record)
        write_once(self.units / f'{index:06d}.json', record)
        self._records.append(record)
        return record


def generation_schedule(rows):
    require(set(rows) == set(DOMAINS), 'Wrong evaluation domains.')
    for domain in DOMAINS:
        require(len(rows[domain]) == COUNTS[domain], 'Wrong fixed dataset size: ' + domain)
        identities = [str(row['source_index']) for row in rows[domain]]
        require(len(set(identities)) == len(identities), 'Duplicate fixed source ID: ' + domain)
        for row in rows[domain]:
            require(isinstance(row['question'], str) and 'gold_final' in row,
                    'Missing fixed evaluation fields.')
    return [(policy, domain, i, checksum(row)) for policy in POLICIES
            for domain in DOMAINS for i, row in enumerate(rows[domain])]


def judge_key(problem, a, b, model='Qwen/Qwen2.5-3B-Instruct'):
    # Deliberately matches the released helper's JSON format, not checksum().
    return hashlib.sha256(json.dumps({'model': model, 'problem': problem, 'a': a, 'b': b},
                                    sort_keys=True).encode()).hexdigest()


def decode_preference(raw, swapped):
    import re
    match = re.search(r'\b(A|B|TIE)\b', raw.strip().upper())
    result = match.group(1) if match else 'TIE'
    if swapped:
        result = {'A': 'B', 'B': 'A', 'TIE': 'TIE'}[result]
    return result, match is None


def verifier_preference(a, b):
    require(a in (0, 1) and b in (0, 1), 'Exact rewards must be binary.')
    return 'A' if a > b else 'B' if a < b else 'TIE'


def preference_summary(pairs):
    require(bool(pairs), 'Empty comparison denominator.')
    matrix = {v: {a: 0 for a in LABELS} for v in LABELS}
    counts = dict.fromkeys(LABELS, 0)
    ambiguous = decisive = decisive_agree = agreement = 0
    for pair in pairs:
        v, a = pair['verifier_preference'], pair['ai_preference']
        require(v in LABELS and a in LABELS, 'Unknown preference.')
        matrix[v][a] += 1
        counts[a] += 1
        agreement += v == a
        ambiguous += bool(pair['unparseable'])
        if v != 'TIE':
            decisive += 1
            decisive_agree += v == a
    n = len(pairs)
    return {'pairs': n, 'wins': counts['A'], 'ties': counts['TIE'], 'losses': counts['B'],
            'ai_win_rate': (counts['A'] + .5 * counts['TIE']) / n,
            'win_fraction': counts['A'] / n, 'tie_fraction': counts['TIE'] / n,
            'loss_fraction': counts['B'] / n,
            'verifier_judge_agreement': agreement / n,
            'verifier_by_ai_confusion_matrix': matrix,
            'verifier_ties': sum(matrix['TIE'].values()), 'unparseable_ai_outputs': ambiguous,
            'decisive_verifier_pairs': decisive,
            'secondary_agreement_decisive_verifier': decisive_agree / decisive if decisive else None}


def policy_summary(responses, pairs=None):
    require(bool(responses), 'Empty policy response denominator.')
    lengths = [r['response_tokens'] for r in responses]
    require(all(type(n) is int and 0 <= n <= 512 for n in lengths), 'Invalid response length.')
    elapsed = sum(r['generation_seconds'] for r in responses)
    result = {'responses': len(responses),
              'exact_accuracy': statistics.mean(r['exact_reward'] for r in responses),
              'format_compliance': statistics.mean(r['format_compliant'] for r in responses),
              'length_mean': statistics.mean(lengths), 'length_population_sd': statistics.pstdev(lengths),
              'truncated_responses': sum(r['truncated'] for r in responses),
              'generation_seconds': elapsed, 'generated_tokens': sum(lengths),
              'generated_tokens_per_generation_second': sum(lengths) / elapsed if elapsed else None,
              'ai_comparison': preference_summary(pairs) if pairs else None}
    return result


def diagnostic_summary(pairs, variants):
    require(len(pairs) == 80 and len({(p['problem_id'], p['variant']) for p in pairs}) == 80,
            'Expected 80 unique diagnostic pairs.')
    require(set(p['variant'] for p in pairs) == set(variants[1:]), 'Diagnostic category coverage differs.')
    contrasts = {}
    for variant in variants[1:]:
        selected = [p for p in pairs if p['variant'] == variant]
        require(len(selected) == 20, 'Expected 20 pairs per diagnostic contrast.')
        contrasts[variant] = preference_summary(selected)
    return {'unique_pairs': 80, 'clean_reference_all_contrasts': preference_summary(pairs),
            'contrasts': contrasts,
            'S_reason': contrasts['corrupt_reasoning_correct_final']['win_fraction'],
            'S_outcome': contrasts['good_reasoning_wrong_final']['win_fraction'],
            'category_denominators_overlap': True,
            'filler_interpretation': 'Ties acceptable; filler preference is possible style bias.',
            'qualitative_examples': 'Selection/classification conventions still require approval.'}


def transfer_drop(gsm, transfer):
    a, b = gsm['ai_comparison'], transfer['ai_comparison']
    return {'exact_accuracy_drop_percentage_points': 100 * (gsm['exact_accuracy'] - transfer['exact_accuracy']),
            'ai_win_rate_drop_percentage_points': 100 * (a['ai_win_rate'] - b['ai_win_rate']) if a and b else None,
            'mean_length_change_gsm_minus_svamp': gsm['length_mean'] - transfer['length_mean'],
            'ai_reference': 'SFT evaluated separately on each fixed domain.'}
