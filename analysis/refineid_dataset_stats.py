"""Reproduce the RefineID dataset-composition statistics used in the paper's
Section III.C table. Reads `data/test.csv` (the 1,000-sample benchmark) and
prints every figure that appears in the table.

Run from the repository root:
    python analysis/refineid_dataset_stats.py
"""
import csv
import re
import sys
from collections import Counter

CSV_PATH = 'data/test.csv'

csv.field_size_limit(sys.maxsize)

with open(CSV_PATH, 'r') as f:
    rows = list(csv.reader(f))

n = len(rows)
codes = [r[1] for r in rows]
names = [r[2] for r in rows]

# --- usage sites ----------------------------------------------------------
sites = [c.count('[MASK]') for c in codes]
total_sites = sum(sites)
sites_sorted = sorted(sites)

# --- code length ----------------------------------------------------------
lens = sorted(len(c) for c in codes)

# --- identifier complexity ------------------------------------------------
def camel_segments(s):
    return len(re.findall(r'[A-Z][a-z0-9]*|[a-z0-9]+', s))

name_lens = sorted(len(x) for x in names)
seg_counts = [camel_segments(x) for x in names]
seg_dist = Counter(seg_counts)

# --- project / system distribution ---------------------------------------
pkg_re = re.compile(r'package\s+([\w\.]+)\s*;')
def bucket(code):
    m = pkg_re.search(code)
    if not m:
        return 'UNKNOWN'
    parts = m.group(1).split('.')
    if not parts:
        return 'UNKNOWN'
    if parts[0] in ('org', 'com', 'io', 'net', 'dev', 'app', 'edu') and len(parts) > 1:
        return parts[1]
    return parts[0]

buckets = Counter(bucket(c) for c in codes)
top3_prefixes = set()
for c in codes:
    m = pkg_re.search(c)
    if m:
        parts = m.group(1).split('.')
        top3_prefixes.add('.'.join(parts[:3]) if len(parts) >= 3 else m.group(1))

# --- identifier kind classification (regex-based, no AST) ----------------
# We approximate the declaration site of the renamed identifier by scanning
# the [MASK] occurrences and picking the strongest declaration context.
JAVA_PRIM = r'(?:void|boolean|byte|short|int|long|float|double|char)'
TYPE_RE = r'(?:[A-Z][\w]*|' + JAVA_PRIM + r')(?:<[^>;{}()]*>)?(?:\[\])*'
KIND_LOCAL = 'local variable'
KIND_PARAM = 'formal parameter'
KIND_FIELD_PRIV = 'private field'
KIND_FIELD_PUB  = 'public/protected field'
KIND_FIELD_PKG  = 'package-private field'
KIND_METHOD = 'method'
KIND_USAGE  = 'usage-only (no decl in file)'
PRECEDENCE = {KIND_METHOD: 6, KIND_FIELD_PUB: 5, KIND_FIELD_PRIV: 5,
              KIND_FIELD_PKG: 5, KIND_PARAM: 4, KIND_LOCAL: 3, KIND_USAGE: 0}

def depth_array(code):
    out = [0] * (len(code) + 1)
    d = 0
    for k, ch in enumerate(code):
        if ch == '{':
            d += 1
        elif ch == '}':
            d -= 1
        out[k + 1] = d
    return out

def classify(code):
    depths = depth_array(code)
    positions = []
    j = 0
    while True:
        i = code.find('[MASK]', j)
        if i < 0:
            break
        positions.append(i)
        j = i + 6
    best, best_prec = None, -1
    for pos in positions:
        ctx_before = code[max(0, pos - 180):pos + 10]
        if re.search(r'(?:public|private|protected|static|final|abstract|default|native|synchronized)\b'
                     r'[\s\w<>,?\[\].]{0,120}\b' + TYPE_RE + r'\s+\[MASK\]\s*\(', ctx_before):
            kind = KIND_METHOD
        elif re.search(r'(?:^|[\s\n])(?:final\s+)?' + TYPE_RE + r'\s+\[MASK\]\s*[=;]',
                       code[max(0, pos - 160):pos + 10]):
            if depths[pos] <= 1:
                ctx = code[max(0, pos - 160):pos]
                if 'private ' in ctx:
                    kind = KIND_FIELD_PRIV
                elif 'public ' in ctx or 'protected ' in ctx:
                    kind = KIND_FIELD_PUB
                else:
                    kind = KIND_FIELD_PKG
            else:
                kind = KIND_LOCAL
        elif re.search(r'(?:\(|,)\s*(?:final\s+)?(?:@\w+(?:\([^)]*\))?\s+)?' + TYPE_RE
                       + r'(?:\s*\.\.\.)?\s+\[MASK\]\s*[,)]', code[max(0, pos - 180):pos + 10]):
            depth_p = 0
            k = pos - 1
            open_pos = None
            while k >= 0:
                if code[k] == ')':
                    depth_p += 1
                elif code[k] == '(':
                    if depth_p == 0:
                        open_pos = k
                        break
                    depth_p -= 1
                k -= 1
            kind = KIND_USAGE
            if open_pos is not None:
                head = code[max(0, open_pos - 160):open_pos]
                has_mod = re.search(r'(?:public|private|protected|static|final|abstract|default|'
                                    r'native|synchronized)\b', head)
                ends_method = re.search(TYPE_RE + r'\s+[A-Za-z_$]\w*\s*$', head)
                if has_mod and ends_method:
                    kind = KIND_PARAM
                elif ends_method and depths[open_pos] <= 1:
                    kind = KIND_PARAM
        else:
            kind = KIND_USAGE
        if PRECEDENCE[kind] > best_prec:
            best = kind
            best_prec = PRECEDENCE[kind]
    return best or KIND_USAGE

kinds = Counter(classify(c) for c in codes)

# --- report ---------------------------------------------------------------
print(f'RefineID statistics (source: {CSV_PATH})')
print('=' * 60)
print(f'Renaming samples (rows)                 : {n}')
print(f'Total masked occurrences (usage sites)  : {total_sites}')
print(f'Unique target identifier strings        : {len(set(names))}')
print()
print(f'Usage sites per renaming  '
      f'min={min(sites)}  median={sites_sorted[n // 2]}  '
      f'mean={total_sites / n:.2f}  max={max(sites)}')
print(f'Source file chars         '
      f'min={lens[0]}  median={lens[n // 2]}  '
      f'mean={sum(lens) // n}  P95={lens[int(0.95 * n)]}  max={lens[-1]}')
print()
print(f'Identifier length (chars)   median={name_lens[n // 2]}  '
      f'mean={sum(name_lens) / n:.2f}  max={name_lens[-1]}')
print(f'CamelCase-segment distribution:')
for s in sorted(seg_dist):
    print(f'   {s} segment(s) : {seg_dist[s]:4d}  ({100 * seg_dist[s] / n:5.1f}%)')

print()
print(f'Distinct top-3 package prefixes: {len(top3_prefixes)}')
print(f'Top "system" buckets (top-2 package label):')
for p, c in buckets.most_common(15):
    print(f'   {c:4d}  {p}')

print()
print('Declaration-site identifier kind (regex heuristic; no AST):')
for k, c in kinds.most_common():
    print(f'   {c:4d}  ({100 * c / n:5.1f}%)  {k}')
