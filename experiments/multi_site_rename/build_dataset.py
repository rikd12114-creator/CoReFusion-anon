"""Build a multiple-identifier / multiple-site rename dataset from real Java code.

For each source file we:
  1. parse it with ``javalang`` (AST),
  2. find renameable identifiers (local variables + formal parameters) that each
     occur at >= ``--min-sites`` usage sites,
  3. group them into one or more *instances* of ``--num-identifiers`` identifiers,
  4. consistently replace every occurrence of each chosen identifier with a
     placeholder ``IDENT_0 .. IDENT_{k-1}`` (token-accurate, via the tokenizer),
  5. emit a JSONL record holding the masked code + the ground-truth name map.

Sources come from (in order, all optional):
  * ``--repos repos.json``  -> fetched from raw.githubusercontent.com (cached)
  * ``--local-dir DIR``     -> every ``*.java`` under DIR (e.g. a cloned repo)
  * ``data/sample_java/``   -> bundled offline fallback

Run from the repository root, e.g.:
    python experiments/multi_site_rename/build_dataset.py \
        --repos experiments/multi_site_rename/repos.json \
        --out   experiments/multi_site_rename/data/multisite_rename.jsonl \
        --num-identifiers 3 --min-sites 2 --instances-per-file 2
"""
import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# Names we never treat as a renameable target even if they parse as a local/param.
JAVA_KEYWORDS = {
    "abstract", "assert", "boolean", "break", "byte", "case", "catch", "char",
    "class", "const", "continue", "default", "do", "double", "else", "enum",
    "extends", "final", "finally", "float", "for", "goto", "if", "implements",
    "import", "instanceof", "int", "interface", "long", "native", "new",
    "package", "private", "protected", "public", "return", "short", "static",
    "strictfp", "super", "switch", "synchronized", "this", "throw", "throws",
    "transient", "try", "void", "volatile", "while", "var", "true", "false",
    "null",
}


def strip_comments(source):
    """Blank out // line and /* */ block comments (incl. Javadoc) with spaces,
    preserving newlines and char offsets. String/char literals are left intact.

    Useful as a stricter setting: comments often restate a variable's name, which
    would leak the answer for a rename benchmark.
    """
    out = []
    i, n = 0, len(source)
    while i < n:
        c = source[i]
        nxt = source[i + 1] if i + 1 < n else ""
        if c == "/" and nxt == "/":
            while i < n and source[i] != "\n":
                out.append(" ")
                i += 1
        elif c == "/" and nxt == "*":
            while i < n and not (source[i] == "*" and i + 1 < n and source[i + 1] == "/"):
                out.append("\n" if source[i] == "\n" else " ")
                i += 1
            out.append("  ")  # the closing */
            i += 2
        elif c == '"' or c == "'":
            quote = c
            out.append(c)
            i += 1
            while i < n:
                out.append(source[i])
                if source[i] == "\\" and i + 1 < n:
                    out.append(source[i + 1])
                    i += 2
                    continue
                if source[i] == quote:
                    i += 1
                    break
                i += 1
        else:
            out.append(c)
            i += 1
    return "".join(out)


def line_offsets(text):
    """Return a list where starts[i] is the char index at which line i+1 starts."""
    starts = [0]
    for i, ch in enumerate(text):
        if ch == "\n":
            starts.append(i + 1)
    return starts


def token_char_span(starts, position, value):
    """Map a javalang (line, column) position to an absolute [start, end) span."""
    line, col = position  # both 1-based; col points at the first char
    start = starts[line - 1] + (col - 1)
    return start, start + len(value)


def parse_unit(source):
    """Parse the file and return (tree, eligible_names).

    eligible_names: local-variable + formal-parameter names that don't shadow a
    class / method / field / enum name (those are excluded to avoid masking
    unrelated tokens that merely share the name).
    """
    import javalang

    tree = javalang.parse.parse(source)
    local_param_names = set()
    excluded = set()

    for _, node in tree.filter(javalang.tree.LocalVariableDeclaration):
        for d in node.declarators:
            local_param_names.add(d.name)
    for _, node in tree.filter(javalang.tree.FormalParameter):
        local_param_names.add(node.name)
    for _, node in tree.filter(javalang.tree.MethodDeclaration):
        excluded.add(node.name)
    for _, node in tree.filter(javalang.tree.ConstructorDeclaration):
        excluded.add(node.name)
    for _, node in tree.filter(javalang.tree.FieldDeclaration):
        for d in node.declarators:
            excluded.add(d.name)
    for node_type in (javalang.tree.ClassDeclaration,
                      javalang.tree.InterfaceDeclaration,
                      javalang.tree.EnumDeclaration):
        for _, node in tree.filter(node_type):
            excluded.add(node.name)

    return tree, {n for n in local_param_names if n not in excluded}


def collect_spans(text, eligible):
    """Token-accurate [start, end) spans (relative to `text`) per eligible name."""
    import javalang

    starts = line_offsets(text)
    spans = {}
    try:
        toks = list(javalang.tokenizer.tokenize(text))
    except Exception:
        return spans
    for tok in toks:
        if isinstance(tok, javalang.tokenizer.Identifier) and tok.value in eligible:
            spans.setdefault(tok.value, []).append(
                token_char_span(starts, tok.position, tok.value))
    return spans


def method_segments(source, tree):
    """Yield (method_name, start, end) char spans for every method/constructor
    that has a body. `start` is the beginning of the signature line; `end` is just
    past the body's closing brace (found by brace matching)."""
    import javalang

    starts = line_offsets(source)
    n = len(source)
    out = []
    node_types = (javalang.tree.MethodDeclaration, javalang.tree.ConstructorDeclaration)
    for node_type in node_types:
        for _, node in tree.filter(node_type):
            if not node.position:
                continue
            name_off = token_char_span(starts, node.position, node.name or "x")[0]
            # walk forward past the parameter list to the body '{' (or ';' = no body)
            i, depth = name_off, 0
            body_open = None
            while i < n:
                c = source[i]
                if c == "(":
                    depth += 1
                elif c == ")":
                    depth -= 1
                elif depth <= 0 and c == ";":
                    break
                elif depth <= 0 and c == "{":
                    body_open = i
                    break
                i += 1
            if body_open is None:
                continue
            d, j, body_end = 0, body_open, None
            while j < n:
                if source[j] == "{":
                    d += 1
                elif source[j] == "}":
                    d -= 1
                    if d == 0:
                        body_end = j + 1
                        break
                j += 1
            if body_end is None:
                continue
            sig_start = source.rfind("\n", 0, name_off) + 1
            out.append((node.name, sig_start, body_end))
    out.sort(key=lambda t: t[1])
    return out


def choose_identifier_groups(spans, num_identifiers, min_sites, min_len,
                             instances_per_file, min_identifiers):
    """Pick disjoint groups of identifiers, each group an instance to mask."""
    ranked = sorted(
        (n for n, s in spans.items() if len(s) >= min_sites and len(n) >= min_len),
        key=lambda n: (-len(spans[n]), n),
    )
    groups = []
    for i in range(0, len(ranked), num_identifiers):
        if len(groups) >= instances_per_file:
            break
        chunk = ranked[i:i + num_identifiers]
        if len(chunk) >= min_identifiers:
            groups.append(chunk)
    return groups


def mask_source(source, group, spans):
    """Replace every site of each chosen name with IDENT_k placeholders."""
    placeholder_of = {name: f"IDENT_{k}" for k, name in enumerate(group)}
    edits = []  # (start, end, placeholder)
    for name in group:
        ph = placeholder_of[name]
        for (start, end) in spans[name]:
            edits.append((start, end, ph))
    edits.sort(key=lambda e: e[0], reverse=True)
    masked = source
    for start, end, ph in edits:
        masked = masked[:start] + ph + masked[end:]
    ground_truth = {placeholder_of[name]: name for name in group}
    sites = {placeholder_of[name]: len(spans[name]) for name in group}
    return masked, ground_truth, sites


def iter_local_sources(local_dir, sample_dir):
    for d in [local_dir, sample_dir]:
        if not d or not os.path.isdir(d):
            continue
        for root, _, files in os.walk(d):
            for fn in files:
                if fn.endswith(".java"):
                    p = os.path.join(root, fn)
                    yield os.path.splitext(os.path.relpath(p, d))[0].replace(os.sep, "_"), p


def fetch_repo_files(repos_path, cache_dir):
    import requests
    os.makedirs(cache_dir, exist_ok=True)
    with open(repos_path) as f:
        spec = json.load(f)
    out = []
    for entry in spec.get("files", []):
        name = entry["name"]
        cache = os.path.join(cache_dir, f"{name}.java")
        if os.path.exists(cache) and os.path.getsize(cache) > 0:
            out.append((name, cache, entry))
            continue
        url = f"https://raw.githubusercontent.com/{entry['repo']}/{entry['ref']}/{entry['path']}"
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200 and r.text.strip():
                with open(cache, "w") as f:
                    f.write(r.text)
                out.append((name, cache, entry))
                print(f"  fetched {name}")
            else:
                print(f"  SKIP {name}: HTTP {r.status_code} for {url}")
        except Exception as e:  # network errors, timeouts, ...
            print(f"  SKIP {name}: {e}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repos", default=os.path.join(HERE, "repos.json"),
                    help="repos.json describing famous OSS files to fetch")
    ap.add_argument("--local-dir", default=None,
                    help="extra directory of .java files to include (e.g. a clone)")
    ap.add_argument("--out", default=os.path.join(HERE, "data", "multisite_rename.jsonl"))
    ap.add_argument("--cache-dir", default=os.path.join(HERE, "data", "raw_java"))
    ap.add_argument("--num-identifiers", type=int, default=3,
                    help="identifiers to mask per instance")
    ap.add_argument("--min-identifiers", type=int, default=2,
                    help="reject an instance with fewer than this many identifiers")
    ap.add_argument("--min-sites", type=int, default=2,
                    help="each identifier must occur at >= this many sites")
    ap.add_argument("--min-len", type=int, default=3,
                    help="ignore identifiers shorter than this (skip i/j loop vars)")
    ap.add_argument("--scope", choices=["method", "file"], default="method",
                    help="'method' carves one instance per method (more, shorter "
                         "samples); 'file' masks across the whole file")
    ap.add_argument("--instances-per-file", type=int, default=6,
                    help="max instances to carve from one file")
    ap.add_argument("--max-chars", type=int, default=6000,
                    help="skip any masked unit (method or file) longer than this")
    ap.add_argument("--strip-comments", action="store_true",
                    help="blank out comments first (stricter: avoids name leaks in prose)")
    ap.add_argument("--no-fetch", action="store_true",
                    help="do not hit the network; use cache + local + sample only")
    args = ap.parse_args()

    try:
        import javalang  # noqa: F401
    except ImportError:
        sys.exit("javalang not installed. Run: pip install javalang")

    sample_dir = os.path.join(HERE, "data", "sample_java")
    sources = []  # (name, path, meta)

    if args.repos and os.path.exists(args.repos) and not args.no_fetch:
        print(f"Fetching repo files listed in {args.repos} ...")
        sources += fetch_repo_files(args.repos, args.cache_dir)
    elif os.path.isdir(args.cache_dir):
        for fn in sorted(os.listdir(args.cache_dir)):
            if fn.endswith(".java"):
                sources.append((fn[:-5], os.path.join(args.cache_dir, fn), {}))

    for name, path in iter_local_sources(args.local_dir, sample_dir):
        sources.append((name, path, {}))

    print(f"\n{len(sources)} candidate source file(s). Building instances ...")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    records, n_parse_fail, n_no_cand = 0, 0, 0
    per_unit_cap = 1 if args.scope == "method" else args.instances_per_file
    with open(args.out, "w") as out_f:
        for name, path, meta in sources:
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    source = f.read()
            except Exception as e:
                print(f"  read fail {name}: {e}")
                continue
            if args.strip_comments:
                source = strip_comments(source)
            try:
                tree, eligible = parse_unit(source)
            except Exception:
                n_parse_fail += 1
                continue

            if args.scope == "method":
                units = [(m, source[s:e]) for (m, s, e) in method_segments(source, tree)]
            else:
                units = [("file", source)]

            file_inst = 0
            for unit_label, unit_text in units:
                if file_inst >= args.instances_per_file:
                    break
                if len(unit_text) > args.max_chars:
                    continue
                spans = collect_spans(unit_text, eligible)
                groups = choose_identifier_groups(
                    spans, args.num_identifiers, args.min_sites, args.min_len,
                    per_unit_cap, args.min_identifiers)
                for group in groups:
                    if file_inst >= args.instances_per_file:
                        break
                    masked, gt, sites = mask_source(unit_text, group, spans)
                    rec = {
                        "id": f"{name}#{file_inst}",
                        "source_name": name,
                        "repo": meta.get("repo", ""),
                        "path": meta.get("path", path),
                        "ref": meta.get("ref", ""),
                        "scope": args.scope,
                        "unit": unit_label,
                        "num_identifiers": len(group),
                        "total_sites": sum(sites.values()),
                        "masked_code": masked,
                        "ground_truth": gt,   # {"IDENT_0": "name", ...}
                        "sites": sites,       # {"IDENT_0": 4, ...}
                        "orig_len": len(unit_text),
                    }
                    out_f.write(json.dumps(rec) + "\n")
                    records += 1
                    file_inst += 1
            if file_inst == 0:
                n_no_cand += 1

    print(f"\nDone. Wrote {records} instance(s) to {args.out}")
    print(f"  parse failures: {n_parse_fail}   no-candidate files: {n_no_cand}")
    if records == 0:
        print("\nNo instances produced. Try lowering --min-sites / --min-identifiers,"
              " raising --max-chars, or pointing --local-dir at a cloned repo.")


if __name__ == "__main__":
    main()
