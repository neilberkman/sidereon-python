"""Enumerate the public API of sidereon-core and sidereon from source.

Walks the module tree from each crate's lib.rs, following `mod` declarations
(pub or private), records every `pub` item defined at module level with its
defining module, resolves `pub use` re-exports, and reports the items reachable
through public paths. Also lists inherent `pub fn` methods of reachable types.

Output: JSON list of {crate, paths, kind, name, defined_in, file, line, cfg,
hidden, non_exhaustive, methods}. Only a source reading: no build, no rustdoc.

    python3 scripts/coverage/pubapi.py <sidereon checkout> > pubapi.json
    python3 scripts/coverage/bindmap.py . pubapi.json bindmap.json
    python3 scripts/coverage/render_coverage.py bindmap.json <core rev> COVERAGE.md
"""

import json
import os
import re
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
CRATES = {
    "sidereon_core": os.path.join(ROOT, "crates/sidereon-core/src/lib.rs"),
    "sidereon": os.path.join(ROOT, "crates/sidereon/src/lib.rs"),
}


def strip(text):
    """Blank comments, string and char literals, keeping offsets and lines."""
    out = list(text)
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if text.startswith("//", i):
            j = text.find("\n", i)
            j = n if j < 0 else j
            for k in range(i, j):
                out[k] = " "
            i = j
        elif text.startswith("/*", i):
            depth, j = 1, i + 2
            while j < n and depth:
                if text.startswith("/*", j):
                    depth += 1
                    j += 2
                elif text.startswith("*/", j):
                    depth -= 1
                    j += 2
                else:
                    j += 1
            for k in range(i, j):
                if out[k] != "\n":
                    out[k] = " "
            i = j
        elif (
            c == "r"
            and re.match(r'r#*"', text[i : i + 10])
            and (i == 0 or not (text[i - 1].isalnum() or text[i - 1] == "_"))
        ):
            m = re.match(r'r(#*)"', text[i:])
            hashes = m.group(1)
            end = text.find('"' + hashes, i + len(m.group(0)))
            end = n if end < 0 else end + 1 + len(hashes)
            for k in range(i, end):
                if out[k] != "\n":
                    out[k] = " "
            i = end
        elif c == '"':
            j = i + 1
            while j < n and text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            for k in range(i, min(j + 1, n)):
                if out[k] != "\n":
                    out[k] = " "
            i = j + 1
        elif c == "'":
            m = re.match(
                r"'(\\.|\\u\{[0-9a-fA-F]+\}|\\x[0-9a-fA-F]{2}|[^\\'])'", text[i:]
            )
            if m:
                for k in range(i, i + len(m.group(0))):
                    out[k] = " "
                i += len(m.group(0))
            else:
                i += 1
        else:
            i += 1
    return "".join(out)


ITEM_RE = re.compile(
    r"\bpub(?:\((?P<vis>[^)]*)\))?\s+"
    r"(?:(?:const|async|unsafe|extern\s+\S+)\s+)*"
    r"(?P<kind>fn|struct|enum|trait|type|const|static|union|mod|use)\b\s*"
)
MOD_RE = re.compile(r"(?<![\w:])mod\s+(?:r#)?(?P<name>\w+)\s*(?P<term>[;{])")


class Module:
    def __init__(self, crate, path, file, public_chain):
        self.crate = crate
        self.path = path  # list of names from crate root
        self.file = file
        self.public = public_chain
        self.items = {}  # name -> item dict
        self.uses = []  # (tree_text, line)
        self.children = {}  # name -> Module
        self.globs = []


def attrs_before(text, pos):
    """Attribute text immediately preceding pos (stripped source)."""
    j = pos
    chunk = []
    while True:
        k = text.rfind("\n", 0, j - 1)
        line = text[k + 1 : j].strip()
        if line.startswith("#[") or line.startswith("#!["):
            chunk.append(line)
            j = k + 1
            if k < 0:
                break
            continue
        if line == "":
            if k < 0:
                break
            # allow blank (comment-stripped) lines inside attribute runs
            j = k + 1
            prev = text.rfind("\n", 0, k)
            pline = text[prev + 1 : k].strip()
            if pline.startswith("#[") or pline == "":
                continue
            break
        break
    return " ".join(reversed(chunk))


def find_block_end(text, start):
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i
    return len(text)


def depth_map(text):
    d = [0] * (len(text) + 1)
    depth = 0
    for i, c in enumerate(text):
        d[i] = depth
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
    return d


def module_file(parent_file, name, is_root_like):
    base = os.path.dirname(parent_file)
    stem = os.path.splitext(os.path.basename(parent_file))[0]
    if not is_root_like:
        base = os.path.join(base, stem)
    for cand in (os.path.join(base, name + ".rs"), os.path.join(base, name, "mod.rs")):
        if os.path.exists(cand):
            return cand
    return None


def parse_module(mod, text, raw, origin, lo, hi, root_like):
    d = depth_map(text)
    base_depth = d[lo] if lo < len(d) else 0
    i = lo
    while i < hi:
        m = ITEM_RE.search(text, i, hi)
        mm = MOD_RE.search(text, i, hi)
        # handle private `mod x;` too
        cands = [x for x in (m, mm) if x]
        if not cands:
            break
        x = min(cands, key=lambda y: y.start())
        if d[x.start()] != base_depth:
            i = x.end()
            continue
        attrs = attrs_before(text, x.start())
        line = raw.count("\n", 0, x.start()) + 1
        cfg_test = "cfg(test)" in attrs.replace(" ", "")
        if x is mm or (x is m and m.group("kind") == "mod"):
            if x is m:
                mm2 = MOD_RE.search(text, m.start(), hi)
                if mm2 is None:
                    print(
                        "NOMOD",
                        origin,
                        raw.count("\n", 0, m.start()) + 1,
                        repr(text[m.start() : m.start() + 80]),
                        file=sys.stderr,
                    )
                    i = m.end()
                    continue
                name, term, end = mm2.group("name"), mm2.group("term"), mm2.end()
                vis = m.group("vis")
                is_pub = vis is None
            else:
                name, term, end = mm.group("name"), mm.group("term"), mm.end()
                # private unless preceded by pub on same token run
                pre = text[max(0, mm.start() - 20) : mm.start()]
                if re.search(r"\bpub(\([^)]*\))?\s*$", pre):
                    i = mm.end()
                    continue
                is_pub = False
            if cfg_test:
                i = end if term == ";" else find_block_end(text, end - 1) + 1
                continue
            child = Module(mod.crate, mod.path + [name], None, mod.public and is_pub)
            child.cfg = re.findall(r"cfg\(([^\]]*)\)", attrs)
            mod.children[name] = child
            if term == ";":
                f = module_file(origin, name, root_like)
                if f:
                    child.file = f
                    load(child, f)
                i = end
            else:
                close = find_block_end(text, end - 1)
                child.file = origin
                parse_module(child, text, raw, origin, end, close, root_like)
                i = close + 1
            continue
        kind = m.group("kind")
        vis = m.group("vis")
        if vis is not None:  # pub(crate), pub(super), pub(in ...)
            i = m.end()
            continue
        if cfg_test:
            i = m.end()
            continue
        if kind == "use":
            semi = text.find(";", m.end())
            mod.uses.append((text[m.end() : semi], line, attrs))
            i = semi + 1
            continue
        nm = re.match(r"(\w+)", text[m.end() :])
        if not nm:
            i = m.end()
            continue
        name = nm.group(1)
        mod.items[name] = {
            "kind": kind,
            "name": name,
            "file": os.path.relpath(origin, ROOT),
            "line": line,
            "cfg": re.findall(r"cfg\(([^\]]*)\)", attrs),
            "hidden": "doc(hidden)" in attrs.replace(" ", ""),
            "non_exhaustive": "non_exhaustive" in attrs,
        }
        # skip the item body so nested pub items (fields) are not taken
        j = m.end()
        if kind in ("fn",):
            # find body start or ;
            k = j
            depth = 0
            while k < hi:
                if text[k] == "{" and depth == 0:
                    k = find_block_end(text, k)
                    break
                if text[k] == ";" and depth == 0:
                    break
                if text[k] in "(<[":
                    depth += 1
                elif text[k] in ")>]":
                    depth = max(0, depth - 1)
                k += 1
            i = k + 1
        else:
            i = j
    # macro_export macros
    for mx in re.finditer(r"macro_rules!\s*(\w+)", text[lo:hi]):
        pos = lo + mx.start()
        if "macro_export" in attrs_before(text, pos):
            mod.items[mx.group(1)] = {
                "kind": "macro",
                "name": mx.group(1),
                "file": os.path.relpath(origin, ROOT),
                "line": raw.count("\n", 0, pos) + 1,
                "cfg": [],
                "hidden": False,
                "non_exhaustive": False,
            }


def load(mod, file):
    raw = open(file).read()
    text = strip(raw)
    root_like = os.path.basename(file) in ("lib.rs", "mod.rs", "main.rs")
    parse_module(mod, text, raw, file, 0, len(text), root_like)
    mod.text = text
    mod.raw = raw


def parse_use_tree_spaced(s):
    """Expand a use tree into (path segments, alias) pairs; `*` stays a segment."""
    s = s.strip()
    out = []

    def split_top(inner):
        parts, depth, cur = [], 0, ""
        for ch in inner:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            if ch == "," and depth == 0:
                parts.append(cur)
                cur = ""
            else:
                cur += ch
        if cur.strip():
            parts.append(cur)
        return parts

    def rec(prefix, t):
        t = t.strip()
        if not t:
            return
        if t.startswith("{"):
            for p in split_top(t[1 : t.rfind("}")]):
                rec(prefix, p)
            return
        k = t.find("::{")
        if k >= 0:
            rec(prefix + [x.strip() for x in t[:k].split("::")], t[k + 2 :])
            return
        alias = None
        am = re.match(r"^(.*?)\s+as\s+(\w+)$", t, re.S)
        if am:
            t, alias = am.group(1).strip(), am.group(2)
        segs = [x.strip().replace("r#", "") for x in t.split("::")]
        out.append(([p.replace("r#", "") for p in prefix] + segs, alias))

    rec([], s)
    return out


def main():
    roots = {}
    for crate, lib in CRATES.items():
        root = Module(crate, [], lib, True)
        load(root, lib)
        roots[crate] = root

    def find_mod(crate, path):
        m = roots[crate]
        for p in path:
            if p not in m.children:
                return None
            m = m.children[p]
        return m

    def resolve(mod, segs, depth=0):
        """Resolve a use path from module `mod`.

        Returns ("item", module, name), ("mod", module), ("glob", module),
        ("reexport", module, name) or None.
        """
        if depth > 30:
            return None
        crate = mod.crate
        segs = list(segs)
        if segs[0] == "crate":
            cur, segs = roots[crate], segs[1:]
        elif segs[0] == "self":
            cur, segs = mod, segs[1:]
        elif segs[0] == "super":
            cur = mod
            while segs and segs[0] == "super":
                cur = find_mod(crate, cur.path[:-1])
                segs = segs[1:]
        elif segs[0] in ("sidereon_core",) and crate != "sidereon_core":
            cur, segs = roots["sidereon_core"], segs[1:]
        elif segs[0] in ("std", "core", "alloc", "serde", "nalgebra"):
            return None
        else:
            cur = mod
        # walk
        for idx, s in enumerate(segs):
            last = idx == len(segs) - 1
            if s == "*":
                return ("glob", cur)
            if s in cur.children:
                if last:
                    return ("mod", cur.children[s])
                cur = cur.children[s]
                continue
            if last:
                if s in cur.items:
                    return ("item", cur, s)
                # maybe re-exported via use in cur
                return ("reexport", cur, s)
            # a non-module segment followed by more: enum variant / assoc item
            if s in cur.items:
                return ("item", cur, s)
            # could be a module brought in by `use`
            tgt = lookup_use(cur, s, depth + 1)
            if tgt and tgt[0] == "mod":
                cur = tgt[1]
                continue
            return None
        return ("mod", cur)

    def lookup_use(mod, name, depth):
        for tree, _line, _attrs in mod.uses:
            for segs, alias in parse_use_tree_spaced(tree):
                nm = alias or segs[-1]
                if nm == name and segs[-1] != "*":
                    return resolve(mod, segs, depth)
        return None

    def item_key(m, name):
        return (m.crate, tuple(m.path), name)

    def export_names(mod, depth=0, seen=None):
        """Names a path through `mod` reaches: name -> ("item", key, item) or
        ("mod", Module)."""
        if seen is None:
            seen = set()
        if id(mod) in seen or depth > 20:
            return {}
        seen = seen | {id(mod)}
        out = {}
        for name, it in mod.items.items():
            out[name] = ("item", item_key(mod, name), it)
        for name, ch in mod.children.items():
            if ch.public:
                out[name] = ("mod", ch)
        for tree, _line, attrs in mod.uses:
            if "cfg(test)" in attrs.replace(" ", ""):
                continue
            for segs, alias in parse_use_tree_spaced(tree):
                r = resolve(mod, segs)
                if r is None:
                    continue
                if r[0] == "glob":
                    for nm, v in export_names(r[1], depth + 1, seen).items():
                        out.setdefault(nm, v)
                    continue
                nm = alias or segs[-1]
                if r[0] == "item":
                    out[nm] = ("item", item_key(r[1], r[2]), r[1].items[r[2]])
                elif r[0] == "mod":
                    out[nm] = ("mod", r[1])
                elif r[0] == "reexport":
                    sub = export_names(r[1], depth + 1, seen)
                    if r[2] in sub:
                        out[nm] = sub[r[2]]
        return out

    results = {}

    def walk(mod, prefix, depth=0, seen=None):
        if seen is None:
            seen = set()
        if (id(mod), tuple(prefix)) in seen or depth > 12:
            return
        seen.add((id(mod), tuple(prefix)))
        for nm, v in export_names(mod).items():
            if v[0] == "item":
                key = v[1]
                entry = results.setdefault(key, {"item": v[2], "paths": set()})
                entry["paths"].add("::".join(prefix + [nm]))
            else:
                walk(v[1], prefix + [nm], depth + 1, seen)

    for crate, root in roots.items():
        walk(root, [crate])

    # methods of reachable types: scan every file for `impl ... Type` blocks
    type_keys = {
        k: v
        for k, v in results.items()
        if v["item"]["kind"] in ("struct", "enum", "trait", "union", "type")
    }
    by_name = {}
    for k in type_keys:
        by_name.setdefault(k[2], []).append(k)
    methods = {}
    for dirpath, _, files in os.walk(ROOT):
        for f in files:
            if not f.endswith(".rs"):
                continue
            path = os.path.join(dirpath, f)
            raw = open(path).read()
            text = strip(raw)
            for im in re.finditer(
                r"\bimpl\b(?:\s*<[^{]*?>)?\s+(?P<body>[^{;]*?)\{", text
            ):
                header = im.group("body")
                if " for " in header:
                    continue  # trait impls: covered by the trait
                tm = re.search(
                    r"(\w+)\s*(?:<[^{]*>)?\s*(?:where[^{]*)?$", header.strip()
                )
                if not tm:
                    continue
                tname = tm.group(1)
                if tname not in by_name:
                    continue
                attrs = attrs_before(text, im.start())
                if "cfg(test)" in attrs.replace(" ", ""):
                    continue
                start = im.end() - 1
                end = find_block_end(text, start)
                block = text[start + 1 : end]
                dm = depth_map(block)
                for fm in re.finditer(
                    r"\bpub(?P<vis>\([^)]*\))?\s+(?:(?:const|async|unsafe)\s+)*fn\s+(\w+)",
                    block,
                ):
                    if fm.group("vis") or dm[fm.start()] != 0:
                        continue
                    fattrs = attrs_before(block, fm.start())
                    if "cfg(test)" in fattrs.replace(" ", ""):
                        continue
                    for key in by_name[tname]:
                        methods.setdefault(key, set()).add(fm.group(2))
    out = []
    for key, v in sorted(
        results.items(),
        key=lambda kv: (kv[0][0], min(kv[1]["paths"], key=lambda p: (len(p), p))),
    ):
        it = v["item"]
        out.append(
            {
                "crate": key[0],
                "defined_in": "::".join([key[0]] + list(key[1])),
                "name": key[2],
                "kind": it["kind"],
                "paths": sorted(v["paths"], key=lambda p: (len(p), p)),
                "file": it["file"],
                "line": it["line"],
                "cfg": it["cfg"],
                "hidden": it["hidden"],
                "non_exhaustive": it.get("non_exhaustive", False),
                "methods": sorted(methods.get(key, [])),
            }
        )
    json.dump(out, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
