"""Map the enumerated core public API onto the Python binding.

Reads pubapi.json (from pubapi.py) and the binding's src/*.rs and the package
stub, and for each core item finds:
- the binding source files that refer to it (by an import from sidereon or
  sidereon_core, or by a qualified path);
- the Python symbols that expose it: a pyclass whose `inner` field (or From
  impl) holds the core type, or a pyfunction/pymethod whose body calls the
  core function.
Writes bindmap.json: one record per core item with `refs` and `symbols`.

    python3 scripts/coverage/bindmap.py <binding root> pubapi.json bindmap.json
"""

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pubapi import find_block_end, parse_use_tree_spaced, strip  # noqa: E402

LANE = sys.argv[1]
API = json.load(open(sys.argv[2]))

# public path (without crate for facade) -> item index
by_path = {}
for idx, it in enumerate(API):
    for p in it["paths"]:
        by_path[p] = idx
        if p.startswith("sidereon::"):
            by_path["sidereon_core::" + p[len("sidereon::") :]] = by_path.get(
                "sidereon_core::" + p[len("sidereon::") :], idx
            )


def resolve_path(segs):
    """Resolve a use path (list of segments) to an item index, or a module path."""
    if not segs:
        return None
    root = segs[0]
    if root not in ("sidereon_core", "sidereon"):
        return None
    full = "::".join(segs)
    if full in by_path:
        return by_path[full]
    # facade re-exports core modules: sidereon::x::Y == sidereon_core::x::Y
    if root == "sidereon":
        alt = "sidereon_core::" + "::".join(segs[1:])
        if alt in by_path:
            return by_path[alt]
    # enum variant or associated item: resolve the parent
    parent = "::".join(segs[:-1])
    if parent in by_path:
        return by_path[parent]
    return None


src_dir = os.path.join(LANE, "src")
refs = {}  # item idx -> set(files)
HELPERS = {}  # helper fn name -> set(item idx)
HELPER_FILES = {}
symbols = {}  # item idx -> set(python symbols)


def add(idx, sym=None, f=None):
    if idx is None:
        return
    if f:
        refs.setdefault(idx, set()).add(f)
    if sym:
        symbols.setdefault(idx, set()).add(sym)


GLOBAL_PYCLASS = {}
for fname in sorted(os.listdir(src_dir)):
    if not fname.endswith(".rs"):
        continue
    raw0 = open(os.path.join(src_dir, fname)).read()
    for pm in re.finditer(r"#\[pyclass\(([^\]]*)\)\]", raw0):
        nm = re.search(r'name\s*=\s*"(\w+)"', pm.group(1))
        sm = re.search(r"\b(?:struct|enum)\s+(\w+)", raw0[pm.end() : pm.end() + 400])
        if sm:
            GLOBAL_PYCLASS[sm.group(1)] = nm.group(1) if nm else sm.group(1)
TYPED_METHOD = {}  # (item idx, method) -> set(py symbols)

for fname in sorted(os.listdir(src_dir)):
    if not fname.endswith(".rs"):
        continue
    raw = open(os.path.join(src_dir, fname)).read()
    text = strip(raw)
    local = {}  # local name -> item idx ; module alias -> path segs
    mods = {}
    for um in re.finditer(r"\buse\s+([^;]+);", text):
        for segs, alias in parse_use_tree_spaced(um.group(1)):
            if not segs or segs[0] not in ("sidereon_core", "sidereon"):
                continue
            if segs[-1] == "self":
                segs = segs[:-1]
            nm = alias or segs[-1]
            if segs[-1] == "*":
                continue
            idx = resolve_path(segs)
            if idx is not None and API[idx]["name"] == segs[-1]:
                local[nm] = idx
                add(idx, f=fname)
            else:
                mods[nm] = segs
    # qualified paths in code
    for qm in re.finditer(r"\b((?:sidereon_core|sidereon)(?:::\w+)+)", text):
        segs = qm.group(1).split("::")
        for k in range(len(segs), 1, -1):
            idx = resolve_path(segs[:k])
            if idx is not None:
                add(idx, f=fname)
                break
    # module-alias paths: alias::Name
    for nm, segs in mods.items():
        for qm in re.finditer(r"(?<![\w:])" + re.escape(nm) + r"((?:::\w+)+)", text):
            tail = qm.group(1).split("::")[1:]
            for k in range(len(tail), 0, -1):
                idx = resolve_path(segs + tail[:k])
                if idx is not None:
                    add(idx, f=fname)
                    break

    def refs_in(block, local=local, mods=mods):
        found = set()
        for tok in set(re.findall(r"\b[A-Za-z_]\w*\b", block)):
            if tok in local:
                found.add(local[tok])
        for qm in re.finditer(r"\b((?:sidereon_core|sidereon)(?:::\w+)+)", block):
            segs = qm.group(1).split("::")
            for k in range(len(segs), 1, -1):
                idx = resolve_path(segs[:k])
                if idx is not None:
                    found.add(idx)
                    break
        for nm, segs in mods.items():
            for qm in re.finditer(
                r"(?<![\w:])" + re.escape(nm) + r"((?:::\w+)+)", block
            ):
                tail = qm.group(1).split("::")[1:]
                for k in range(len(tail), 0, -1):
                    idx = resolve_path(segs + tail[:k])
                    if idx is not None:
                        found.add(idx)
                        break
        return found

    # pyclasses
    pyname_of_struct = dict(GLOBAL_PYCLASS)
    for pm in re.finditer(r"#\[pyclass\(([^\]]*)\)\]", raw):
        attrs = pm.group(1)
        nm = re.search(r'name\s*=\s*"(\w+)"', attrs)
        after = text[pm.end() : pm.end() + 400]
        sm = re.search(r"\b(?:struct|enum)\s+(\w+)", after)
        if not sm:
            continue
        rust = sm.group(1)
        py = nm.group(1) if nm else rust
        pyname_of_struct[rust] = py
        # body of struct: field types
        body_start = pm.end() + sm.end()
        brace = text.find("{", body_start)
        semi = text.find(";", body_start)
        if brace >= 0 and (semi < 0 or brace < semi):
            end = find_block_end(text, brace)
            for idx in refs_in(text[brace:end]):
                add(idx, sym=py, f=fname)
        elif semi >= 0:
            for idx in refs_in(text[body_start:semi]):
                add(idx, sym=py, f=fname)
    # From/Into impls between a pyclass and a core type
    for im in re.finditer(r"\bimpl\s+From<([^>]+)>\s+for\s+(\w+)", text):
        a, b = im.group(1), im.group(2)
        if b in pyname_of_struct:
            for idx in refs_in(a):
                add(idx, sym=pyname_of_struct[b], f=fname)
        else:
            m2 = re.search(r"(\w+)\s*$", a.strip())
            if m2 and m2.group(1) in pyname_of_struct:
                for idx in refs_in(b):
                    add(idx, sym=pyname_of_struct[m2.group(1)], f=fname)
    # pymethods blocks
    for im in re.finditer(r"#\[pymethods\]\s*impl\s+(\w+)\s*\{", text):
        cls = pyname_of_struct.get(im.group(1), im.group(1))
        end = find_block_end(text, im.end() - 1)
        block = text[im.end() : end]
        for fm in re.finditer(r"\bfn\s+(\w+)\s*(?:<[^>]*>)?\s*\(", block):
            start = fm.start()
            body_open = block.find("{", fm.end())
            if body_open < 0:
                continue
            body_end = find_block_end(block, body_open)
            pre = block[max(0, start - 300) : start]
            nm = re.findall(
                r'name\s*=\s*"(\w+)"',
                pre[pre.rfind("fn ") + 1 if "fn " in pre else 0 :],
            )
            meth = nm[-1] if nm else fm.group(1)
            body = block[body_open:body_end]
            sym = f"{cls}.{meth}"
            for idx in refs_in(block[fm.start() : body_end]):
                add(idx, sym=sym, f=fname)
            # method calls on inner: record method names for method coverage
            for cm in re.finditer(r"\.(\w+)\s*\(", body):
                symbols.setdefault(("call", cm.group(1)), set()).add(sym)
            for cm in re.finditer(r"(?<![\w.:])(\w+)\s*\(", body):
                symbols.setdefault(("fcall", cm.group(1)), set()).add(sym)
            for cm in re.finditer(r"\b(\w+)::(\w+)\s*[(<]", body):
                if cm.group(1) in local:
                    TYPED_METHOD.setdefault(
                        (local[cm.group(1)], cm.group(2)), set()
                    ).add(sym)
    # pyfunctions
    for fm in re.finditer(r"#\[pyfunction\]", text):
        rest = text[fm.end() :]
        sm = re.search(r"\bfn\s+(\w+)", rest)
        if not sm:
            continue
        pre = rest[: sm.start()]
        nm = re.search(r'name\s*=\s*"(\w+)"', pre)
        py = nm.group(1) if nm else sm.group(1)
        body_open = rest.find("{", sm.end())
        body_end = find_block_end(rest, body_open)
        for idx in refs_in(rest[sm.start() : body_end]):
            add(idx, sym=py, f=fname)
        for cm in re.finditer(r"\.(\w+)\s*\(", rest[body_open:body_end]):
            symbols.setdefault(("call", cm.group(1)), set()).add(py)
        for cm in re.finditer(r"(?<![\w.:])(\w+)\s*\(", rest[body_open:body_end]):
            symbols.setdefault(("fcall", cm.group(1)), set()).add(py)
        for cm in re.finditer(r"\b(\w+)::(\w+)\s*[(<]", rest[body_open:body_end]):
            if cm.group(1) in local:
                TYPED_METHOD.setdefault((local[cm.group(1)], cm.group(2)), set()).add(
                    py
                )
    # helper fns (non-py): record their refs; py symbols that call them inherit
    for hm in re.finditer(r"(?m)^(?:pub(?:\([^)]*\))?\s+)?fn\s+(\w+)", text):
        pre = text[max(0, hm.start() - 200) : hm.start()]
        if "#[pyfunction]" in pre[pre.rfind("}") + 1 if "}" in pre else 0 :]:
            continue
        body_open = text.find("{", hm.end())
        if body_open < 0:
            continue
        body_end = find_block_end(text, body_open)
        HELPERS.setdefault(hm.group(1), set()).update(
            refs_in(text[hm.start() : body_end])
        )
        HELPER_FILES.setdefault(hm.group(1), set()).add(fname)

# a py symbol that calls a helper fn inherits the items the helper refers to
for h, idxs in HELPERS.items():
    for idx in idxs:
        for sym in symbols.get(("fcall", h), set()):
            add(idx, sym=sym)

out = []
for idx, it in enumerate(API):
    rec = dict(it)
    rec["refs"] = sorted(refs.get(idx, []))
    rec["symbols"] = sorted(symbols.get(idx, []))
    rec["method_calls"] = {
        m: sorted(TYPED_METHOD.get((idx, m), set()) | symbols.get(("call", m), set()))[
            :6
        ]
        for m in it["methods"]
    }
    out.append(rec)
json.dump(out, open(sys.argv[3], "w"), indent=1)
