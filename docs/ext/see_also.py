"""Griffe extension: render NumPy "See Also" sections as cross-linked lists.

Griffe has no parser for "See Also", so the section becomes an admonition whose
``name : description`` lines are joined into one paragraph. This extension rewrites
the section body into a Markdown list and turns each name into an autorefs
cross-reference, so the docstrings in ``src/`` stay plain numpydoc.

Names are resolved like Python names in the docstring's scope: members of the
object itself (for classes and modules), then of each enclosing object, then as
an absolute path, then as a unique public name anywhere in the package. Names
under another top-level package (``pandas.concat``) are rendered as plain code;
any other unresolved name is logged as a warning, which fails ``mkdocs build
--strict``.
"""

from __future__ import annotations

import importlib.util
import re
from collections import defaultdict
from typing import Any

import griffe

logger = griffe.get_logger(__name__)

_HEADER = re.compile(r"^see also$", re.IGNORECASE)
_DASHES = re.compile(r"^-{3,}\s*$")
_ENTRY = re.compile(r"^(?P<names>[^\s:][^:]*?)(?:\s*:\s*(?P<desc>.*))?$")


class SeeAlsoLinks(griffe.Extension):
    def on_package(self, *, pkg: griffe.Module, loader: griffe.GriffeLoader, **kwargs: Any) -> None:
        index = _index_by_name(pkg)
        for obj in _walk(pkg):
            if obj.docstring is not None:
                obj.docstring.value = _rewrite(obj.docstring.value, obj, loader, index)


def _walk(obj: griffe.Object):
    yield obj
    for member in obj.members.values():
        if not member.is_alias:
            yield from _walk(member)


def _index_by_name(pkg: griffe.Module) -> dict[str, list[griffe.Object]]:
    index = defaultdict(list)
    for obj in _walk(pkg):
        if not obj.name.startswith("_"):
            index[obj.name].append(obj)
    return index


def _rewrite(text: str, obj: griffe.Object, loader: griffe.GriffeLoader, index) -> str:
    lines = text.split("\n")
    for start in range(len(lines) - 1):
        if _HEADER.match(lines[start].strip()) and _DASHES.match(lines[start + 1]):
            break
    else:
        return text

    # Body runs until the next "Header\n-----" pair or the end
    body_start = start + 2
    end = body_start
    while end < len(lines):
        if end + 1 < len(lines) and lines[end].strip() and _DASHES.match(lines[end + 1]):
            break
        end += 1
    while end > body_start and not lines[end - 1].strip():
        end -= 1

    # Group "name : desc" lines with their indented continuation lines
    entries: list[tuple[str, str]] = []
    for line in lines[body_start:end]:
        if not line.strip():
            continue
        if line[0].isspace() and entries:
            names, desc = entries[-1]
            entries[-1] = (names, f"{desc} {line.strip()}".strip())
            continue
        match = _ENTRY.match(line.strip())
        entries.append((match["names"].strip(), (match["desc"] or "").strip()))

    items = []
    for names, desc in entries:
        refs = ", ".join(_link(name.strip(), obj, loader, index) for name in names.split(","))
        items.append(f"- {refs}: {desc}" if desc else f"- {refs}")

    return "\n".join(lines[:body_start] + items + lines[end:])


def _link(name: str, obj: griffe.Object, loader: griffe.GriffeLoader, index) -> str:
    target = _resolve(name, obj, loader, index)
    if target is not None:
        return f"[`{name}`][{target}]"
    top = name.split(".", 1)[0]
    if top != obj.package.name and importlib.util.find_spec(top) is not None:
        return f"`{name}`"
    logger.warning("%s: cannot resolve See Also entry %r", obj.path, name)
    return f"`{name}`"


def _resolve(name: str, obj: griffe.Object, loader: griffe.GriffeLoader, index) -> str | None:
    first, *rest = name.split(".")

    # Scope lookup: the object itself, then enclosing objects
    scope = obj if obj.is_class or obj.is_module else obj.parent
    while scope is not None:
        found = _get(scope, first)
        for part in rest:
            found = _get(found, part) if found is not None else None
        if found is not None:
            return found.path
        scope = scope.parent

    # Absolute path
    try:
        return _final(loader.modules_collection[name]).path
    except (KeyError, ValueError, griffe.AliasResolutionError, griffe.CyclicAliasError):
        pass

    # Unique public name in the package
    if not rest and len(index.get(first, [])) == 1:
        return index[first][0].path
    return None


def _get(scope: griffe.Object, name: str) -> griffe.Object | None:
    members = scope.all_members if scope.is_class else scope.members
    member = members.get(name)
    if member is None:
        return None
    try:
        return _final(member)
    except (griffe.AliasResolutionError, griffe.CyclicAliasError):
        return None


def _final(obj):
    return obj.final_target if obj.is_alias else obj
