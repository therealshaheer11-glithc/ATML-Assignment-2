"""Inspect installed requirement chains, including markers and requested extras.

This module reads distribution metadata; it does not install or import models.
"""
from collections import deque
from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name


def inspect_chain(requirement_lines, lookup, environment=None):
    environment = dict(environment or default_environment())
    nodes, active_extras, processed, edges, issues = {}, {}, {}, {}, {}
    pending = deque()

    def issue(kind, **details):
        value = {"kind": kind, **details}
        issues[repr(sorted(value.items()))] = value

    def active(requirement, extras):
        return requirement.marker is None or any(
            requirement.marker.evaluate({**environment, "extra": extra})
            for extra in {"", *extras}
        )

    def add(raw, parent=None):
        requirement = Requirement(raw)
        name = canonicalize_name(requirement.name)
        edge = {"parent": parent, "dependency": name, "requirement": raw}
        edges[(parent, raw)] = edge
        if name not in nodes:
            path = (nodes[parent]["first_path"] if parent else []) + [name]
            record = lookup(name)
            nodes[name] = {"first_path": path, "installed": record is not None,
                           **(record or {}), "requested_extras": []}
            active_extras[name] = set()
            pending.append(name)
        record = nodes[name]
        if not record["installed"]:
            issue("missing_distribution", **edge)
            return
        if canonicalize_name(record["name"]) != name:
            issue("metadata_name_mismatch", **edge, actual=record["name"])
        if not requirement.specifier.contains(record["version"], prereleases=True):
            issue("incompatible_version", **edge, actual=record["version"])
        if requirement.url:
            issue("direct_url_identity_not_validated", **edge)
        requested = {canonicalize_name(extra) for extra in requirement.extras}
        provided = {canonicalize_name(extra) for extra in record.get("provides_extra", [])}
        for extra in sorted(requested - provided):
            issue("unprovided_extra", **edge, extra=extra)
        before = set(active_extras[name])
        active_extras[name].update(requested)
        record["requested_extras"] = sorted(active_extras[name])
        if active_extras[name] != before:
            pending.append(name)

    roots = []
    for line in requirement_lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        requirement = Requirement(line)
        if active(requirement, set()):
            roots.append(line)
            add(line)
    while pending:
        name = pending.popleft()
        node = nodes[name]
        extras = frozenset(active_extras[name])
        if not node["installed"] or processed.get(name) == extras:
            continue
        processed[name] = extras
        python_bound = node.get("requires_python")
        if python_bound and not SpecifierSet(python_bound).contains(
                environment["python_full_version"], prereleases=True):
            issue("incompatible_python", distribution=name, required=python_bound,
                  actual=environment["python_full_version"])
        for raw in node.get("requires_dist", []):
            requirement = Requirement(raw)
            if active(requirement, extras):
                add(raw, parent=name)
    return {"root_requirements": roots, "marker_environment": environment,
            "nodes": dict(sorted(nodes.items())),
            "active_edges": sorted(edges.values(), key=lambda x: (x["parent"] or "", x["requirement"])),
            "issues": list(issues.values()), "required_chain_passed": not issues}
