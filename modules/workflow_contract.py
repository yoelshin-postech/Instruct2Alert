"""Deterministic instruction and workflow-contract checks.

The planner remains responsible for semantic compilation, but the executor
accepts only a typed, executable workflow.  These checks deliberately reject
malformed or explicitly unsupported requests rather than silently inventing a
default rule.
"""

from __future__ import annotations

import math
import re
from typing import Any


ALLOWED_MODULES = {"Pose", "Depth"}
REQUIRED_RULE_FIELDS = {
    "target_pair", "condition", "parameters", "verification_prompt", "od_hit_supplement",
}
REQUIRED_PROMPT_FIELDS = {"system", "instruction", "query"}
_DURATION_CLAUSE = re.compile(
    r"\b(?:for\s+)?(?:more\s+than|at\s+least)\s+\d+(?:\.\d+)?\s*(?:seconds?|secs?|s)\b",
    re.IGNORECASE,
)


def validate_instruction_language(query: str) -> list[dict[str, str]]:
    """Return deterministic rejections for instruction forms not modeled here.

    Supported requests may use conjunction, disjunction, one negation/count
    condition, multiple named entities, qualitative spatial relations, and one
    duration clause per alert rule. Nested Boolean expressions and multiple
    independent duration clauses are rejected because the current workflow
    schema has no unambiguous binding syntax for them.
    """
    issues = []
    text = str(query or "").strip()
    if not text:
        return [{"code": "empty_instruction", "message": "Scenario query is empty."}]
    if "(" in text or ")" in text:
        issues.append({
            "code": "nested_boolean_expression",
            "message": "Parenthesized/nested Boolean expressions are unsupported.",
        })
    has_and = bool(re.search(r"\band\b", text, flags=re.IGNORECASE))
    has_or = bool(re.search(r"\bor\b", text, flags=re.IGNORECASE))
    if has_and and has_or:
        issues.append({
            "code": "ambiguous_mixed_boolean_operators",
            "message": "Mixed and/or expressions require explicit separate scenario instructions.",
        })
    if len(_DURATION_CLAUSE.findall(text)) > 1:
        issues.append({
            "code": "multiple_duration_clauses",
            "message": "Use one duration clause per independently monitored rule.",
        })
    if re.search(r"\b(?:it|they|them|this|that|these|those)\b", text, flags=re.IGNORECASE):
        issues.append({
            "code": "ambiguous_anaphora",
            "message": "Use explicit entity names instead of pronouns or deictic references.",
        })
    return issues


def _is_finite_nonnegative_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def validate_workflow_contract(workflow: Any) -> list[dict[str, str]]:
    """Validate the executable workflow schema."""
    issues: list[dict[str, str]] = []
    if not isinstance(workflow, dict):
        return [{"code": "workflow_not_object", "message": "Planner output must be a JSON object."}]

    for field in ("scenario_name", "targets", "additional_modules", "alert_rules"):
        if field not in workflow:
            issues.append({"code": "missing_field", "message": f"Missing top-level field: {field}."})
    if issues:
        return issues

    if not isinstance(workflow["scenario_name"], str) or not workflow["scenario_name"].strip():
        issues.append({"code": "invalid_scenario_name", "message": "scenario_name must be a nonempty string."})

    targets = workflow["targets"]
    if not isinstance(targets, dict):
        issues.append({"code": "invalid_targets", "message": "targets must be an object."})
        targets = {}
    main_target = targets.get("main")
    other_targets = targets.get("other_targets")
    if not isinstance(main_target, str) or not main_target.strip():
        issues.append({"code": "invalid_main_target", "message": "targets.main must be a nonempty string."})
    if not isinstance(other_targets, list) or not all(isinstance(item, str) and item.strip() for item in other_targets):
        issues.append({"code": "invalid_other_targets", "message": "targets.other_targets must be a list of nonempty strings."})
    elif isinstance(main_target, str) and main_target in other_targets:
        issues.append({"code": "duplicate_main_target", "message": "main target cannot also be an other target."})

    modules = workflow["additional_modules"]
    if not isinstance(modules, list) or not all(isinstance(module, str) for module in modules):
        issues.append({"code": "invalid_modules", "message": "additional_modules must be a string list."})
    elif len(modules) != len(set(modules)) or any(module not in ALLOWED_MODULES for module in modules):
        issues.append({
            "code": "unsupported_module",
            "message": f"additional_modules may contain only {sorted(ALLOWED_MODULES)} without duplicates.",
        })

    rules = workflow["alert_rules"]
    if not isinstance(rules, list) or not rules:
        issues.append({"code": "missing_alert_rule", "message": "alert_rules must contain at least one rule."})
        return issues

    for index, rule in enumerate(rules):
        prefix = f"alert_rules[{index}]"
        if not isinstance(rule, dict):
            issues.append({"code": "invalid_alert_rule", "message": f"{prefix} must be an object."})
            continue
        for field in REQUIRED_RULE_FIELDS - set(rule):
            issues.append({"code": "missing_rule_field", "message": f"{prefix}.{field} is required."})
        target_pair = rule.get("target_pair")
        if not isinstance(target_pair, list) or len(target_pair) != 2:
            issues.append({"code": "invalid_target_pair", "message": f"{prefix}.target_pair must contain two entries."})
        elif not isinstance(target_pair[0], str) or target_pair[1] is not None and not isinstance(target_pair[1], str):
            issues.append({"code": "invalid_target_pair", "message": f"{prefix}.target_pair has invalid entity types."})
        if not isinstance(rule.get("condition"), str) or not rule["condition"].strip():
            issues.append({"code": "invalid_condition", "message": f"{prefix}.condition must be a nonempty string."})

        parameters = rule.get("parameters")
        if not isinstance(parameters, dict):
            issues.append({"code": "invalid_parameters", "message": f"{prefix}.parameters must be an object."})
        else:
            missing_parameters = {"duration_sec", "distance_m"} - set(parameters)
            if missing_parameters:
                issues.append({
                    "code": "missing_parameter",
                    "message": f"{prefix}.parameters is missing {sorted(missing_parameters)}.",
                })
            duration = parameters.get("duration_sec")
            distance = parameters.get("distance_m")
            if not _is_finite_nonnegative_number(duration):
                issues.append({"code": "invalid_duration", "message": f"{prefix}.parameters.duration_sec must be finite and nonnegative."})
            if distance is not None and not _is_finite_nonnegative_number(distance):
                issues.append({"code": "invalid_distance", "message": f"{prefix}.parameters.distance_m must be null or finite and nonnegative."})

        prompt = rule.get("verification_prompt")
        if not isinstance(prompt, dict):
            issues.append({"code": "invalid_verification_prompt", "message": f"{prefix}.verification_prompt must be an object."})
        else:
            missing = REQUIRED_PROMPT_FIELDS - set(prompt)
            if missing or not all(isinstance(prompt.get(field), str) and prompt[field].strip()
                                  for field in REQUIRED_PROMPT_FIELDS):
                issues.append({"code": "invalid_verification_prompt", "message": f"{prefix}.verification_prompt requires nonempty system, instruction, and query."})

        supplement = rule.get("od_hit_supplement")
        if supplement is not None and not isinstance(supplement, str):
            issues.append({"code": "invalid_grounding_hint", "message": f"{prefix}.od_hit_supplement must be a string or null."})
    return issues


def workflow_execution_semantics() -> dict[str, str]:
    """Machine-readable statement of the executor's multi-rule behavior."""
    return {
        "rule_composition": "Each alert rule is evaluated independently on every sampled frame.",
        "temporal_state": "Yes/no streaks and alert state are indexed by rule, not tracker identity.",
        "scenario_pooling": "Scenario-level reporting pools simultaneous rule outputs with any-positive semantics.",
        "conflict_policy": "No priority resolver is applied; simultaneous positive rules remain independently positive.",
    }
