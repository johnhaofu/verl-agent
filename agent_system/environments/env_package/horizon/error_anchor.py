"""ESSA — Error-State Step Anchor for Horizon.

Replaces GiGPO's default observation-string anchor (which never repeats
across rollouts in JSON-parameterized state spaces) with an anchor based on
the *canonical error fingerprint* of the validator's response.

The Horizon task is fundamentally an error-driven loop:

    propose JSON -> validator returns errors -> fix -> validator returns
    errors -> ... -> submit pass

Two trajectories that wrote totally different JSON but face the *same
outstanding error* are at the same problem-instance, regardless of their
JSON content. ESSA captures this by stripping entity names (quoted/backticked
identifiers, numbers) from validator error messages and keeping the structural
template.

Examples:

    "Section type 'featured-collection' does not refer to an existing section file"
    "Section type 'cta' does not refer to an existing section file"
    "Section type 'team-members' does not refer to an existing section file"
        -> all canonicalize to:
           "Section type '<X>' does not refer to an existing section file"

    "JSON Error: Expecting ',' delimiter: line 1 column 645 (char 644)"
        -> "JSON Error: Expecting '<X>' delimiter: line <N> column <N> (char <N>)"

This yields a finite-vocabulary error code set (~30-50 codes empirically),
producing meaningful inner-group clusters for GiGPO's step-level advantage.
"""

from __future__ import annotations

import re
from typing import Tuple


_QUOTED_RE = re.compile(r"'[^']*'")
_BACKTICK_RE = re.compile(r"`[^`]*`")
_DOUBLEQUOTE_RE = re.compile(r'"[^"]*"')
_NUMBER_RE = re.compile(r"\b\d+\b")
_WHITESPACE_RE = re.compile(r"\s+")


def canonicalize_error(error_msg: str) -> str:
    """Strip variable identifiers and numbers from a validator error message.

    Intended to map errors-with-different-entities into the same equivalence
    class. The remaining tokens are the structural template.
    """
    if not error_msg:
        return ""
    s = error_msg.strip()
    s = _QUOTED_RE.sub("'<X>'", s)
    s = _BACKTICK_RE.sub("`<X>`", s)
    s = _DOUBLEQUOTE_RE.sub('"<X>"', s)
    s = _NUMBER_RE.sub("<N>", s)
    s = _WHITESPACE_RE.sub(" ", s).strip()
    return s


def essa_anchor(
    verb: str,
    arg: str,
    won: bool,
    invalid: bool,
    error_message: str,
) -> Tuple[str, ...]:
    """Compute the ESSA anchor for a step.

    Args:
        verb: parsed action verb (list_sections / describe_section /
              describe_block / fix / submit / "" for invalid).
        arg: action argument; only used as anchor key for read actions.
        won: True iff this step terminated with success (submit pass).
        invalid: True iff the model's output failed to parse.
        error_message: validator's primary error string (empty if no
            validation occurred or it passed).

    Returns:
        A hashable tuple. Trajectories whose step produces equal anchors
        are treated as having reached the same anchor state for GiGPO.
    """
    if invalid:
        return ("invalid",)

    if verb == "list_sections":
        # All list calls share state (deterministic listing).
        return ("read", "list")

    if verb in ("describe_section", "describe_block"):
        # State partitioned by which schema was read.
        return ("read", verb, arg.strip().lstrip("_"))

    if verb in ("fix", "submit"):
        if won:
            return ("ok", verb)
        canon = canonicalize_error(error_message)
        if not canon:
            return ("write", verb, "unknown")
        return ("write", verb, canon)

    return ("unknown", verb)
