"""Appearance authority shared by single-clip and segmented H3 swap prompts."""

from .skills import load_skill

APPEARANCE_AUTHORITY = (
    'For each replacement target, the assigned reference picture supplies the complete replacement '
    'appearance: facial identity, hair, ears, outfit, body proportions and distinguishing features. '
    'The source video supplies performance, expressions, position, visibility, interactions, camera, '
    'lighting, environment and timing. Source appearance descriptions identify and track the original '
    'person only; they do not prescribe the replacement appearance. Keep source people and replacement '
    'identities distinct when describing observations. Transfer the observed performance to the '
    'replacement without copying the original face, hair or clothing. Explicit user requests to retain '
    'a particular appearance attribute take precedence over this default; do not infer such exceptions '
    'from a request to preserve motion, camera or background.'
)


def swap_writing_guide():
    """Load the same character-swap guide for both cloud entry points."""
    body, references = load_skill('h3-prompt-writing', 'Ref2VA')
    _, swap_references = load_skill('h3-prompt-writing', 'CharacterSwap')
    return body, references + swap_references
