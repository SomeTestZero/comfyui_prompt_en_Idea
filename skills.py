import os

NODE_DIR = os.path.dirname(os.path.abspath(__file__))
SKILLS_DIR = os.path.join(NODE_DIR, "skills")

MODES = ["Auto", "T2VA", "I2VA", "FL2VA", "L2VA", "Ref2VA"]
BASE_MODES = ("T2VA", "I2VA", "FL2VA", "L2VA")


def scan_skills():
    """Skill folders under skills/ containing a SKILL.md."""
    names = []
    if os.path.isdir(SKILLS_DIR):
        for name in sorted(os.listdir(SKILLS_DIR)):
            if os.path.isfile(os.path.join(SKILLS_DIR, name, "SKILL.md")):
                names.append(name)
    return names or ["h3-prompt-writing"]


def _safe_path(root, rel):
    path = os.path.abspath(os.path.join(root, rel))
    if not path.startswith(os.path.abspath(root) + os.sep):
        raise FileNotFoundError(f"Path escapes skills folder: {rel}")
    return path


def load_skill(name, mode):
    """Return (skill_body, [(ref_name, ref_text), ...]) for the given generation mode.

    Follows the h3-prompt-writing routing rule: base modes read base-en.txt,
    Ref2VA reads ref-en.txt. Skills with a different references layout get all
    their reference files.
    """
    skill_dir = _safe_path(SKILLS_DIR, name)
    skill_path = os.path.join(skill_dir, "SKILL.md")
    if not os.path.isfile(skill_path):
        raise FileNotFoundError(f"Skill not found: {name} (expected {skill_path})")
    with open(skill_path, "r", encoding="utf-8") as f:
        body = f.read()

    refs = []
    refs_dir = os.path.join(skill_dir, "references")
    if os.path.isdir(refs_dir):
        base = os.path.join(refs_dir, "base-en.txt")
        ref = os.path.join(refs_dir, "ref-en.txt")
        if mode in BASE_MODES and os.path.isfile(base):
            chosen = [base]
        elif mode == "Ref2VA" and os.path.isfile(ref):
            chosen = [ref]
        else:
            chosen = [
                os.path.join(refs_dir, n)
                for n in sorted(os.listdir(refs_dir))
                if n.lower().endswith((".txt", ".md"))
            ]
        for path in chosen:
            with open(path, "r", encoding="utf-8") as f:
                refs.append((os.path.relpath(path, skill_dir), f.read()))
    return body, refs
