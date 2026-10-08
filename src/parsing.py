def parse_yes_no(text: str) -> int:
    """Extract 0/1 from model response. Returns -1 if unparseable."""
    if text is None:
        return -1
    t = str(text).strip().upper()
    if not t:
        return -1

    has_yes = "YES" in t
    has_no = "NO" in t

    # Ambiguous:
    if has_yes and has_no:
        return -1

    if t.startswith("YES") or (has_yes and not has_no):
        return 1
    if t.startswith("NO") or (has_no and not has_yes):
        return 0

    return -1
