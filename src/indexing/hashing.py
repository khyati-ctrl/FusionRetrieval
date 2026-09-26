import hashlib


def normalize_code(code):
    """
    Normalize code before hashing so insignificant
    formatting differences do not create unnecessary changes.
    """

    lines = code.replace("\r\n", "\n").replace("\r", "\n").splitlines()

    normalized_lines = [
        line.rstrip()
        for line in lines
    ]

    return "\n".join(normalized_lines).strip()


def calculate_content_hash(code):
    """
    Generate a SHA-256 content hash.
    """

    normalized = normalize_code(code)

    return hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()