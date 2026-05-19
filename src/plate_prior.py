from __future__ import annotations

import re


INDIAN_STATE_CODES = {
    "AN", "AP", "AR", "AS", "BR", "CH", "CG", "DD", "DL", "DN", "GA", "GJ",
    "HR", "HP", "JH", "JK", "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP",
    "MZ", "NL", "OD", "PB", "PY", "RJ", "SK", "TN", "TR", "TS", "UK", "UP",
    "WB",
}

LETTER_AMBIGUITIES = {
    "0": ["O", "D", "Q"],
    "1": ["I", "L", "T"],
    "2": ["Z"],
    "5": ["S"],
    "6": ["G"],
    "8": ["B"],
    "9": ["G"],
    "M": ["N"],
    "N": ["M"],
    "V": ["Y"],
}

DIGIT_AMBIGUITIES = {
    "O": ["0"],
    "D": ["0"],
    "Q": ["0"],
    "I": ["1"],
    "L": ["1"],
    "T": ["1"],
    "Z": ["2"],
    "S": ["5"],
    "G": ["6", "9"],
    "B": ["8"],
    "A": ["4"],
}


def normalize_plate_with_prior(raw_text: str) -> dict:
    cleaned = re.sub(r"[^A-Z0-9]", "", raw_text.upper())[:12]
    if not cleaned:
        return {
            "text": "",
            "raw_text": "",
            "format_hint": "unknown",
            "corrections": [],
            "confidence": 0.0,
        }

    chars = list(cleaned)
    corrections = []
    apply_indian = should_apply_indian_prior(cleaned)
    if apply_indian:
        expected = expected_roles(len(chars))
        for idx, role in enumerate(expected):
            original = chars[idx]
            if role == "letter":
                chars[idx] = coerce_letter(chars[idx])
            elif role == "digit":
                chars[idx] = coerce_digit(chars[idx])
            if chars[idx] != original:
                corrections.append({"position": idx, "from": original, "to": chars[idx], "reason": f"expected_{role}"})

        if len(chars) >= 2:
            state_raw = "".join(chars[:2])
            state = nearest_state_code(state_raw)
            if state != state_raw and state_code_cost(state_raw, state) <= 0.55:
                corrections.append({"position": "0-1", "from": state_raw, "to": state, "reason": "indian_state_prior"})
                chars[0], chars[1] = state[0], state[1]

    text = "".join(chars)
    format_hint = "india_private" if apply_indian and re.fullmatch(r"[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}", text) else "alphanumeric_sequence"
    confidence = prior_confidence(cleaned, text, corrections, format_hint, apply_indian)
    return {
        "text": text,
        "raw_text": cleaned,
        "format_hint": format_hint,
        "corrections": corrections,
        "confidence": confidence,
    }


def should_apply_indian_prior(cleaned: str) -> bool:
    if len(cleaned) < 7:
        return False
    if len(cleaned) >= 2 and cleaned[:2].isalpha() and cleaned[:2] in INDIAN_STATE_CODES:
        return True
    if len(cleaned) < 8 or not cleaned[:2].isalpha():
        return False
    digits = sum(ch.isdigit() for ch in cleaned)
    if digits < 4:
        return False
    candidate = nearest_state_code(cleaned[:2])
    return state_code_cost(cleaned[:2], candidate) <= 0.45


def expected_roles(length: int) -> list[str]:
    if length >= 9:
        roles = ["letter", "letter", "digit", "digit", "letter", "letter"]
        roles.extend(["digit"] * (length - len(roles)))
        return roles[:length]
    return ["unknown"] * length


def coerce_letter(char: str) -> str:
    if "A" <= char <= "Z":
        return char
    return LETTER_AMBIGUITIES.get(char, [char])[0]


def coerce_digit(char: str) -> str:
    if "0" <= char <= "9":
        return char
    return DIGIT_AMBIGUITIES.get(char, [char])[0]


def nearest_state_code(code: str) -> str:
    if code in INDIAN_STATE_CODES:
        return code
    scored = sorted((state_code_cost(code, candidate), candidate) for candidate in INDIAN_STATE_CODES)
    return scored[0][1]


def state_code_cost(observed: str, candidate: str) -> float:
    cost = 0.0
    for left, right in zip(observed, candidate):
        if left == right:
            continue
        if right in LETTER_AMBIGUITIES.get(left, []) or left in LETTER_AMBIGUITIES.get(right, []):
            cost += 0.35
        else:
            cost += 1.0
    return cost


def prior_confidence(raw_text: str, text: str, corrections: list[dict], format_hint: str, apply_indian: bool) -> float:
    base = 0.68 if raw_text else 0.0
    if apply_indian and format_hint == "india_private":
        base += 0.22
    base -= min(0.35, len(corrections) * 0.045)
    if "?" in text:
        base -= 0.2
    return round(max(0.0, min(0.99, base)), 4)

