"""Opt-in short acronym indels, backed by independent local context."""
import re


def code_spans(text, key, positions, registered):
    """Yield the same span shape as DirectEntitySpeller's literal route.

    Never infer numeric changes, substitutions, or substrings of longer codes.
    The declared context belongs to the entity, not to an observed ASR typo.
    """
    targets = [(k, items) for k, items in registered.items()
               if re.fullmatch(r'[a-z]{3,8}[0-9]{0,3}', k)]
    for token in re.finditer(r'(?<![a-z0-9_])[a-z]{2,9}[0-9]{0,3}(?![a-z0-9_])', key):
        observed = token.group()
        if observed in registered:
            continue  # A registered literal spelling always owns its surface.
        start, end = positions[token.start()][0], positions[token.end()-1][1]
        raw = text[start:end]
        letters = ''.join(re.findall(r'[A-Za-z]', raw))
        if not letters or not letters.isupper():
            continue
        prefix, digits = re.fullmatch(r'([a-z]+)([0-9]*)', observed).groups()
        context = text[max(0, start-192):start] + ' ' + text[end:end+192]
        matches = []
        for target, items in targets:
            expected, number = re.fullmatch(r'([a-z]+)([0-9]*)', target).groups()
            if digits != number or abs(len(prefix)-len(expected)) != 1:
                continue
            short, long = sorted((prefix, expected), key=len)
            if not any(long[:i]+long[i+1:] == short for i in range(len(long))):
                continue
            # Collect competitors even if one lacks permission/context. Never
            # choose a canonical owner simply because its rival was disabled
            # for fuzzy spelling.
            matches.extend((item, 'context_code_indel') for item in items)
        if len({item[0].canonical_text for item, _ in matches}) != 1:
            continue
        allowed = []
        for item, rule in matches:
            definition = item[0]
            policy = definition.spelling_context
            if (policy is None or policy.allow_near_spelling is not True
                    or (policy.canonical_text, policy.entity_type, policy.domain)
                    != (definition.canonical_text, definition.entity_type, definition.domain)):
                continue
            score = sum(weight for word, weight in policy.keywords
                        if word.casefold() in context.casefold())
            if score >= policy.minimum_score:
                allowed.append((item, rule))
        if allowed:
            yield start, end, raw, allowed
