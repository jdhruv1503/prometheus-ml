"""Competition DNA classifier."""


def classify_archetype(profile: dict) -> str:
    rows = profile.get("shape", {}).get("rows", 0)
    if rows > 100000:
        return "large-scale-tabular"
    return "standard-tabular"
