# Example missions

These are examples of the intended natural-language interface. They are not
hard-coded commands.

- `Take off to 10 m. Find the nearest building. Circle it once while avoiding obstacles. Return home and land.`
- `Find a tree, move to a safe viewing position, photograph it, then return home.`
- `Inspect the building from all sides at 9 m. Avoid obstacles and return home when finished.`
- `Take off to 7 m and do not exceed 7 m. Find the nearest building, hover near it for 10 seconds, then return home and land.`

If the mission is ambiguous or cannot be represented safely, the interpreter should
clarify or reject rather than invent hidden coordinates.
