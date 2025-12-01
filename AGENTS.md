# AI Agent Instructions

Universal Python patterns for AI coding agents. Project-specific details are in CLAUDE.md.

Follows [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html).

## Code Quality Gate

Before completing any task, run:

```bash
uv run ruff check . --fix
uv run ruff format .
uv run ruff check .
```

If ruff fails, the task is not complete.

## Google Style Conventions

### Imports

Order imports in three groups separated by blank lines:
1. Standard library
2. Third-party packages
3. Local modules

```python
import os
import sys
from pathlib import Path

import httpx
from pydantic import BaseModel

from myproject.models import User
from myproject.utils import helpers
```

Do not use relative imports. Do not use `from module import *`.

### Naming

| Type | Convention | Example |
|------|------------|---------|
| Modules | `lowercase_underscores` | `user_service.py` |
| Classes | `CapWords` | `UserService` |
| Functions | `lowercase_underscores` | `get_user_by_id` |
| Variables | `lowercase_underscores` | `user_count` |
| Constants | `ALL_CAPS` | `MAX_RETRIES` |
| Private | `_leading_underscore` | `_internal_cache` |

### Docstrings

Use Google-style docstrings for public functions:

```python
def fetch_user(user_id: str, include_deleted: bool = False) -> User | None:
    """Fetch a user by ID from the database.

    Args:
        user_id: The unique identifier of the user.
        include_deleted: Whether to include soft-deleted users.

    Returns:
        The User object if found, None otherwise.

    Raises:
        DatabaseError: If the database connection fails.
    """
```

Skip docstrings for private functions, simple getters, and obvious one-liners.

### Exceptions

Be specific with exception types. Never use bare `except:`.

```python
# BAD
try:
    value = data["key"]
except:
    value = default

# BAD - Too broad
try:
    value = data["key"]
except Exception:
    value = default

# GOOD - Specific exception
try:
    value = data["key"]
except KeyError:
    value = default
```

### Comparisons

Use `is` for None, True, False. Use implicit truthiness for sequences.

```python
# BAD
if x == None:
if len(items) == 0:
if len(items) > 0:
if valid == True:

# GOOD
if x is None:
if not items:
if items:
if valid:
```

### Default Arguments

Never use mutable default arguments.

```python
# BAD - Mutable default shared across calls
def append_item(item, items=[]):
    items.append(item)
    return items

# GOOD - Use None and create new list
def append_item(item, items: list | None = None) -> list:
    if items is None:
        items = []
    items.append(item)
    return items
```

### Comprehensions

Use comprehensions for simple transformations. Use loops for complex logic.

```python
# GOOD - Simple transformation
names = [user.name for user in users]
active = {u.id: u for u in users if u.active}

# BAD - Too complex for comprehension
result = [
    transform(x) 
    for x in items 
    if x.valid and x.type == "special" 
    for y in x.children 
    if y.enabled
]

# GOOD - Use explicit loop for complex logic
result = []
for x in items:
    if not (x.valid and x.type == "special"):
        continue
    for y in x.children:
        if y.enabled:
            result.append(transform(x))
```

## Pydantic Patterns

### 1. Models End-to-End

When a Pydantic model exists, use it for serialization and deserialization. Never convert to dict just to serialize.

```python
# BAD - Model exists but we use dict
def update_metadata(self, data: dict[str, Any]) -> None:
    current = json.load(f)
    current.update(data)
    json.dump(current, f)

def read_metadata(self) -> dict[str, Any]:
    return json.load(f)

# GOOD - Use the model
def save_metadata(self, metadata: RunMetadata) -> None:
    self.path.write_text(metadata.model_dump_json(indent=2))

def load_metadata(self) -> RunMetadata:
    return RunMetadata.model_validate_json(self.path.read_text())
```

### 2. Fail Fast on Required Fields

Do not use `.get()` with defaults for required fields. Let validation fail early.

```python
# BAD - Silent failures, invalid data propagates
agent = data.get("agent_name", "")
dataset = data.get("dataset_name", "")
count = int(data.get("count", 0) or 0)

# GOOD - Validation fails immediately if data is malformed
result = CaseResult.model_validate(data)
agent = result.agent_name
dataset = result.dataset_name
count = result.count
```

### 3. Return Types, Not Dicts

Functions should return typed models, not dictionaries.

```python
# BAD - Type erasure
def load_results(path: Path) -> list[dict]:
    return [json.loads(line) for line in f]

# GOOD - Typed returns
def load_results(path: Path) -> list[CaseResult]:
    return [CaseResult.model_validate_json(line) for line in f]
```

### 4. Single Source of Truth

Do not duplicate utility functions across modules.

```python
# BAD - Same function in two files
# manager.py
def _read_metadata_file(path: Path) -> dict: ...
# discovery.py
def _read_metadata_file(path: Path) -> dict: ...

# GOOD - Single location, imported everywhere
# storage.py
def load_metadata(path: Path) -> RunMetadata: ...
```

### 5. Direct Attribute Access

Do not use defensive getattr or dict.get for known model fields.

```python
# BAD - Unnecessary indirection
duration = getattr(result, "duration_ms", 0)
name = result.__dict__.get("name", "")

# GOOD - Direct access, fails fast if field missing
duration = result.duration_ms
name = result.name
```

### 6. Type Hints Required

Every function must have complete type hints.

```python
# BAD
def process(items):
    return [x.name for x in items]

# GOOD
def process(items: list[Item]) -> list[str]:
    return [x.name for x in items]
```

### 7. Minimal Context Objects

Do not create god objects that hold everything. Pass parameters directly.

```python
# BAD - Context as a dumping ground
class Context:
    db: Database
    cache: Cache
    config: Config
    user: User
    request: Request
    logger: Logger
    metrics: Metrics

def process(ctx: Context) -> None: ...

# GOOD - Pass what you need
def process(db: Database, user: User) -> Result: ...
```

### 8. No TYPE_CHECKING Guards

Using `if TYPE_CHECKING:` is a code smell indicating circular dependencies. Fix the architecture instead.

```python
# BAD - Hiding circular imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from myapp.models import User

# GOOD - Fix the dependency structure
# Move shared types to a base module that both can import
```

## Anti-Pattern Reference

| Anti-Pattern | Correct Pattern |
|--------------|-----------------|
| `dict[str, Any]` when model exists | Use the model type |
| `json.dumps(obj.__dict__)` | `obj.model_dump_json()` |
| `Model(**json.loads(s))` | `Model.model_validate_json(s)` |
| `data.get("field", default)` for required fields | `model.field` after validation |
| Return `dict` from functions | Return typed model |
| `if TYPE_CHECKING:` import guards | Fix circular dependencies |
| Duplicate utility functions | Single canonical location |
| `getattr(obj, "field", default)` | `obj.field` |
| Bare `except:` | Specific exception type |
| `if x == None` | `if x is None` |
| `if len(items) == 0` | `if not items` |
| Mutable default arguments | `None` with conditional init |
| `from module import *` | Explicit imports |

## Do Not

1. Add redundant comments that restate the code
2. Create abstractions for one-time operations
3. Use magic numbers without named constants
4. Commit code that fails ruff check
5. Skip type hints on any function
6. Use bare except clauses
7. Use mutable default arguments
8. Use relative imports
