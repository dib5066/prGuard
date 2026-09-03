"""
Unified diff parser for GitHub Pull Request diffs.

This module provides stateless functions for parsing the unified diff format
returned by the GitHub API into structured, queryable data. It is used by the
review pipeline to understand exactly which files changed, which lines were
added/removed, and which code symbols (functions, classes, imports) were
introduced.

Usage:
    from app.github.tools import parse_unified_diff

    diff_text = "diff --git a/app.py b/app.py\n..."
    parsed = parse_unified_diff(diff_text)

    for file in parsed.files:
        print(f"{file.path}: +{file.additions} -{file.deletions}")
        for hunk in file.hunks:
            print(f"  Lines {hunk.start_line}-{hunk.start_line + hunk.length}")
            print(f"    Added: {hunk.added_lines}")
            print(f"    Removed: {hunk.removed_lines}")
"""

import logging
import re
from dataclasses import dataclass, field
# data classes let you create simple classes for storing data
# instead of writing Class Diffclass:
            #               __init__():
#  can use dacorator @dataclass instead python generate a construcor automatically


logger = logging.getLogger(__name__)


# ============================================================
# Regular expressions used to read GitHub diff text
# ============================================================

# example of diff --git a/main.py b/main.py
# it extract --> main.py
#                main.py

FILE_HEADER_PATTERN = re.compile(
    r"^diff --git a/(.+?) b/(.+)$",
    re.MULTILINE,
)

# @@ -10,5 +10,6 @@ def calculate():
# it extract --> -10,5 -> lines 10 to 14
#               +10,6 -> lines 10 to 15             

HUNK_HEADER_PATTERN = re.compile(
    r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$",
    re.MULTILINE,
)

# new file mode 100644
#it extract --> new file mode
# 100644
# it just match the first three letters 'new file mode'

NEW_FILE_PATTERN = re.compile(
    r"^new file mode \d+$",
    re.MULTILINE,
)

# deleted file mode 100644
# it extract --> deleted file mode
#                100644

DELETED_FILE_PATTERN = re.compile(
    r"^deleted file mode \d+$",
    re.MULTILINE,
)

# it extract --> rename from --> old name
#                rename to    --> new name  

RENAMED_FILE_PATTERN = re.compile(
    r"^rename from (.+)$",
    re.MULTILINE,
)


# ============================================================
# Data classes
# ============================================================

# Create a data structure representing one changed section.
@dataclass
class DiffHunk:
    """One changed section inside a file."""
    
    old_start_line: int
    old_line_count: int

    new_start_line: int
    new_line_count: int

    # @@ -10,5 +10,6 @@ def calculate():
    # stores the context after the @@.
    # For example:
    # @@ -10,5 +10,6 @@ def calculate():
    # The context is:
    # def calculate():
    context_header: str

    added_lines: list[int] = field(default_factory=list)
    removed_lines: list[int] = field(default_factory=list)

    content: str = ""

# Store information about one changed file.
@dataclass
class DiffFile:
    """Information about one changed file."""

    path: str

    previous_path: str | None = None

    status: str = "modified"

    # Number of added and removed lines
    additions: int = 0
    deletions: int = 0
    
    # one file can contains many hunk
    # main.py
    # ├── hunk 1
    # ├── hunk 2
    # └── hunk 3
    
    hunks: list[DiffHunk] = field(default_factory=list)
    
    # the patch text for the file
    # store the original diff section 
    patch: str = ""

# Represent the whole PR diff
# PR #42
# │
# ├── main.py
# │   ├── hunk 1
# │   ├── hunk 2
# │   └── hunk 3
# │
# ├── auth.py
# │   └── hunk 1
# │
# └── database.py
#     ├── hunk 1
#     ├── hunk 2
#     └── hunk 3

@dataclass
class ParsedDiff:
    """Complete parsed information about a pull request diff."""

    files: list[DiffFile] = field(default_factory=list)

    total_additions: int = 0
    total_deletions: int = 0
    total_files: int = 0


@dataclass
class CodeSymbol:
    """A function, class, or import introduced by the PR."""

    name: str
    symbol_type: str
    language: str


# ============================================================
# Parse one hunk
# ============================================================

def parse_hunk(hunk_text: str) -> DiffHunk:
    """Parse one @@ ... @@ section from a GitHub diff."""

    lines = hunk_text.splitlines()

    if not lines:
        return DiffHunk(
            old_start_line=0,
            old_line_count=0,
            new_start_line=0,
            new_line_count=0,
            context_header="",
        )

    # The first line should look like:
    # @@ -10,5 +10,6 @@ def calculate():

    header_match = HUNK_HEADER_PATTERN.match(lines[0])

    if not header_match:
        logger.warning(
            "Could not parse hunk header: %s",
            lines[0],
        )

        return DiffHunk(
            old_start_line=0,
            old_line_count=0,
            new_start_line=0,
            new_line_count=0,
            context_header="",
        )

    old_start_line = int(header_match.group(1))
    old_line_count = int(header_match.group(2) or "1")

    new_start_line = int(header_match.group(3))
    new_line_count = int(header_match.group(4) or "1")

    context_header = header_match.group(5).strip()

    added_lines = []
    removed_lines = []

    current_old_line = old_start_line
    current_new_line = new_start_line

    # Read every line after the @@ header.
    for line in lines[1:]:

        # Added line
        if line.startswith("+"):
            added_lines.append(current_new_line)
            current_new_line += 1

        # Removed line
        elif line.startswith("-"):
            removed_lines.append(current_old_line)
            current_old_line += 1

        # Context line
        else:
            current_old_line += 1
            current_new_line += 1

    return DiffHunk(
        old_start_line=old_start_line,
        old_line_count=old_line_count,
        new_start_line=new_start_line,
        new_line_count=new_line_count,
        context_header=context_header,
        added_lines=added_lines,
        removed_lines=removed_lines,
        content=hunk_text,
    )


# ============================================================
# Parse one file
# ============================================================

def parse_diff_file(file_section: str) -> DiffFile:
    """Parse the diff section belonging to one file."""

    lines = file_section.splitlines(keepends=True)

    if not lines:
        return DiffFile(path="")

    # Example:
    #
    # diff --git a/main.py b/main.py
    #
    first_line = lines[0].rstrip("\n")

    file_match = FILE_HEADER_PATTERN.match(first_line)

    if not file_match:
        logger.warning(
            "Could not parse file header: %s",
            first_line,
        )

        return DiffFile(path="")

    current_path = file_match.group(2)
    previous_path = file_match.group(1)

    # If both paths are the same, the file was not renamed.
    if current_path == previous_path:
        previous_path = None

    # --------------------------------------------------------
    # Find file status
    # --------------------------------------------------------

    status = "modified"

    if NEW_FILE_PATTERN.search(file_section):
        status = "added"

    elif DELETED_FILE_PATTERN.search(file_section):
        status = "removed"

    elif RENAMED_FILE_PATTERN.search(file_section):
        status = "renamed"

    # --------------------------------------------------------
    # Find all hunks
    # --------------------------------------------------------

    hunk_positions = [
        match.start()
        for match in HUNK_HEADER_PATTERN.finditer(file_section)
    ]

    hunks = []

    for index, start_position in enumerate(hunk_positions):

        if index + 1 < len(hunk_positions):
            end_position = hunk_positions[index + 1]
        else:
            end_position = len(file_section)

        hunk_text = file_section[
            start_position:end_position
        ].rstrip("\n")

        hunks.append(parse_hunk(hunk_text))

    # --------------------------------------------------------
    # Count added and removed lines
    # --------------------------------------------------------

    additions = 0
    deletions = 0

    for hunk in hunks:
        additions += len(hunk.added_lines)
        deletions += len(hunk.removed_lines)

    return DiffFile(
        path=current_path,
        previous_path=previous_path,
        status=status,
        additions=additions,
        deletions=deletions,
        hunks=hunks,
        patch=file_section.rstrip("\n"),
    )


# ============================================================
# Parse the complete pull request diff
# ============================================================

def parse_unified_diff(diff_text: str) -> ParsedDiff:
    """Parse the complete diff returned by GitHub."""

    if not diff_text or not diff_text.strip():
        logger.warning("Empty diff text provided")
        return ParsedDiff()

    # GitHub starts every file section with:
    # diff --git a/... b/...
    file_sections = re.split(
        r"(?=^diff --git )",
        diff_text,
        flags=re.MULTILINE,
    )

    files = []

    total_additions = 0
    total_deletions = 0

    for file_section in file_sections:

        file_section = file_section.strip()

        if not file_section:
            continue

        if not file_section.startswith("diff --git"):
            continue

        diff_file = parse_diff_file(file_section)

        if not diff_file.path:
            continue

        files.append(diff_file)

        total_additions += diff_file.additions
        total_deletions += diff_file.deletions

    parsed_diff = ParsedDiff(
        files=files,
        total_additions=total_additions,
        total_deletions=total_deletions,
        total_files=len(files),
    )

    logger.debug(
        "Parsed diff: %d files, +%d -%d",
        parsed_diff.total_files,
        parsed_diff.total_additions,
        parsed_diff.total_deletions,
    )

    return parsed_diff


# ============================================================
# Helper functions
# ============================================================

def get_changed_file_paths(
    parsed_diff: ParsedDiff,
) -> list[str]:
    """Return the paths of all changed files."""

    return [
        diff_file.path
        for diff_file in parsed_diff.files
    ]


def get_files_by_status(
    parsed_diff: ParsedDiff,
    file_status: str,
) -> list[DiffFile]:
    """Return files matching a specific status."""

    return [
        diff_file
        for diff_file in parsed_diff.files
        if diff_file.status == file_status
    ]


def get_added_lines_for_file(
    diff_file: DiffFile,
) -> list[int]:
    """Return all added line numbers in a file."""

    added_lines = []

    for hunk in diff_file.hunks:
        added_lines.extend(hunk.added_lines)

    return sorted(added_lines)


def get_removed_lines_for_file(
    diff_file: DiffFile,
) -> list[int]:
    """Return all removed line numbers in a file."""

    removed_lines = []

    for hunk in diff_file.hunks:
        removed_lines.extend(hunk.removed_lines)

    return sorted(removed_lines)


# ============================================================
# Extract added symbols from a parsed diff
# ============================================================


def extract_added_symbols(
    parsed_diff: ParsedDiff,
) -> dict[str, list[CodeSymbol]]:
    """
    Extract function, class, and import names introduced by the PR.

    Scans added lines (lines starting with '+') from the raw patch
    for Python definitions:
        - def func_name(...)
        - class ClassName(...)
        - import module
        - from x import y

    Args:
        parsed_diff: The fully parsed diff.

    Returns:
        Dictionary mapping file paths to lists of CodeSymbol objects.
    """
    import re

    symbol_patterns = [
        (r"^\s*def\s+(\w+)", "function"),
        (r"^\s*class\s+(\w+)", "class"),
        (r"^\s*import\s+(\w+)", "import"),
        (r"^\s*from\s+\S+\s+import\s+(\w+)", "import"),
    ]

    result: dict[str, list[CodeSymbol]] = {}

    for diff_file in parsed_diff.files:
        symbols: list[CodeSymbol] = []
        seen_names: set[str] = set()

        for hunk in diff_file.hunks:
            for raw_line in hunk.content.split("\n"):
                # Only scan added lines (start with +, skip +++ header)
                if not raw_line.startswith("+") or raw_line.startswith("+++"):
                    continue

                # Strip the leading +
                line_text = raw_line[1:]

                for pattern, symbol_type in symbol_patterns:
                    match = re.match(pattern, line_text)
                    if match:
                        name = match.group(1)
                        if name not in seen_names:
                            seen_names.add(name)
                            symbols.append(
                                CodeSymbol(
                                    name=name,
                                    symbol_type=symbol_type,
                                    language="python",
                                )
                            )
                        break

        if symbols:
            result[diff_file.path] = symbols

    return result
