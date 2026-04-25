"""SWE-bench tool definitions for LangChain agent.

Provides real tool implementations scoped to a repository working directory.

Two tool sets:
  - create_swebench_tools(): Original 6-tool set (bash, file_view, file_read, file_write, file_str_replace, search)
  - create_sweagent_tools(): SWE-agent compatible 3-tool set (bash, str_replace_editor, submit)
"""

import os
import shutil
import subprocess
from pathlib import Path
from typing import Literal, Optional

from langchain_core.tools import tool


def create_swebench_tools(workdir: str, repo: str = "") -> list:
    """Create SWE-bench tools that operate within the given working directory.

    Args:
        workdir: Absolute path to the repository root directory.
        repo: GitHub repo identifier (e.g. 'astropy/astropy') used to
              auto-strip redundant path prefixes from tool arguments.

    Returns:
        List of LangChain tool objects.
    """
    workdir = os.path.abspath(workdir)

    # Build list of prefixes to try stripping when a path is not found.
    # For repo='astropy/astropy', we try stripping 'astropy/astropy/' and 'astropy/' etc.
    _strip_prefixes = []
    if repo:
        parts = repo.split("/")
        # e.g. 'astropy/astropy/' then 'astropy/'
        for i in range(len(parts)):
            prefix = "/".join(parts[i:]) + "/"
            if prefix not in _strip_prefixes:
                _strip_prefixes.append(prefix)

    def _safe_path(path: str) -> str:
        """Resolve a path and ensure it stays within workdir.

        If the resolved path does not exist, tries stripping known repo
        prefixes (e.g. 'org/repo/') to recover from common LLM path errors.
        """
        if os.path.isabs(path):
            full = os.path.abspath(path)
        else:
            full = os.path.abspath(os.path.join(workdir, path))
        if not full.startswith(workdir):
            raise ValueError(f"Path escapes working directory: {path}")

        # Auto-correct: if path doesn't exist, try stripping repo prefixes
        if not os.path.exists(full) and _strip_prefixes:
            for prefix in _strip_prefixes:
                if path.startswith(prefix):
                    stripped = path[len(prefix):]
                    candidate = os.path.abspath(os.path.join(workdir, stripped))
                    if candidate.startswith(workdir) and os.path.exists(candidate):
                        return candidate

        return full

    @tool
    def bash(command: str) -> str:
        """Execute a shell command in the repository directory.

        Args:
            command: The shell command to execute.

        Returns:
            Combined stdout and stderr output (truncated to 4000 chars).
        """
        try:
            # Python 경로 자동 감지 (python3 > python)
            python_exe = shutil.which("python3") or shutil.which("python") or "python3"
            if command.startswith("python "):
                command = python_exe + " " + command[7:]

            # GitHub 접근 설정 (Personal Access Token 또는 SSH)
            env = {
                **os.environ,
                "GIT_TERMINAL_PROMPT": "0",
            }
            result = subprocess.run(
                command,
                shell=True,
                cwd=workdir,
                capture_output=True,
                text=True,
                timeout=60,
                env=env,
            )
            output = ""
            if result.stdout:
                output += result.stdout
            if result.stderr:
                output += result.stderr
            if not output:
                output = f"[exit code {result.returncode}]"
            else:
                output += f"\n[exit code {result.returncode}]"
            return output[:4000]
        except subprocess.TimeoutExpired:
            return "[ERROR] Command timed out after 60 seconds."
        except Exception as e:
            return f"[ERROR] {e}"

    @tool
    def file_view(path: str, view_range: tuple[int, int] = None) -> str:
        """View a file or directory. For files, optionally specify line range.

        Args:
            path: Path to file or directory (relative to repo root or absolute).
            view_range: Optional tuple (start_line, end_line) with 1-based line numbers.
                       If None, shows first 100 lines.

        Returns:
            File/directory contents with line numbers (for files).
        """
        try:
            full_path = _safe_path(path)

            # Handle directories
            if os.path.isdir(full_path):
                items = []
                try:
                    for item in sorted(os.listdir(full_path)):
                        item_path = os.path.join(full_path, item)
                        if os.path.isdir(item_path):
                            items.append(f"  {item}/")
                        else:
                            items.append(f"  {item}")
                except PermissionError:
                    return f"[ERROR] Permission denied: {path}"
                return "Directory contents:\n" + "\n".join(items)

            # Handle files
            with open(full_path) as f:
                lines = f.readlines()

            # Determine range
            if view_range:
                start_line, end_line = view_range
                start = max(0, start_line - 1)
                end = min(len(lines), end_line)
            else:
                start = 0
                end = min(100, len(lines))  # Default: first 100 lines

            selected = lines[start:end]

            numbered = []
            for i, line in enumerate(selected, start=start + 1):
                numbered.append(f"{i:>6}\t{line.rstrip()}")

            total_lines = len(lines)
            header = f"File: {path} ({end - start}/{total_lines} lines shown)"
            output = header + "\n" + "\n".join(numbered)
            return output[:8000] if output else "[empty file]"
        except FileNotFoundError:
            return f"[ERROR] File not found: {path}"
        except ValueError as e:
            return f"[ERROR] {e}"
        except Exception as e:
            return f"[ERROR] {e}"

    @tool
    def file_read(path: str, start_line: int = None, end_line: int = None) -> str:
        """Read a file from the repository (compatibility wrapper for file_view).

        Args:
            path: Path to the file (relative to repo root or absolute).
            start_line: Optional 1-based start line number.
            end_line: Optional 1-based end line number.

        Returns:
            File contents with line numbers.
        """
        try:
            full_path = _safe_path(path)
            with open(full_path) as f:
                lines = f.readlines()

            start = (start_line - 1) if start_line and start_line >= 1 else 0
            end = end_line if end_line else len(lines)
            selected = lines[start:end]

            numbered = []
            for i, line in enumerate(selected, start=start + 1):
                numbered.append(f"{i:>6}\t{line.rstrip()}")

            output = "\n".join(numbered)
            return output[:8000] if output else "[empty file]"
        except FileNotFoundError:
            return f"[ERROR] File not found: {path}"
        except ValueError as e:
            return f"[ERROR] {e}"
        except Exception as e:
            return f"[ERROR] {e}"

    @tool
    def file_write(path: str, content: str) -> str:
        """Write content to a file in the repository. WARNING: This overwrites the entire file.
        For partial edits, use file_str_replace instead.

        Args:
            path: Path to the file (relative to repo root or absolute).
            content: Full content to write to the file.

        Returns:
            Success or error message, with warning if existing file was significantly truncated.
        """
        try:
            full_path = _safe_path(path)
            warning = ""
            # Warn if overwriting a much larger existing file
            if os.path.exists(full_path):
                old_size = os.path.getsize(full_path)
                new_size = len(content.encode("utf-8"))
                if old_size > 0 and new_size < old_size * 0.5:
                    warning = (
                        f" [WARNING] Original file was {old_size} bytes but new content is only "
                        f"{new_size} bytes ({new_size*100//old_size}% of original). "
                        f"You may have accidentally deleted important code. "
                        f"Consider using file_str_replace for targeted edits instead."
                    )
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w") as f:
                f.write(content)
            return f"Successfully wrote {len(content)} bytes to {path}" + warning
        except ValueError as e:
            return f"[ERROR] {e}"
        except Exception as e:
            return f"[ERROR] {e}"

    @tool
    def file_str_replace(path: str, old_str: str, new_str: str) -> str:
        """Replace a string in a file (exact match).

        This is safer than file_write for partial edits. Searches for old_str
        and replaces it with new_str. The old_str must match exactly once.

        Args:
            path: Path to the file (relative to repo root or absolute).
            old_str: The exact string to find and replace.
            new_str: The string to replace it with.

        Returns:
            Success message with line count, or error message.
        """
        try:
            full_path = _safe_path(path)
            with open(full_path, "r") as f:
                content = f.read()

            # Count occurrences
            count = content.count(old_str)
            if count == 0:
                return f"[ERROR] Pattern not found in {path}. Read the file first with file_read to see its current content."
            if count > 1:
                # Show line numbers where the pattern appears to help the agent
                lines = content.split("\n")
                locations = []
                for i, line in enumerate(lines, 1):
                    if old_str.split("\n")[0] in line:
                        locations.append(f"  line {i}: {line.strip()[:80]}")
                loc_info = "\n".join(locations[:5])
                return (
                    f"[ERROR] Pattern appears {count} times in {path} (must appear exactly once). "
                    f"Include more surrounding context in old_str to make it unique.\n"
                    f"Occurrences found at:\n{loc_info}"
                )

            # Replace
            new_content = content.replace(old_str, new_str)
            with open(full_path, "w") as f:
                f.write(new_content)

            # Count lines changed
            old_lines = old_str.count("\n")
            new_lines = new_str.count("\n")
            return f"Successfully replaced in {path} ({old_lines} lines removed, {new_lines} lines added)"
        except FileNotFoundError:
            return f"[ERROR] File not found: {path}"
        except ValueError as e:
            return f"[ERROR] {e}"
        except Exception as e:
            return f"[ERROR] {e}"

    @tool
    def search(pattern: str, path: str = ".", file_pattern: str = None) -> str:
        """Search for a pattern in the repository using grep.

        Args:
            pattern: Grep-compatible regex pattern to search for.
            path: Directory or file to search in (relative to repo root).
            file_pattern: Optional glob to filter files (e.g. '*.py').

        Returns:
            Matching lines with file paths and line numbers.
        """
        try:
            search_path = _safe_path(path)
            cmd = ["grep", "-rn", "--include", file_pattern or "*", pattern, search_path]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=workdir,
            )
            output = result.stdout
            if not output:
                return f"No matches found for pattern: {pattern}"
            # Strip workdir prefix from paths for cleaner output
            output = output.replace(workdir + "/", "")
            return output[:8000]
        except subprocess.TimeoutExpired:
            return "[ERROR] Search timed out after 30 seconds."
        except ValueError as e:
            return f"[ERROR] {e}"
        except Exception as e:
            return f"[ERROR] {e}"

    return [bash, file_view, file_read, file_write, file_str_replace, search]


# ── SWE-agent compatible tools ──────────────────────────────────────


def create_sweagent_tools(workdir: str, repo: str = "") -> list:
    """Create SWE-agent compatible tools (bash, str_replace_editor, submit).

    These match the exact tool names and signatures used to train
    SWE-agent-LM models (SWE-smith paper, arXiv:2504.21798).

    Args:
        workdir: Absolute path to the repository root directory.
        repo: GitHub repo identifier for path auto-correction.

    Returns:
        List of LangChain tool objects.
    """
    workdir = os.path.abspath(workdir)

    # Path prefix stripping (same logic as create_swebench_tools)
    _strip_prefixes = []
    if repo:
        parts = repo.split("/")
        for i in range(len(parts)):
            prefix = "/".join(parts[i:]) + "/"
            if prefix not in _strip_prefixes:
                _strip_prefixes.append(prefix)

    def _safe_path(path: str) -> str:
        """Resolve path, ensure it stays within workdir."""
        if os.path.isabs(path):
            full = os.path.abspath(path)
        else:
            full = os.path.abspath(os.path.join(workdir, path))
        if not full.startswith(workdir):
            raise ValueError(f"Path escapes working directory: {path}")
        if not os.path.exists(full) and _strip_prefixes:
            for prefix in _strip_prefixes:
                if path.startswith(prefix):
                    stripped = path[len(prefix):]
                    candidate = os.path.abspath(os.path.join(workdir, stripped))
                    if candidate.startswith(workdir) and os.path.exists(candidate):
                        return candidate
        return full

    # File edit history for undo_edit
    _file_history: dict[str, list[str]] = {}

    # Interactive command blocklist (SWE-agent blocks these)
    _blocked_commands = {"python", "python3", "ipython", "bash", "sh", "vim", "vi", "emacs", "nano", "nohup"}
    _blocked_prefixes = ("vim ", "vi ", "emacs ", "nano ", "gdb ", "less ", "tail -f ", "nohup ")

    @tool
    def bash(command: str) -> str:
        """Execute a bash command in the repository directory.

        Args:
            command: The bash command to execute.

        Returns:
            Command output (stdout + stderr), truncated to 16000 chars.
        """
        # Block interactive commands
        cmd_stripped = command.strip()
        if cmd_stripped in _blocked_commands:
            return f"[ERROR] Interactive command '{cmd_stripped}' is not allowed. Use non-interactive alternatives."
        for prefix in _blocked_prefixes:
            if cmd_stripped.startswith(prefix):
                return f"[ERROR] Interactive command '{prefix.strip()}' is not allowed."

        try:
            python_exe = shutil.which("python3") or shutil.which("python") or "python3"
            if command.startswith("python "):
                command = python_exe + " " + command[7:]

            env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
            result = subprocess.run(
                command, shell=True, cwd=workdir,
                capture_output=True, text=True, timeout=120, env=env,
            )
            output = (result.stdout or "") + (result.stderr or "")
            if not output:
                return f"[exit code {result.returncode}]"
            if len(output) > 16000:
                output = output[:8000] + "\n... (truncated) ...\n" + output[-8000:]
            return output + f"\n[exit code {result.returncode}]"
        except subprocess.TimeoutExpired:
            return "[ERROR] Command timed out after 120 seconds."
        except Exception as e:
            return f"[ERROR] {e}"

    @tool
    def str_replace_editor(
        command: Literal["view", "create", "str_replace", "insert", "undo_edit"],
        path: str,
        file_text: Optional[str] = None,
        view_range: Optional[list[int]] = None,
        old_str: Optional[str] = None,
        new_str: Optional[str] = None,
        insert_line: Optional[int] = None,
    ) -> str:
        """Custom editing tool for viewing, creating and editing files.

        Commands:
          view: View file contents or directory listing.
          create: Create a new file with given content.
          str_replace: Replace an exact string in a file (must match exactly once).
          insert: Insert text after a given line number.
          undo_edit: Undo the last edit to a file.

        Args:
            command: The operation to perform.
            path: Absolute or relative path to the file/directory.
            file_text: Content for 'create' command.
            view_range: [start_line, end_line] for 'view' (1-indexed, -1 for EOF).
            old_str: String to find for 'str_replace'.
            new_str: Replacement string for 'str_replace' or text for 'insert'.
            insert_line: Line number after which to insert (0 = before first line).

        Returns:
            Result of the operation.
        """
        try:
            full_path = _safe_path(path)
        except ValueError as e:
            return f"[ERROR] {e}"

        # ── VIEW ──
        if command == "view":
            if os.path.isdir(full_path):
                # Directory listing (2 levels deep)
                items = []
                for root, dirs, files in os.walk(full_path):
                    depth = root.replace(full_path, "").count(os.sep)
                    if depth >= 2:
                        dirs.clear()
                        continue
                    indent = "  " * depth
                    dirname = os.path.basename(root)
                    if depth > 0:
                        items.append(f"{indent}{dirname}/")
                    for fname in sorted(files):
                        if not fname.startswith("."):
                            items.append(f"{indent}  {fname}")
                    dirs[:] = sorted(d for d in dirs if not d.startswith("."))
                output = f"Directory: {path}\n" + "\n".join(items)
                return output[:16000]

            if not os.path.exists(full_path):
                return f"[ERROR] File not found: {path}"

            with open(full_path, errors="replace") as f:
                lines = f.readlines()

            if view_range:
                start = max(0, view_range[0] - 1)
                end = len(lines) if (len(view_range) < 2 or view_range[1] == -1) else min(len(lines), view_range[1])
            else:
                start = 0
                end = len(lines)

            numbered = []
            for i, line in enumerate(lines[start:end], start=start + 1):
                numbered.append(f"{i:>6}\t{line.rstrip()}")

            header = f"Here's the result of running `cat -n` on {path}:"
            output = header + "\n" + "\n".join(numbered)
            return output[:16000]

        # ── CREATE ──
        elif command == "create":
            if os.path.exists(full_path):
                return f"[ERROR] File already exists at: {path}. Use str_replace to edit it."
            if file_text is None:
                return "[ERROR] file_text is required for create command."
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w") as f:
                f.write(file_text)
            return f"File created successfully at: {path}"

        # ── STR_REPLACE ──
        elif command == "str_replace":
            if not os.path.exists(full_path):
                return f"[ERROR] File not found: {path}"
            if old_str is None:
                return "[ERROR] old_str is required for str_replace command."
            if new_str is None:
                new_str = ""

            with open(full_path, "r") as f:
                content = f.read()

            # Save history for undo
            if full_path not in _file_history:
                _file_history[full_path] = []
            _file_history[full_path].append(content)

            count = content.count(old_str)
            if count == 0:
                return f"[ERROR] No replacement was performed, old_str `{old_str[:100]}...` did not appear verbatim in {path}."
            if count > 1:
                return (
                    f"[ERROR] No replacement was performed. Multiple occurrences of old_str "
                    f"found ({count} times). Please ensure it is unique."
                )

            new_content = content.replace(old_str, new_str, 1)
            with open(full_path, "w") as f:
                f.write(new_content)

            # Show result around the edit
            new_lines = new_content.split("\n")
            # Find where the replacement happened
            old_first_line = old_str.split("\n")[0]
            edit_line = 0
            for i, line in enumerate(new_lines):
                if new_str and new_str.split("\n")[0] in line:
                    edit_line = i
                    break

            start = max(0, edit_line - 3)
            end = min(len(new_lines), edit_line + len((new_str or "").split("\n")) + 3)
            snippet = []
            for i, line in enumerate(new_lines[start:end], start=start + 1):
                snippet.append(f"{i:>6}\t{line}")

            return f"The file {path} has been edited. Here's the result of running `cat -n` on a snippet:\n" + "\n".join(snippet)

        # ── INSERT ──
        elif command == "insert":
            if not os.path.exists(full_path):
                return f"[ERROR] File not found: {path}"
            if insert_line is None:
                return "[ERROR] insert_line is required for insert command."
            if new_str is None:
                return "[ERROR] new_str is required for insert command."

            with open(full_path, "r") as f:
                content = f.read()

            # Save history for undo
            if full_path not in _file_history:
                _file_history[full_path] = []
            _file_history[full_path].append(content)

            lines = content.split("\n")
            insert_idx = max(0, min(insert_line, len(lines)))
            new_lines_to_insert = new_str.split("\n")
            lines[insert_idx:insert_idx] = new_lines_to_insert

            new_content = "\n".join(lines)
            with open(full_path, "w") as f:
                f.write(new_content)

            # Show snippet around insertion
            start = max(0, insert_idx - 3)
            end = min(len(lines), insert_idx + len(new_lines_to_insert) + 3)
            snippet = []
            for i, line in enumerate(lines[start:end], start=start + 1):
                snippet.append(f"{i:>6}\t{line}")

            return f"The file {path} has been edited. Here's the result of running `cat -n` on a snippet:\n" + "\n".join(snippet)

        # ── UNDO_EDIT ──
        elif command == "undo_edit":
            if not os.path.exists(full_path):
                return f"[ERROR] File not found: {path}"
            if full_path not in _file_history or not _file_history[full_path]:
                return f"[ERROR] No edit history for {path}."

            prev_content = _file_history[full_path].pop()
            with open(full_path, "w") as f:
                f.write(prev_content)
            return f"Last edit to {path} undone successfully."

        else:
            return f"[ERROR] Unknown command: {command}. Use view, create, str_replace, insert, or undo_edit."

    @tool
    def submit() -> str:
        """Submit the current state of the repository as the final answer.

        Call this when you are done fixing the issue and want to submit your changes.

        Returns:
            Submission confirmation.
        """
        return "Submission successful."

    return [bash, str_replace_editor, submit]


# ── mini-swe-agent compatible: bash-only ────────────────────────────


_MSWA_LONG_OUTPUT_WARNING = (
    "The output of your last command was too long.\n"
    "Please try a different command that produces less output.\n"
    "If you're looking at a file you can try use head, tail or sed to view a smaller number of lines selectively.\n"
    "If you're using grep or find and it produced too much output, you can use a more selective search pattern.\n"
    "If you really need to see something from the full command's output, you can redirect output to a file and then search in that file."
)


def _format_minisweagent_observation(
    output: str, returncode: int, exception_info: str = "",
) -> str:
    """Render mini-swe-agent's official observation_template (default.yaml).

    Threshold: 10000 chars on the *bash output* (stdout+stderr). Below it,
    the model sees the full output; above, head[:5000] + elided count +
    tail[-5000:]. Thinking tokens in the model's own response are NOT
    affected — only this tool result is bounded.
    """
    parts: list[str] = []
    if exception_info:
        parts.append(f"<exception>{exception_info}</exception>")
    parts.append(f"<returncode>{returncode}</returncode>")
    if len(output) < 10000:
        parts.append(f"<output>\n{output}\n</output>")
    else:
        elided = len(output) - 10000
        parts.append(f"<warning>\n{_MSWA_LONG_OUTPUT_WARNING}\n</warning>")
        parts.append(f"<output_head>\n{output[:5000]}\n</output_head>")
        parts.append(
            f"<elided_chars>\n{elided} characters elided\n</elided_chars>")
        parts.append(f"<output_tail>\n{output[-5000:]}\n</output_tail>")
    return "\n".join(parts)


def create_minisweagent_tools(workdir: str, repo: str = "") -> list:
    """Create a bash-only tool set matching mini-swe-agent's official
    SWE-Bench config (``benchmarks/swebench.yaml`` + ``default.yaml``).

    Mini-swe-agent's reference setup gives the model a single shell
    interpreter (``bash -c``) and lets it submit by running a sentinel
    echo. We expose only ``bash`` here; submission is handled by the
    agent loop's marker detection on the issued command.

    Tool output is rendered with the official observation_template:
      • ``<returncode>`` always present
      • output < 10000 chars → ``<output>...</output>`` verbatim
      • output ≥ 10000 chars → warning + head[:5000] + elided count + tail[-5000:]

    Args:
        workdir: Absolute path to the repository root directory.
        repo:    GitHub repo identifier (unused, kept for API parity).

    Returns:
        Single-element list containing the bash tool.
    """
    workdir = os.path.abspath(workdir)

    @tool
    def bash(command: str) -> str:
        """Execute a shell command in the repository directory.

        Args:
            command: The shell command to execute.

        Returns:
            Output rendered with mini-swe-agent's observation_template
            (head/tail truncation when long; full text otherwise).
        """
        try:
            python_exe = shutil.which("python3") or shutil.which("python") or "python3"
            if command.startswith("python "):
                command = python_exe + " " + command[7:]
            env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
            result = subprocess.run(
                command, shell=True, cwd=workdir,
                capture_output=True, text=True, timeout=60, env=env,
            )
            output = (result.stdout or "") + (result.stderr or "")
            return _format_minisweagent_observation(
                output, result.returncode)
        except subprocess.TimeoutExpired:
            return _format_minisweagent_observation(
                "", -1,
                exception_info="Command timed out after 60 seconds.")
        except Exception as e:
            return _format_minisweagent_observation(
                "", -1, exception_info=str(e))

    return [bash]
