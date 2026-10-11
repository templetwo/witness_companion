# Validation receipts

Date: 2026-10-11. Seat: Codex, codex-astra-studio, gpt-6-astra.

Implementation HEAD SHA: `49625a916ddaedc93125f2d1d63f133c07a786ac`.
Branch: `prep/mcp-tool-server`, based on the existing local `origin/master`.
The source and tests at this SHA are exactly those used for the passing run.
A following receipt-only commit records these files and RESULT.md. Resolve the
final branch tip with `git rev-parse HEAD`; the report does not embed its own
commit hash.

Working directory for every command below:
`/Users/tony_studio/Projects/witness-companion-prep/wc-mcp`.

Tests used `/opt/homebrew/bin/python3.12`, Python 3.12.12. This preferred
interpreter was present, so no fallback was needed. Exact test command:

```sh
/opt/homebrew/bin/python3.12 -m unittest discover -s tests -v > receipts/unittest.log 2>&1
```

Exit code: 0. Result: 47 tests passed, 0 failed, 0 errors, 0 skipped.
Full stdout and stderr are in [unittest.log](unittest.log).
The suite includes ten shared contract tests run both in-process and in
subprocesses over stdio, plus guard, failure, import and transport tests.
All temporary test files were confined to worktree subdirectories and removed.
The file-access audit test checks local note writes and rejects external file,
socket and subprocess access during the note call.

Exact requested Python 3.10 grammar check:

```sh
python3 -c "import ast,pathlib;[ast.parse(p.read_text(),feature_version=(3,10)) for p in pathlib.Path('witness_mcp').rglob('*.py')]"
```

The default `python3` resolves to `/usr/bin/python3`, Python 3.9.6. That command
exited 0. Because this default interpreter predates 3.10, the same grammar check
was also run with Python 3.12.12:

```sh
/opt/homebrew/bin/python3.12 -c "import ast,pathlib;[ast.parse(p.read_text(),feature_version=(3,10)) for p in pathlib.Path('witness_mcp').rglob('*.py')]"
```

It exited 0. [py310_syntax.log](py310_syntax.log) records both commands,
interpreter versions, their complete output (the checks themselves were silent)
and exit codes. This is a syntax check, not execution under Python 3.10.12.

The first full run had 46 passes and one failure. Its full output is retained
in [unittest-initial.log](unittest-initial.log). The note-access test incorrectly
rejected opening its own local directory. Its predicate was corrected to allow
the directory itself as well as its children. A focused reproduction confirmed
the mistaken assertion before the final full passing run. Review also tightened
large-number validation so extreme integer values produce bounded refusals or
invalid-parameter errors instead of numeric conversion exceptions.

Nothing here proves OpenClaw compatibility or physical behaviour. No installed
OpenClaw instance, model backend, network service, camera, microphone, speaker,
robot or physical stop was exercised. No packages were installed or changes
pushed. The real service adapter is deliberately unwired.
