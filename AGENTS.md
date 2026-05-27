# Agent Instructions

Read [`admm_nxcore/docs/README.md`](admm_nxcore/docs/README.md) first. The canonical project
documentation is in [`admm_nxcore/docs/`](admm_nxcore/docs/).

Key files:

- [`admm_nxcore/docs/architecture.md`](admm_nxcore/docs/architecture.md): current repeated on-board MPC
  architecture.
- [`admm_nxcore/docs/testing.md`](admm_nxcore/docs/testing.md): host, board, Ethernet, profiling, and
  closed-loop test commands.
- [`admm_nxcore/docs/profiling.md`](admm_nxcore/docs/profiling.md): profiling modes and report fields.
- [`admm_nxcore/docs/development_notes.md`](admm_nxcore/docs/development_notes.md): remote workflow,
  shared-board rules, debugging notes, and known caveats.

Do not use stale root documentation files; they were removed during docs
cleanup. Keep new project documentation under `admm_nxcore/docs/` unless there is a strong
reason for a root-level file.

For board/debug work, keep instrumentation simple and avoid adding extra
indirection unless it removes real duplication. Do not broad-kill Python
processes on the shared board; identify exact processes first.
