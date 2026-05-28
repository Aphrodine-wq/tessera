"""Tessera CLI — parse, verify, emit, run; plus vault and substrate commands."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .adapters.aeon import verify_sir_text
from .adapters.obsidian import scaffold_agent, scan_vault
from .interp.eval import run_agent
from .parser.module import parse_file
from .sir.build import lower
from .sir.emit import emit_module
from .substrate_docs import render_text as render_substrates
from .verify.passes import run_local


def _compile_run(file: str, *, emit_sir: str | None = None, use_aeon: bool = False,
                 run: str | None = None,
                 sets: list[str] | None = None, audit: str | None = None,
                 sequential: bool = False,
                 train: bool = False) -> int:
    pm = parse_file(file)
    module = lower(pm)
    sir_text = emit_module(module)

    if emit_sir:
        Path(emit_sir).write_text(sir_text)
        print(f"wrote SIR → {emit_sir}")

    local = run_local(module)
    remote = verify_sir_text(sir_text) if use_aeon else []
    diagnostics = local + remote
    for d in diagnostics:
        print(d)

    errors = [d for d in diagnostics if d.severity == "error"]
    if errors:
        print(f"\n{len(errors)} error(s)")
        return 1

    if train:
        from .training import train_all_trainable
        paths = train_all_trainable(module)
        if not paths:
            print("no trainable models found in this file")
        else:
            for p in paths:
                print(f"trained checkpoint → {p}")

    if run:
        from .interp.eval import World
        beliefs = dict(kv.split("=", 1) for kv in (sets or []))
        world = World(module=module, concurrent=not sequential)
        result = run_agent(module, run, initial_beliefs=beliefs, world=world,
                           concurrent=not sequential)
        print(f"\n{run}() = {result!r}")
        if audit:
            import json
            with open(audit, "w") as fh:
                for ev in world.audit:
                    fh.write(json.dumps(ev.to_dict()) + "\n")
            print(f"wrote audit trace ({len(world.audit)} events) → {audit}")
    return 0


def _cmd_compile(args: argparse.Namespace) -> int:
    return _compile_run(
        args.file,
        emit_sir=args.emit_sir,
        use_aeon=args.aeon,
        run=args.run,
        sets=args.set,
        audit=args.audit,
        sequential=args.sequential,
        train=args.train,
    )


def _cmd_vault_scan(args: argparse.Namespace) -> int:
    scan = scan_vault(args.vault)
    print(f"VAULT: {scan.vault_root}")
    print(f"Found {len(scan.agents)} agent(s){f' ({len(scan.skipped)} skipped)' if scan.skipped else ''}\n")
    for a in scan.agents:
        print(f"  {a.agent_name}  ←  {a.vault_relative}")
        print(f"      substrates: {', '.join(a.substrates)}")
        if a.capabilities_requested:
            print(f"      capabilities: {', '.join(a.capabilities_requested)}")
        if a.prompts:
            print(f"      prompts: {', '.join(a.prompts)}")
        if a.tools:
            print(f"      tools: {', '.join(a.tools)}")
        if a.neural_models:
            print(f"      models: {', '.join(a.neural_models)}")
        print()
    if scan.skipped and args.verbose:
        print(f"--- skipped ({len(scan.skipped)}) ---")
        for path, reason in scan.skipped:
            try:
                rel = path.relative_to(scan.vault_root)
            except ValueError:
                rel = path
            print(f"  {rel} — {reason}")
    return 0


def _cmd_vault_run(args: argparse.Namespace) -> int:
    return _compile_run(
        args.file,
        emit_sir=None,
        use_aeon=args.aeon,
        run=args.agent,
        sets=args.set,
    )


def _cmd_vault_new(args: argparse.Namespace) -> int:
    path = scaffold_agent(args.file, args.agent, template=args.template,
                          overwrite=args.force)
    print(f"scaffolded {args.template} agent {args.agent!r} → {path}")
    return 0


def _cmd_substrates(_args: argparse.Namespace) -> int:
    print(render_substrates())
    return 0


def _cmd_providers(args: argparse.Namespace) -> int:
    from .adapters.llm import list_providers
    rows = list_providers()
    print(f"{len(rows)} providers supported\n")
    print(f"  {'id':14s} {'kind':7s} {'schema':14s} {'env var':30s} {'model env'}")
    print(f"  {'-'*14} {'-'*7} {'-'*14} {'-'*30} {'-'*32}")
    for r in rows:
        print(f"  {r['id']:14s} {r['kind']:7s} {r['schema']:14s} "
              f"{r['env']:30s} {r['model_env']}")
    if args.check:
        import os
        print("\n--- reachability check ---")
        for r in rows:
            env = r["env"]
            has_key = (env and env != "(none)" and os.environ.get(env))
            local = (r["kind"] == "local")
            mark = "✓" if (has_key or local) else " "
            why = "key present" if has_key else ("local" if local else "no key")
            print(f"  [{mark}] {r['id']:14s}  {why}")
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    pm = parse_file(args.file)
    module = lower(pm)
    if not module.eval_cases:
        print(f"no eval cases declared in {args.file}")
        return 0
    # Pick the first declared agent as the target unless --agent specified.
    target = args.agent or next(iter(module.agents.keys()), None)
    if not target:
        print(f"no agent declared in {args.file}")
        return 1

    passed = 0
    failed = 0
    from .interp.eval import Refusal
    for case in module.eval_cases:
        try:
            result = run_agent(module, target, initial_beliefs=case.inputs)
        except Exception as e:
            result = f"<exception: {e}>"
        ok = True
        reasons: list[str] = []
        if case.expect_contains is not None:
            if not (isinstance(result, str) and case.expect_contains in result):
                ok = False
                reasons.append(f"missing expected substring {case.expect_contains!r}")
        if case.expect_equals is not None:
            if result != case.expect_equals:
                ok = False
                reasons.append(f"expected {case.expect_equals!r}, got {result!r}")
        if case.expect_refusal:
            if not isinstance(result, Refusal):
                ok = False
                reasons.append(f"expected refusal, got {result!r}")
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {case.name}")
        if not ok:
            for r in reasons:
                print(f"           {r}")
        if ok:
            passed += 1
        else:
            failed += 1
    print(f"\n{passed}/{passed + failed} cases passed")
    return 0 if failed == 0 else 1


def _cmd_audit_query(args: argparse.Namespace) -> int:
    from .adapters.audit import query_events
    import json
    rows = query_events(
        agent=args.agent,
        intent=args.intent,
        action=args.action,
        since=args.since,
        until=args.until,
        limit=args.limit,
        tier=args.tier,
    )
    for r in rows:
        print(json.dumps(r))
    if args.count:
        print(f"# {len(rows)} event(s)", file=sys.stderr)
    return 0


def _cmd_audit_purge(args: argparse.Namespace) -> int:
    from .adapters.audit import purge_operational
    n = purge_operational(
        before=args.before,
        retention_days=args.days,
    )
    print(f"purged {n} operational event(s); governance untouched")
    return 0


def _cmd_version(_args: argparse.Namespace) -> int:
    print(f"tessera {__version__}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tessera")
    sub = p.add_subparsers(dest="cmd", required=True)

    # compile
    cp = sub.add_parser("compile", help="Parse, verify, optionally emit/run a .t.md file")
    cp.add_argument("file")
    cp.add_argument("--emit-sir", metavar="OUT")
    cp.add_argument("--aeon", action="store_true")
    cp.add_argument("--sequential", action="store_true",
                    help="Disable concurrent actor scheduling (default is concurrent)")
    cp.add_argument("--train", action="store_true",
                    help="Train every `trainable` neural model and write checkpoints")
    cp.add_argument("--run", metavar="AGENT")
    cp.add_argument("--set", action="append")
    cp.add_argument("--audit", metavar="OUT", help="write the run's audit trace as JSONL")
    cp.set_defaults(fn=_cmd_compile)

    # vault — Obsidian adapter
    vp = sub.add_parser("vault", help="Obsidian vault commands")
    vsub = vp.add_subparsers(dest="vault_cmd", required=True)

    vscan = vsub.add_parser("scan", help="Scan an Obsidian vault for Tessera agents")
    vscan.add_argument("vault")
    vscan.add_argument("-v", "--verbose", action="store_true")
    vscan.set_defaults(fn=_cmd_vault_scan)

    vrun = vsub.add_parser("run", help="Run an agent at a vault path")
    vrun.add_argument("file")
    vrun.add_argument("--agent", required=True)
    vrun.add_argument("--set", action="append")
    vrun.add_argument("--aeon", action="store_true")
    vrun.set_defaults(fn=_cmd_vault_run)

    vnew = vsub.add_parser("new", help="Scaffold a new agent file in the vault")
    vnew.add_argument("file")
    vnew.add_argument("--agent", required=True)
    vnew.add_argument("--template", default="basic",
                      choices=["basic", "llm", "journal"])
    vnew.add_argument("--force", action="store_true",
                      help="Overwrite if file exists")
    vnew.set_defaults(fn=_cmd_vault_new)

    # substrates
    spr = sub.add_parser("substrates", help="Print all substrate categories in English")
    spr.set_defaults(fn=_cmd_substrates)

    # providers
    pp = sub.add_parser("providers", help="List supported LLM providers")
    pp.add_argument("--check", action="store_true",
                    help="Probe each provider for reachability / configured keys")
    pp.set_defaults(fn=_cmd_providers)

    # eval
    ep = sub.add_parser("eval", help="Run declared eval cases against agent")
    ep.add_argument("file")
    ep.add_argument("--agent", help="Override which agent to run cases against")
    ep.set_defaults(fn=_cmd_eval)

    # audit — query the persistent audit store
    aud = sub.add_parser("audit", help="Query the persistent audit store")
    audsub = aud.add_subparsers(dest="audit_cmd", required=True)
    audq = audsub.add_parser("query", help="Filter events by agent/intent/action/time")
    audq.add_argument("--agent")
    audq.add_argument("--intent")
    audq.add_argument("--action", help="substring or LIKE pattern")
    audq.add_argument("--since", help="ISO timestamp (>=)")
    audq.add_argument("--until", help="ISO timestamp (<=)")
    audq.add_argument("--limit", type=int, default=100)
    audq.add_argument("--tier", choices=["governance", "operational"],
                      help="Restrict to one tier (default: both)")
    audq.add_argument("--count", action="store_true",
                      help="Print row count to stderr after results")
    audq.set_defaults(fn=_cmd_audit_query)

    audp = audsub.add_parser("purge", help="Drop operational events older than a cutoff")
    audp.add_argument("--before", help="ISO timestamp (default: now - retention_days)")
    audp.add_argument("--days", type=int,
                      help="Override TESSERA_AUDIT_RETENTION_DAYS (default 30)")
    audp.set_defaults(fn=_cmd_audit_purge)

    # version
    verp = sub.add_parser("version")
    verp.set_defaults(fn=_cmd_version)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
