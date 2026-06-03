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


def _cmd_calibrate(args: argparse.Namespace) -> int:
    """Run an agent against its eval cases, fit a temperature, report ECE."""
    pm = parse_file(args.file)
    module = lower(pm)
    if module.metacognition is None:
        print(f"no tsr:metacognition block in {args.file}")
        return 1
    if not module.eval_cases:
        print(f"no eval cases in {args.file} — need cases to calibrate against")
        return 1

    # MVP calibration: for each eval case, run the target agent and treat the
    # output as a binary correct/incorrect against the case's expected output.
    # Per-sample "logit" is a 2-class synthetic vector — confidence proxied by
    # 1.0 on the chosen class. Real per-token logits arrive when LLM backends
    # surface them; this MVP exercises the math + audit emission.
    target = args.agent or next(iter(module.agents.keys()), None)
    if not target:
        print(f"no agent in {args.file}")
        return 1

    logits_list: list[list[float]] = []
    labels: list[int] = []
    from .interp.eval import run_agent, Refusal
    for case in module.eval_cases:
        try:
            result = run_agent(module, target, initial_beliefs=case.inputs,
                               concurrent=False)
        except Exception:
            result = None
        ok = False
        if isinstance(result, Refusal):
            ok = bool(case.expect_refusal)
        else:
            ok = True
            if case.expect_contains is not None:
                ok = ok and (isinstance(result, str) and case.expect_contains in result)
            if case.expect_equals is not None:
                ok = ok and (result == case.expect_equals)
        # 2-class: class 0 = wrong, class 1 = right. The agent always "picks"
        # class 1 (it answered confidently). The label is whether class 1 was
        # actually correct.
        logits_list.append([0.0, 3.0])  # confident-correct guess
        labels.append(1 if ok else 0)

    from .calibration import calibrate
    report = calibrate(logits_list, labels, n_bins=module.metacognition.n_bins)

    # Emit a governance audit event so the calibration history is queryable.
    from .adapters.audit import record_event
    record_event({
        "seq": 0,
        "agent": target,
        "plan": None,
        "intent": None,
        "action": "calibration:ece",
        "ece_before": report.ece_before,
        "ece_after": report.ece_after,
        "temperature": report.temperature,
        "n_samples": report.n_samples,
        "n_bins": report.n_bins,
    })

    print(f"calibration report for {target}")
    print(f"  n_samples:     {report.n_samples}")
    print(f"  ECE before:    {report.ece_before:.4f}")
    print(f"  ECE after:     {report.ece_after:.4f}")
    print(f"  temperature:   {report.temperature:.4f}")
    return 0


def _cmd_evolve(args: argparse.Namespace) -> int:
    pm = parse_file(args.file)
    module = lower(pm)
    if module.evolve is None:
        print(f"no tsr:evolve block in {args.file}")
        return 1
    from .evolve import evolve as run_evolve
    history = run_evolve(module)
    print(f"target agent: {module.evolve.target_agent}")
    for gen in history:
        print(f"  gen {gen.generation}: best_score={gen.best_score:.3f} "
              f"variant={gen.best_variant_id}")
    return 0


def _cmd_audit_purge(args: argparse.Namespace) -> int:
    from .adapters.audit import purge_operational
    n = purge_operational(
        before=args.before,
        retention_days=args.days,
    )
    print(f"purged {n} operational event(s); governance untouched")
    return 0


def _cmd_audit_assemble_corpus(args: argparse.Namespace) -> int:
    from .training_corpus import assemble_for_skill
    path, n = assemble_for_skill(args.skill)
    print(f"wrote {n} pair(s) → {path}")
    return 0


def _print_facts(rows: list[dict], *, as_json: bool) -> None:
    import json
    for r in rows:
        if as_json:
            print(json.dumps(r))
        else:
            fields = json.dumps(r["fields"])
            if len(fields) > 80:
                fields = fields[:77] + "..."
            print(f"{r['created_at']}  {r['schema']}  {r['agent_id'] or '-'}  {fields}")


def _cmd_facts_list(args: argparse.Namespace) -> int:
    from .adapters.semantic import query_facts, schema_summary
    has_filter = any((args.schema, args.agent, args.plan, args.contains))
    if not has_filter and not args.json:
        summary = schema_summary()
        total = sum(n for _, n in summary)
        for sch, n in summary:
            print(f"  {n:>6}  {sch}")
        print(f"# {total} fact(s) across {len(summary)} schema(s)")
        return 0
    rows = query_facts(
        schema=args.schema,
        agent_id=args.agent,
        plan_id=args.plan,
        contains=args.contains,
        limit=args.limit,
    )
    _print_facts(rows, as_json=args.json)
    if args.count:
        print(f"# {len(rows)} fact(s)", file=sys.stderr)
    return 0


def _cmd_facts_search(args: argparse.Namespace) -> int:
    from .adapters.semantic import query_facts
    rows = query_facts(contains=args.query, schema=args.schema, limit=args.limit)
    _print_facts(rows, as_json=args.json)
    return 0


def _cmd_facts_clear(args: argparse.Namespace) -> int:
    from .adapters.semantic import clear_facts
    try:
        n = clear_facts(
            schema=args.schema,
            agent_id=args.agent,
            before=args.before,
            all=args.all,
        )
    except ValueError:
        print(
            "refusing to clear all facts without --all "
            "(or a --schema/--agent/--before filter)",
            file=sys.stderr,
        )
        return 1
    print(f"cleared {n} fact(s)")
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

    audc = audsub.add_parser("assemble-corpus",
                             help="Build a training-corpus JSONL for a procedural skill")
    audc.add_argument("skill")
    audc.set_defaults(fn=_cmd_audit_assemble_corpus)

    # facts — inspect/clean the memory:semantic store (~/.tessera/semantic.db)
    fct = sub.add_parser("facts", help="Inspect or clean the semantic fact store")
    fctsub = fct.add_subparsers(dest="facts_cmd", required=True)

    fctl = fctsub.add_parser("list", help="List facts, or a schema breakdown when unfiltered")
    fctl.add_argument("--schema")
    fctl.add_argument("--agent")
    fctl.add_argument("--plan")
    fctl.add_argument("--contains", help="substring match against field JSON")
    fctl.add_argument("--limit", type=int, default=100)
    fctl.add_argument("--json", action="store_true", help="emit one JSON object per line")
    fctl.add_argument("--count", action="store_true",
                      help="Print row count to stderr after results")
    fctl.set_defaults(fn=_cmd_facts_list)

    fcts = fctsub.add_parser("search", help="Find facts whose fields contain a substring")
    fcts.add_argument("query")
    fcts.add_argument("--schema")
    fcts.add_argument("--limit", type=int, default=100)
    fcts.add_argument("--json", action="store_true", help="emit one JSON object per line")
    fcts.set_defaults(fn=_cmd_facts_search)

    fctc = fctsub.add_parser("clear", help="Delete facts (needs a filter or --all)")
    fctc.add_argument("--schema")
    fctc.add_argument("--agent")
    fctc.add_argument("--before", help="ISO timestamp — delete facts created before it")
    fctc.add_argument("--all", action="store_true", help="Delete every fact (required to wipe)")
    fctc.set_defaults(fn=_cmd_facts_clear)

    # evolve
    ev = sub.add_parser("evolve", help="Run the tsr:evolve block in a file")
    ev.add_argument("file")
    ev.set_defaults(fn=_cmd_evolve)

    # calibrate
    cb = sub.add_parser("calibrate",
                        help="Fit a temperature scaler against the file's eval cases (research substrate A1)")
    cb.add_argument("file")
    cb.add_argument("--agent", help="Target agent (default: first declared)")
    cb.set_defaults(fn=_cmd_calibrate)

    # version
    verp = sub.add_parser("version")
    verp.set_defaults(fn=_cmd_version)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
