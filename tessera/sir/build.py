"""Lower ParsedModule → SIR Module.

Grammar (MVP-expanded for the researcher example):

  Logic:
    fn name(arg: T, ...) -> T = expr

  Workspace:
    workspace Name {
      capacity: 1
      arbiter: highest_salience
      contenders: [a, b, c]
    }

  Agents (one or more per block):
    agent Name {
      beliefs:
        @policy name: Type
      intentions:
        plan plan_name { stmt; stmt; ... }
    }

  Plan statements:
    let name = expr
    return expr
    send refExpr msgExpr
    broadcast (expr, salience=N) to WorkspaceName

  Expressions:
    literal | ident | f(args) | a + b | spawn AgentName with [Cap, Cap]
                              | recv from refExpr

Anything richer fails fast with a clear error — the language grows incrementally.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from ..parser.module import ParsedModule, SubstrateBlock
from .nodes import (
    AutonomyDecl, Effect, EpisodicEventDecl, EthicsDecl, EthicsPrinciple,
    EvalCaseDecl, IntentDecl, KnowledgeSchemaDecl, Module, Node,
    BayesianDeclSIR, BayesianLikelihoodSpec, BayesianVarSpec,
    CausalDAGDecl, EvolveDecl, MetacognitionDecl, NeuralModelDecl, Op, PolicyDecl, PromptDecl, Region, SkillDecl, ToolDecl,
    TraitDecl, WorkspaceDecl,
)


class SyntaxFail(Exception):
    pass


# --- micro-tokenizer ---------------------------------------------------------

_TOKEN_RE = re.compile(
    r"""
    \s+                            |   # whitespace
    "[^"]*"                        |   # string literal
    [A-Za-z_][A-Za-z_0-9]*         |   # ident
    \d+(?:\.\d+)?                  |   # number
    ->|::|==|!=|<=|>=              |   # multi-char ops
    [(){}\[\],:;=+\-*/<>!?@.|]         # single-char punctuation
    """,
    re.VERBOSE,
)


def _tokens(src: str) -> list[str]:
    out: list[str] = []
    for m in _TOKEN_RE.finditer(src):
        tok = m.group()
        if tok.strip():
            out.append(tok)
    return out


# --- expression parser → SIR nodes ------------------------------------------


@dataclass
class _Parser:
    toks: list[str]
    pos: int = 0

    def peek(self, n: int = 0) -> str | None:
        return self.toks[self.pos + n] if self.pos + n < len(self.toks) else None

    def eat(self, expected: str | None = None) -> str:
        if self.pos >= len(self.toks):
            raise SyntaxFail(f"unexpected end (expected {expected!r})")
        tok = self.toks[self.pos]
        if expected is not None and tok != expected:
            raise SyntaxFail(f"expected {expected!r}, got {tok!r}")
        self.pos += 1
        return tok

    def done(self) -> bool:
        return self.pos >= len(self.toks)


_CMP_OPS = {"==", "!=", "<", "<=", ">", ">="}


def _emit_expr(p: _Parser, region: Region, block: SubstrateBlock) -> str:
    """Parse a single expression and emit nodes; return the id of the value node."""
    return _emit_logical(p, region, block)


def _emit_logical(p: _Parser, region: Region, block: SubstrateBlock) -> str:
    """Lowest precedence — `and` / `or` as left-associative chain."""
    lhs = _emit_comparison(p, region, block)
    while p.peek() in ("and", "or"):
        op = p.eat()
        rhs = _emit_comparison(p, region, block)
        n = region.add(Node(
            op=Op.BinOp,
            inputs=[lhs, rhs],
            attributes={"op": op},
            substrate="logic",
            output_type="Bool",
            provenance=block.span,
        ))
        lhs = n.id
    return lhs


def _emit_comparison(p: _Parser, region: Region, block: SubstrateBlock) -> str:
    lhs = _emit_additive(p, region, block)
    if p.peek() in _CMP_OPS:
        op = p.eat()
        rhs = _emit_additive(p, region, block)
        n = region.add(Node(
            op=Op.BinOp,
            inputs=[lhs, rhs],
            attributes={"op": op},
            substrate="logic",
            output_type="Bool",
            provenance=block.span,
        ))
        return n.id
    return lhs


def _emit_additive(p: _Parser, region: Region, block: SubstrateBlock) -> str:
    lhs = _emit_primary(p, region, block)
    while p.peek() in ("+", "-"):
        op = p.eat()
        rhs = _emit_primary(p, region, block)
        n = region.add(Node(
            op=Op.BinOp,
            inputs=[lhs, rhs],
            attributes={"op": op},
            substrate="logic",
            output_type="any",
            provenance=block.span,
        ))
        lhs = n.id
    return lhs


def _split_args(src: str) -> list[str]:
    """Split a comma-separated argument list at top level (depth-1 parens)."""
    out: list[str] = []
    depth = 0
    buf: list[str] = []
    for ch in src:
        if ch in "([{":
            depth += 1
            buf.append(ch)
        elif ch in ")]}":
            depth -= 1
            buf.append(ch)
        elif ch == "," and depth == 0:
            out.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    if buf:
        out.append("".join(buf).strip())
    return [x for x in out if x]


def _emit_primary(p: _Parser, region: Region, block: SubstrateBlock) -> str:
    tok = p.peek()
    if tok is None:
        raise SyntaxFail("unexpected end in expression")

    # `recall EventName`
    if tok == "recall":
        p.eat("recall")
        event_name = p.eat()
        n = region.add(Node(
            op=Op.EM_Query,
            attributes={"event": event_name},
            substrate="memory:episodic",
            effects={Effect.mem_episodic_r.value},
            output_type="List",
            provenance=block.span,
        ))
        return n.id

    # `lookup SchemaName where field == value`  OR  `lookup SchemaName`
    if tok == "lookup":
        p.eat("lookup")
        schema_name = p.eat()
        where_field: str | None = None
        value_id: str | None = None
        if p.peek() == "where":
            p.eat("where")
            where_field = p.eat()
            # accept either `==` or `=` for SQL-flavored ergonomics
            if p.peek() == "==":
                p.eat("==")
            elif p.peek() == "=":
                p.eat("=")
            value_id = _emit_primary(p, region, block)
        attrs = {"schema": schema_name}
        if where_field:
            attrs["where_field"] = where_field
        n = region.add(Node(
            op=Op.SM_Search,
            inputs=[value_id] if value_id else [],
            attributes=attrs,
            substrate="memory:semantic",
            effects={Effect.mem_semantic_r.value},
            output_type="List",
            provenance=block.span,
        ))
        return n.id

    # `spawn AgentName with [Cap, Cap]`
    if tok == "spawn":
        p.eat("spawn")
        agent_name = p.eat()
        caps: list[str] = []
        if p.peek() == "with":
            p.eat("with")
            p.eat("[")
            if p.peek() != "]":
                caps.append(p.eat())
                while p.peek() == ",":
                    p.eat(",")
                    caps.append(p.eat())
            p.eat("]")
        n = region.add(Node(
            op=Op.Spawn,
            attributes={"agent": agent_name, "capabilities": caps},
            substrate="agent",
            effects={Effect.spawn.value},
            capability_requires=set(caps),  # parent must hold caps it grants
            output_type="AgentRef",
            provenance=block.span,
        ))
        return n.id

    # `recv from refExpr [timeout Ns]`
    if tok == "recv":
        p.eat("recv")
        p.eat("from")
        ref_id = _emit_primary(p, region, block)
        timeout_s: float | None = None
        if p.peek() == "timeout":
            p.eat("timeout")
            tok_val = p.eat()
            # Accept "5s", "5.0s", "30", "30s"
            v = tok_val.rstrip("s")
            try:
                timeout_s = float(v)
            except ValueError:
                raise SyntaxFail(f"recv timeout expects a number (got {tok_val!r})")
        attrs: dict = {}
        if timeout_s is not None:
            attrs["timeout_s"] = timeout_s
        n = region.add(Node(
            op=Op.Recv,
            inputs=[ref_id],
            attributes=attrs,
            substrate="agent",
            effects={Effect.msg_recv.value},
            output_type="any",
            provenance=block.span,
        ))
        return n.id

    # string literal
    if tok.startswith('"'):
        p.eat()
        n = region.add(Node(
            op=Op.Const,
            attributes={"value": tok[1:-1], "type": "String"},
            substrate="logic",
            output_type="String",
            provenance=block.span,
        ))
        return n.id

    # number literal
    if tok[0].isdigit():
        p.eat()
        val: int | float = float(tok) if "." in tok else int(tok)
        n = region.add(Node(
            op=Op.Const,
            attributes={"value": val, "type": "Int" if isinstance(val, int) else "Float"},
            substrate="logic",
            output_type="Int" if isinstance(val, int) else "Float",
            provenance=block.span,
        ))
        return n.id

    # identifier (possibly followed by call)
    if tok[0].isalpha() or tok[0] == "_":
        name = p.eat()
        if p.peek() == "(":
            p.eat("(")
            args: list[str] = []
            if p.peek() != ")":
                args.append(_emit_expr(p, region, block))
                while p.peek() == ",":
                    p.eat(",")
                    args.append(_emit_expr(p, region, block))
            p.eat(")")
            callee = region.add(Node(
                op=Op.Const,
                attributes={"value": name, "type": "FnRef"},
                substrate="logic",
                output_type="FnRef",
                provenance=block.span,
            ))
            applied = region.add(Node(
                op=Op.Apply,
                inputs=[callee.id, *args],
                attributes={"callee": name},
                substrate="logic",
                output_type="any",
                provenance=block.span,
            ))
            return applied.id
        # bare identifier → BeliefRead (resolved at interp time)
        bind = region.add(Node(
            op=Op.BeliefRead,
            attributes={"name": name},
            substrate="agent" if region.name.startswith(("agent:", "plan:")) else "logic",
            output_type="any",
            effects={Effect.mem_working_r.value},
            provenance=block.span,
        ))
        return bind.id

    if tok == "(":
        p.eat("(")
        inner = _emit_expr(p, region, block)
        p.eat(")")
        return inner

    raise SyntaxFail(f"unexpected token {tok!r}")


# --- block-level handlers ---------------------------------------------------


_FN_RE = re.compile(
    r"\s*fn\s+(\w+)\s*\(([^)]*)\)\s*->\s*(\w+)\s*=\s*(.+)$",
    re.DOTALL,
)


def _lower_logic(block: SubstrateBlock, mod: Module) -> None:
    for raw_line in block.body.strip().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("//"):
            continue
        m = _FN_RE.match(line)
        if not m:
            raise SyntaxFail(f"logic line not recognized: {line!r}")
        name, params_src, ret_t, body_src = m.groups()
        params = []
        for chunk in (params_src.split(",") if params_src.strip() else []):
            pname, _, ptype = chunk.partition(":")
            params.append((pname.strip(), ptype.strip()))
        region = Region(name=f"fn:{name}", params=params, return_type=ret_t.strip())
        p = _Parser(_tokens(body_src))
        val_id = _emit_expr(p, region, block)
        region.add(Node(
            op=Op.Return,
            inputs=[val_id],
            substrate="logic",
            output_type=ret_t.strip(),
            provenance=block.span,
        ))
        mod.regions.append(region)
        mod.functions[name] = region


_WORKSPACE_HEAD_RE = re.compile(r"workspace\s+(\w+)\s*\{")
_AGENT_HEAD_RE = re.compile(r"agent\s+(\w+)(?:\s+intends\s+(\w+))?\s*\{")
_PLAN_HEAD_RE = re.compile(r"plan\s+(\w+)(?:\s+serves\s+(\w+))?\s*\{")
_NOTICE_HEAD_RE = re.compile(r"notice\s+when\s+(.+?)\s*\{")
_BELIEF_LINE_RE = re.compile(r"@(\w+)\s+(\w+)\s*:\s*(\w+)")
_TRAITS_LINE_RE = re.compile(r"^\s*traits\s*:\s*\[([^\]]*)\]", re.MULTILINE)


def _balanced_extract(src: str, start_idx: int) -> tuple[str, int]:
    """Given src[start_idx] == '{', return (inner, end_idx_after_close)."""
    assert src[start_idx] == "{"
    depth = 0
    for i in range(start_idx, len(src)):
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                return src[start_idx + 1 : i], i + 1
    raise SyntaxFail("unbalanced braces")


def _lower_workspace(block: SubstrateBlock, mod: Module) -> None:
    src = block.body
    m = _WORKSPACE_HEAD_RE.search(src)
    if not m:
        raise SyntaxFail("expected `workspace Name { ... }`")
    name = m.group(1)
    brace = src.index("{", m.end() - 1)
    body, _ = _balanced_extract(src, brace)

    decl = WorkspaceDecl(name=name)
    for raw in body.splitlines():
        line = raw.strip()
        if not line or line.startswith("//"):
            continue
        if line.startswith("capacity"):
            decl.capacity = int(line.partition(":")[2].strip())
        elif line.startswith("arbiter"):
            decl.arbiter = line.partition(":")[2].strip()
        elif line.startswith("contenders"):
            inner = line.partition(":")[2].strip()
            if inner.startswith("[") and inner.endswith("]"):
                decl.contenders = [s.strip() for s in inner[1:-1].split(",") if s.strip()]
        elif line.startswith("gwt_bottleneck"):
            decl.gwt_bottleneck = int(line.partition(":")[2].strip())
        elif line.startswith("track_ignition"):
            val = line.partition(":")[2].strip().lower()
            decl.track_ignition = (val == "true")
    mod.workspaces[name] = decl


def _lower_agents_in_block(block: SubstrateBlock, mod: Module) -> None:
    """Parse one or more `agent Name { ... }` blocks from one substrate block."""
    src = block.body
    cursor = 0
    found_any = False
    while True:
        m = _AGENT_HEAD_RE.search(src, cursor)
        if not m:
            break
        found_any = True
        brace = src.index("{", m.end() - 1)
        body, end = _balanced_extract(src, brace)
        _lower_one_agent(m.group(1), body, block, mod, intent=m.group(2))
        cursor = end
    if not found_any:
        raise SyntaxFail("expected at least one `agent Name { ... }` in agent block")


def _lower_one_agent(agent_name: str, body: str, block: SubstrateBlock, mod: Module,
                     intent: str | None = None) -> None:
    agent_region = Region(name=f"agent:{agent_name}", intent=intent)
    notice_nodes: list[Node] = []
    plan_nodes: list[Node] = []

    # Attach cognitive traits declared on the agent (`traits: [a, b]`). Without
    # this the line is silently swallowed by belief-absorption and never wired.
    tl = _TRAITS_LINE_RE.search(body)
    if tl:
        agent_region.trait_names = [t.strip() for t in tl.group(1).split(",") if t.strip()]

    cursor = 0
    while cursor < len(body):
        plan_m = _PLAN_HEAD_RE.search(body, cursor)
        notice_m = _NOTICE_HEAD_RE.search(body, cursor)
        next_m = None
        kind = None
        if plan_m and (notice_m is None or plan_m.start() <= notice_m.start()):
            next_m, kind = plan_m, "plan"
        elif notice_m:
            next_m, kind = notice_m, "notice"

        if next_m is None:
            _absorb_belief_lines(body[cursor:], agent_region, block)
            break

        pre = body[cursor:next_m.start()]
        _absorb_belief_lines(pre, agent_region, block)

        if kind == "plan":
            plan_name = next_m.group(1)
            plan_intent = next_m.group(2) or intent  # `serves X`, else inherit agent's
            plan_brace = body.index("{", next_m.end() - 1)
            plan_body, end_idx = _balanced_extract(body, plan_brace)
            plan_region = Region(name=f"plan:{plan_name}", parent=agent_region.id,
                                 intent=plan_intent)
            _lower_plan_body(plan_body, plan_region, block)
            mod.regions.append(plan_region)
            plan_nodes.append(Node(
                op=Op.IntentionCommit,
                attributes={"plan": plan_name, "region": plan_region.id},
                substrate="agent",
                effects={Effect.intention_commit.value},
                provenance=block.span,
            ))
            cursor = end_idx
        else:  # notice
            pred_src = next_m.group(1)
            handler_brace = body.index("{", next_m.end() - 1)
            handler_body, end_idx = _balanced_extract(body, handler_brace)

            pred_region = Region(name=f"agent:{agent_name}:notice_pred")
            pp = _Parser(_tokens(pred_src))
            pred_id = _emit_expr(pp, pred_region, block)
            pred_region.add(Node(
                op=Op.Return, inputs=[pred_id],
                substrate="logic", provenance=block.span,
            ))

            handler_region = Region(name=f"agent:{agent_name}:notice_handler")
            _lower_plan_body(handler_body, handler_region, block)
            mod.regions.append(handler_region)

            notice_nodes.append(Node(
                op=Op.Notice_Subscribe,
                attributes={
                    "pred_region": pred_region,
                    "handler_region": handler_region,
                },
                substrate="agent",
                effects={Effect.notice_subscribe.value},
                provenance=block.span,
            ))
            cursor = end_idx

    # Order matters: notices registered first so they're active during plans;
    # plans last so the agent region's final value is the intention's return.
    for n in notice_nodes:
        agent_region.add(n)
    for n in plan_nodes:
        agent_region.add(n)

    mod.regions.append(agent_region)
    mod.agents[agent_name] = agent_region


def _absorb_belief_lines(src: str, region: Region, block: SubstrateBlock) -> None:
    in_beliefs = False
    for raw in src.splitlines():
        line = raw.strip()
        if not line or line.startswith("//"):
            continue
        if line.endswith(":"):
            in_beliefs = line.startswith("beliefs")
            continue
        if not in_beliefs:
            continue
        bm = _BELIEF_LINE_RE.search(line)
        if not bm:
            continue
        policy, name, typ = bm.groups()
        region.add(Node(
            op=Op.BeliefRead,
            attributes={"name": name, "policy": f"@{policy}", "type": typ, "declared": True},
            substrate="agent",
            effects={Effect.mem_working_r.value},
            provenance=block.span,
        ))


_UNTIL_HEAD_RE = re.compile(r"^until\s+(.+?)\s*\{\s*$")


def _lower_plan_body(src: str, region: Region, block: SubstrateBlock) -> None:
    """Parse statements in a plan body.

    Statements:
      let name = expr
      send refExpr msgExpr
      broadcast (expr, salience=N) to WorkspaceName
      log EventName(args)
      until <pred> { body }
      return expr

    After lowering, runs a parallelism analysis: consecutive `let` statements
    whose right-hand sides don't reference any earlier name in the run are
    flagged as a parallel group via WM_Write.attributes['parallel_group'].
    The interpreter dispatches them concurrently when a thread executor is
    available.
    """
    pending_value: str | None = None
    lines = src.splitlines()
    i = 0
    while i < len(lines):
        raw = lines[i]
        line = raw.strip()
        if not line or line.startswith("//"):
            i += 1
            continue

        # until-loop — collect lines until matching brace
        um = _UNTIL_HEAD_RE.match(line)
        if um:
            pred_src = um.group(1)
            # find matching close brace
            depth = 1
            body_lines = []
            j = i + 1
            while j < len(lines) and depth > 0:
                inner = lines[j]
                depth += inner.count("{") - inner.count("}")
                if depth == 0:
                    break
                body_lines.append(inner)
                j += 1
            body_src = "\n".join(body_lines)
            # Build the loop node: stores pred + body region id
            pred_region = Region(name=region.name + ":until_pred")
            pp = _Parser(_tokens(pred_src))
            pred_id = _emit_expr(pp, pred_region, block)
            pred_region.add(Node(
                op=Op.Return, inputs=[pred_id],
                substrate="logic", provenance=block.span,
            ))
            body_region = Region(name=region.name + ":until_body")
            _lower_plan_body(body_src, body_region, block)
            # Attach both sub-regions to the module via the parent region's
            # attribute so the interp can find them. We stash regions in the
            # node attributes; the interp resolves them.
            region.add(Node(
                op=Op.Until,
                attributes={
                    "pred_region": pred_region,
                    "body_region": body_region,
                    "max_iter": 100,
                },
                substrate="logic",
                provenance=block.span,
            ))
            i = j + 1
            continue

        if line.startswith("let "):
            rest = line[4:]
            name, _, expr_src = rest.partition("=")
            name = name.strip()
            p = _Parser(_tokens(expr_src))
            val_id = _emit_expr(p, region, block)
            region.add(Node(
                op=Op.WM_Write,
                inputs=[val_id],
                attributes={"name": name},
                substrate="agent",
                effects={Effect.mem_working_w.value},
                provenance=block.span,
            ))
            pending_value = val_id
            i += 1
            continue

        if line.startswith("remember "):
            # remember SchemaName(field=value, field=value, ...)
            inner = line[len("remember "):].strip()
            mm = re.match(r"^(\w+)\s*\((.*)\)\s*$", inner)
            if not mm:
                raise SyntaxFail(f"remember statement malformed: {line!r}")
            schema_name = mm.group(1)
            fields_src = mm.group(2)
            field_attrs: dict[str, str] = {}
            arg_ids: list[str] = []
            field_names: list[str] = []
            for chunk in _split_args(fields_src):
                if "=" in chunk:
                    k, _, v = chunk.partition("=")
                    field_names.append(k.strip())
                    p = _Parser(_tokens(v.strip()))
                    arg_ids.append(_emit_expr(p, region, block))
                else:
                    raise SyntaxFail(f"remember field must be key=value: {chunk!r}")
            region.add(Node(
                op=Op.SM_Insert,
                inputs=arg_ids,
                attributes={"schema": schema_name, "fields": field_names},
                substrate="memory:semantic",
                effects={Effect.mem_semantic_w.value},
                provenance=block.span,
            ))
            i += 1
            continue

        if line.startswith("log "):
            # log EventName(arg, arg, ...)
            inner = line[len("log "):].strip()
            m = re.match(r"^(\w+)\s*\((.*)\)\s*$", inner)
            if not m:
                raise SyntaxFail(f"log statement malformed: {line!r}")
            event_name = m.group(1)
            args_src = m.group(2)
            arg_ids: list[str] = []
            if args_src.strip():
                for chunk in _split_args(args_src):
                    p = _Parser(_tokens(chunk))
                    arg_ids.append(_emit_expr(p, region, block))
            region.add(Node(
                op=Op.EM_Append,
                inputs=arg_ids,
                attributes={"event": event_name},
                substrate="memory:episodic",
                effects={Effect.mem_episodic_w.value},
                provenance=block.span,
            ))
            i += 1
            continue

        if line.startswith("send "):
            p = _Parser(_tokens(line[len("send "):]))
            ref_id = _emit_primary(p, region, block)
            msg_id = _emit_expr(p, region, block)
            region.add(Node(
                op=Op.Send,
                inputs=[ref_id, msg_id],
                substrate="agent",
                effects={Effect.msg_send.value},
                provenance=block.span,
            ))
            i += 1
            continue

        if line.startswith("broadcast "):
            # broadcast (expr, salience=N) to WorkspaceName
            inner = line[len("broadcast "):].strip()
            if not inner.startswith("("):
                raise SyntaxFail(f"broadcast must start with '(': {line!r}")
            close = inner.index(")")
            args_src = inner[1:close]
            rest = inner[close + 1:].strip()
            ws_match = re.match(r"to\s+(\w+)\s*$", rest)
            if not ws_match:
                raise SyntaxFail(f"broadcast missing `to WorkspaceName`: {line!r}")
            ws_name = ws_match.group(1)

            # split args on top-level commas
            parts = [a.strip() for a in args_src.split(",")]
            value_src = parts[0]
            salience: float = 0.5
            for part in parts[1:]:
                if "=" in part:
                    k, _, v = part.partition("=")
                    if k.strip() == "salience":
                        salience = float(v.strip())

            p = _Parser(_tokens(value_src))
            val_id = _emit_expr(p, region, block)
            region.add(Node(
                op=Op.Workspace_Broadcast,
                inputs=[val_id],
                attributes={"workspace": ws_name, "salience": salience},
                substrate="memory:workspace",
                effects={Effect.mem_workspace_w.value},
                provenance=block.span,
            ))
            i += 1
            continue

        if line.startswith("return"):
            expr_src = line[len("return"):].strip()
            if expr_src:
                p = _Parser(_tokens(expr_src))
                val_id = _emit_expr(p, region, block)
            else:
                val_id = ""
            ret = Node(
                op=Op.Return,
                substrate="agent",
                provenance=block.span,
            )
            if val_id:
                ret.inputs = [val_id]
            region.add(ret)
            return

        raise SyntaxFail(f"plan statement not recognized: {line!r}")
        i += 1

    if pending_value is not None:
        region.add(Node(
            op=Op.Return,
            inputs=[pending_value],
            substrate="agent",
            provenance=block.span,
        ))

    # Parallelism analysis: tag consecutive WM_Write nodes whose dependencies
    # all live OUTSIDE the group. The interpreter uses this tag to dispatch
    # them via a ThreadPoolExecutor.
    _annotate_parallel_groups(region)


def _annotate_parallel_groups(region: Region) -> None:
    """Mark consecutive let-binding WM_Writes that don't depend on each other."""
    # Walk in order. A "group" is a maximal run of WM_Writes such that no
    # node's input is produced by an earlier node in the group.
    by_id = {n.id: n for n in region.nodes}

    def _produced_in_group(start_idx: int, end_idx: int) -> set[str]:
        out = set()
        for n in region.nodes[start_idx:end_idx]:
            out.add(n.id)
            # Also include sub-nodes produced for this WM_Write's RHS — the
            # parser emits expr nodes BEFORE the WM_Write that consumes them.
        return out

    def _direct_deps(node: Node) -> set[str]:
        deps = set(node.inputs)
        # Pull through pure sub-expressions: collect all ids transitively
        # reachable in the same region (not crossing region boundaries).
        seen = set()
        stack = list(deps)
        while stack:
            nid = stack.pop()
            if nid in seen:
                continue
            seen.add(nid)
            sub = by_id.get(nid)
            if sub is not None:
                stack.extend(sub.inputs)
        return seen

    # Find WM_Write nodes (the let-binding boundaries) and the inclusive
    # range of region nodes that produce each binding's value.
    wm_indices = [i for i, n in enumerate(region.nodes) if n.op is Op.WM_Write]
    if len(wm_indices) < 2:
        return

    # Compute the start index of each WM's "expression nodes" — everything
    # since the previous WM_Write (or start of region).
    expr_ranges = []
    prev_end = 0
    for idx in wm_indices:
        expr_ranges.append((prev_end, idx + 1))  # [start, end) inclusive of WM_Write
        prev_end = idx + 1

    # A WM_Write at index k is parallelizable with the one before it if
    # NEITHER's expression range produces an id consumed by the OTHER's range.
    # We tag each consecutive pair; runs of pair-compatible WMs form a group.
    group_id = 0
    current_group: list[int] = []
    last_produced: set[str] = set()

    for wi, (start, end) in enumerate(expr_ranges):
        # Ids produced within this WM's expression range
        produced_here = {region.nodes[k].id for k in range(start, end)}
        # Ids this WM's range consumes
        consumed_here: set[str] = set()
        for k in range(start, end):
            consumed_here |= _direct_deps(region.nodes[k])
        # Does this expression range consume anything produced by the
        # CURRENT group?  If so, we close the group and start a new one.
        if consumed_here & last_produced:
            if len(current_group) >= 2:
                _tag_group(region, current_group, group_id)
                group_id += 1
            current_group = []
            last_produced = set()
        current_group.append(wi)
        last_produced |= produced_here

    if len(current_group) >= 2:
        _tag_group(region, current_group, group_id)


def _tag_group(region: Region, wm_positions: list[int], group_id: int) -> None:
    """Annotate the WM_Write nodes in this group with parallel metadata."""
    wm_nodes = [n for n in region.nodes if n.op is Op.WM_Write]
    for pos in wm_positions:
        n = wm_nodes[pos]
        n.attributes = dict(n.attributes)
        n.attributes["parallel_group"] = group_id
        n.attributes["parallel_size"] = len(wm_positions)


# --- entry point ------------------------------------------------------------


_PROMPT_RE = re.compile(
    r'\s*prompt\s+(\w+)\s*\(([^)]*)\)\s*->\s*(\w+)\s*=\s*"(.+)"\s*$',
    re.DOTALL,
)

_TOOL_RE = re.compile(
    r"\s*tool\s+(\w+)\s*\(([^)]*)\)\s*->\s*(\w+)\s+from\s+([\w.]+)(?:\s+via\s+(\w+))?\s*$",
)

_MODEL_HEAD_RE = re.compile(r"model\s+(\w+)\s*\{")

_EPISODIC_HEAD_RE = re.compile(r"episodic\s*\{")
_EVENT_DECL_RE = re.compile(r"event\s+(\w+)\s*\(([^)]*)\)")

_KNOWLEDGE_HEAD_RE = re.compile(r"knowledge\s*\{")
_SCHEMA_DECL_RE = re.compile(r"schema\s+(\w+)\s*\(([^)]*)\)")

_POLICY_HEAD_RE = re.compile(r"policy\s+(\w+)\s*\{")
_POLICY_FORBID_CONTAINS_RE = re.compile(r'forbid\s+contains\s+"([^"]+)"')
_POLICY_FORBID_MATCH_RE = re.compile(r'forbid\s+match\s+"([^"]+)"')
_POLICY_REQUIRE_CONTAINS_RE = re.compile(r'require\s+contains\s+"([^"]+)"')
# Constraint-logic forms (decision 12). The expression body runs to end of line
# or to the next semicolon. Newlines inside parens are OK.
_POLICY_FORBID_WHEN_RE = re.compile(r'forbid\s+when\s+([^\n;]+)')
_POLICY_PERMIT_WHEN_RE = re.compile(r'permit\s+when\s+([^\n;]+)')

_EVAL_CASE_RE = re.compile(r'case\s+"([^"]+)"\s*\{')
_EVAL_INPUT_RE = re.compile(r'input\s+(\w+)\s*=\s*"([^"]*)"')
_EVAL_EXPECT_CONTAINS_RE = re.compile(r'expect_contains\s*=\s*"([^"]*)"')
_EVAL_EXPECT_EQUALS_RE = re.compile(r'expect_equals\s*=\s*"([^"]*)"')
_EVAL_EXPECT_REFUSAL_RE = re.compile(r'expect_refusal\s*=\s*(true|false)')

_PROCEDURAL_HEAD_RE = re.compile(r"procedural\s*\{")
# `skill name(params) -> T from <kind> Y` optionally followed by
# `promote_to: neural { threshold: N }` on the same or next non-blank line.
_SKILL_DECL_RE = re.compile(
    r"skill\s+(\w+)\s*\(([^)]*)\)\s*->\s*(\w+)\s+from\s+(model|prompt|tool|fn)\s+(\w+)"
    r"(?:\s+promote_to\s*:\s*(\w+)(?:\s*\{\s*threshold\s*:\s*(\d+)\s*\})?)?"
)


def _parse_typed_params(src: str) -> list[tuple[str, str]]:
    out = []
    for chunk in (src.split(",") if src.strip() else []):
        pname, _, ptype = chunk.partition(":")
        out.append((pname.strip(), ptype.strip() or "Any"))
    return out


def _lower_prompt(block: SubstrateBlock, mod: Module) -> None:
    for raw in block.body.strip().splitlines():
        line = raw.strip()
        if not line or line.startswith("//"):
            continue
        m = _PROMPT_RE.match(line)
        if not m:
            raise SyntaxFail(f"prompt line not recognized: {line!r}")
        name, params_src, ret_t, template = m.groups()
        mod.prompts[name] = PromptDecl(
            name=name,
            params=_parse_typed_params(params_src),
            return_type=ret_t.strip(),
            template=template,
        )


def _lower_tool(block: SubstrateBlock, mod: Module) -> None:
    for raw in block.body.strip().splitlines():
        line = raw.strip()
        if not line or line.startswith("//"):
            continue
        m = _TOOL_RE.match(line)
        if not m:
            raise SyntaxFail(f"tool line not recognized: {line!r}")
        name, params_src, ret_t, import_path, invoke_method = m.groups()
        mod.tools[name] = ToolDecl(
            name=name,
            params=_parse_typed_params(params_src),
            return_type=ret_t.strip(),
            import_path=import_path.strip(),
            invoke_method=(invoke_method or "invoke").strip(),
        )


def _lower_episodic(block: SubstrateBlock, mod: Module) -> None:
    src = block.body
    m = _EPISODIC_HEAD_RE.search(src)
    if not m:
        raise SyntaxFail("expected `episodic { event Name(fields) }`")
    brace = src.index("{", m.end() - 1)
    body, _ = _balanced_extract(src, brace)
    for em in _EVENT_DECL_RE.finditer(body):
        name, params_src = em.groups()
        mod.episodic_events[name] = EpisodicEventDecl(
            name=name, fields=_parse_typed_params(params_src),
        )


def _lower_policy(block: SubstrateBlock, mod: Module) -> None:
    src = block.body
    m = _POLICY_HEAD_RE.search(src)
    if not m:
        raise SyntaxFail("expected `policy Name { ... }`")
    name = m.group(1)
    brace = src.index("{", m.end() - 1)
    body, _ = _balanced_extract(src, brace)
    rules: list[tuple[str, dict]] = []
    for rm in _POLICY_FORBID_CONTAINS_RE.finditer(body):
        rules.append(("forbid_contains", {"needle": rm.group(1)}))
    for rm in _POLICY_FORBID_MATCH_RE.finditer(body):
        rules.append(("forbid_match", {"pattern": rm.group(1)}))
    for rm in _POLICY_REQUIRE_CONTAINS_RE.finditer(body):
        rules.append(("require_contains", {"needle": rm.group(1)}))
    # New constraint-logic forms — only parse the `forbid contains` / `forbid match`
    # forms above first so a `forbid when contains_pii(value)` doesn't get
    # mistakenly matched by the older regexes.
    from ..policy_lang import parse as _parse_policy_expr, PolicySyntaxError
    for rm in _POLICY_FORBID_WHEN_RE.finditer(body):
        try:
            expr = _parse_policy_expr(rm.group(1).strip())
        except PolicySyntaxError as e:
            raise SyntaxFail(f"policy {name!r} forbid-when expression: {e}")
        rules.append(("forbid_when", {"expr": expr, "src": rm.group(1).strip()}))
    for rm in _POLICY_PERMIT_WHEN_RE.finditer(body):
        try:
            expr = _parse_policy_expr(rm.group(1).strip())
        except PolicySyntaxError as e:
            raise SyntaxFail(f"policy {name!r} permit-when expression: {e}")
        rules.append(("permit_when", {"expr": expr, "src": rm.group(1).strip()}))
    mod.policies[name] = PolicyDecl(name=name, rules=rules)


def _lower_eval(block: SubstrateBlock, mod: Module) -> None:
    src = block.body
    # Each `case "name" { ... }` is one EvalCaseDecl
    pos = 0
    while True:
        m = _EVAL_CASE_RE.search(src, pos)
        if not m:
            break
        case_name = m.group(1)
        brace = src.index("{", m.end() - 1)
        case_body, end_idx = _balanced_extract(src, brace)
        inputs: dict[str, str] = {}
        for im in _EVAL_INPUT_RE.finditer(case_body):
            inputs[im.group(1)] = im.group(2)
        expect_contains = None
        expect_equals = None
        expect_refusal = False
        cm = _EVAL_EXPECT_CONTAINS_RE.search(case_body)
        if cm:
            expect_contains = cm.group(1)
        em = _EVAL_EXPECT_EQUALS_RE.search(case_body)
        if em:
            expect_equals = em.group(1)
        rm = _EVAL_EXPECT_REFUSAL_RE.search(case_body)
        if rm:
            expect_refusal = (rm.group(1) == "true")
        mod.eval_cases.append(EvalCaseDecl(
            name=case_name,
            inputs=inputs,
            expect_contains=expect_contains,
            expect_equals=expect_equals,
            expect_refusal=expect_refusal,
        ))
        pos = end_idx


def _lower_procedural(block: SubstrateBlock, mod: Module) -> None:
    src = block.body
    m = _PROCEDURAL_HEAD_RE.search(src)
    if not m:
        raise SyntaxFail("expected `procedural { skill ... from ... }`")
    brace = src.index("{", m.end() - 1)
    body, _ = _balanced_extract(src, brace)
    for sm in _SKILL_DECL_RE.finditer(body):
        name, params_src, ret_t, kind, target, promote_to, threshold = sm.groups()
        mod.skills[name] = SkillDecl(
            name=name,
            params=_parse_typed_params(params_src),
            return_type=ret_t.strip(),
            binds_to_kind=kind,
            binds_to_name=target,
            promote_to=promote_to or None,
            promote_threshold=int(threshold) if threshold else 100,
        )


_TRAIT_HEAD_RE = re.compile(r"trait\s+(\w+)\s*\{")
# trigger stops at newline OR ';' so both the multi-line and inline
# (`trigger: x; behavior: '...'`) forms parse correctly.
_TRAIT_TRIGGER_RE = re.compile(r"trigger\s*:\s*([^\n;]+)")
_TRAIT_BEHAVIOR_RE = re.compile(r"behavior\s*:\s*(['\"])(.*?)\1", re.DOTALL)
_TRAIT_PRIORITY_RE = re.compile(r"priority\s*:\s*([0-9.]+)")
_TRAIT_SCOPE_RE = re.compile(r"scope\s*:\s*(['\"]?)(\w+)\1")


def _lower_traits(block: SubstrateBlock, mod: Module) -> None:
    src = block.body
    cursor = 0
    while True:
        m = _TRAIT_HEAD_RE.search(src, cursor)
        if not m:
            break
        name = m.group(1)
        brace = src.index("{", m.end() - 1)
        tbody, end = _balanced_extract(src, brace)

        tm = _TRAIT_TRIGGER_RE.search(tbody)
        trigger = [t.strip() for t in tm.group(1).split("|")] if tm else []
        trigger = [t for t in (x.strip().rstrip(";").strip() for x in trigger) if t]
        bm = _TRAIT_BEHAVIOR_RE.search(tbody)
        behavior = bm.group(2).strip() if bm else ""
        pm = _TRAIT_PRIORITY_RE.search(tbody)
        priority = float(pm.group(1)) if pm else 0.5
        sm = _TRAIT_SCOPE_RE.search(tbody)
        scope = sm.group(2).strip() if sm else "per_call"

        if not trigger or not behavior:
            raise SyntaxFail(f"trait {name!r} requires both `trigger` and `behavior`")
        mod.traits[name] = TraitDecl(
            name=name, trigger=trigger, behavior=behavior,
            priority=priority, scope=scope,
        )
        cursor = end


_INTENT_HEAD_RE = re.compile(r"intent\s+(\w+)\s*\{")
_INTENT_GOAL_RE = re.compile(r"goal\s*:\s*(['\"])(.*?)\1", re.DOTALL)
_INTENT_WHY_RE = re.compile(r"why\s*:\s*(['\"])(.*?)\1", re.DOTALL)
_INTENT_SUCCESS_RE = re.compile(r"success\s*:\s*([^\n]+)")
_INTENT_FORBIDDEN_RE = re.compile(r"forbidden\s*:\s*\[([^\]]*)\]")


def _lower_intent(block: SubstrateBlock, mod: Module) -> None:
    src = block.body
    cursor = 0
    while True:
        m = _INTENT_HEAD_RE.search(src, cursor)
        if not m:
            break
        name = m.group(1)
        brace = src.index("{", m.end() - 1)
        ibody, end = _balanced_extract(src, brace)

        gm = _INTENT_GOAL_RE.search(ibody)
        goal = gm.group(2).strip() if gm else ""
        wm = _INTENT_WHY_RE.search(ibody)
        why = wm.group(2).strip() if wm else ""
        fm = _INTENT_FORBIDDEN_RE.search(ibody)
        forbidden = [s.strip() for s in fm.group(1).split(",") if s.strip()] if fm else []
        # success: may be a bracketed list or a single bare predicate per line.
        success: list[str] = []
        for sm in _INTENT_SUCCESS_RE.finditer(ibody):
            raw = sm.group(1).strip().rstrip(";").strip()
            if raw.startswith("[") and raw.endswith("]"):
                success.extend(s.strip() for s in raw[1:-1].split(",") if s.strip())
            elif raw:
                success.append(raw)

        if not goal:
            raise SyntaxFail(f"intent {name!r} requires a `goal`")
        mod.intents[name] = IntentDecl(
            name=name, goal=goal, success=success, forbidden=forbidden, why=why,
        )
        cursor = end


_ETHICS_HEAD_RE = re.compile(r"ethics\s*\{")
_PRINCIPLE_RE = re.compile(r"principle\s+(\w+)\s*\{([^}]*)\}", re.DOTALL)
_PRINCIPLE_RULE_RE = re.compile(r"rule\s*:\s*(['\"])(.*?)\1", re.DOTALL)
_PRINCIPLE_WEIGHT_RE = re.compile(r"weight\s*:\s*([0-9.]+)")
_ON_CONFLICT_RE = re.compile(r"on_conflict\s*:\s*(\w+)")
_ON_VIOLATION_RE = re.compile(r"on_violation\s*:\s*(\w+)")


def _lower_ethics(block: SubstrateBlock, mod: Module) -> None:
    src = block.body
    m = _ETHICS_HEAD_RE.search(src)
    if not m:
        raise SyntaxFail("expected `ethics { ... }`")
    brace = src.index("{", m.end() - 1)
    body, _ = _balanced_extract(src, brace)

    principles: list[EthicsPrinciple] = []
    for pm in _PRINCIPLE_RE.finditer(body):
        pname, pbody = pm.group(1), pm.group(2)
        rm = _PRINCIPLE_RULE_RE.search(pbody)
        rule = rm.group(2).strip() if rm else ""
        wm = _PRINCIPLE_WEIGHT_RE.search(pbody)
        weight = float(wm.group(1)) if wm else 0.5
        principles.append(EthicsPrinciple(name=pname, rule=rule, weight=weight))

    cm = _ON_CONFLICT_RE.search(body)
    vm = _ON_VIOLATION_RE.search(body)
    mod.ethics = EthicsDecl(
        principles=principles,
        on_conflict=cm.group(1) if cm else "highest_weight",
        on_violation=vm.group(1) if vm else "refuse",
    )


_AUTONOMY_HEAD_RE = re.compile(r"autonomy\s*\{")
_AUTO_LEVEL_RE = re.compile(r"level\s*:\s*(\w+)")
_AUTO_APPROVAL_RE = re.compile(r"require_approval\s*:\s*\[([^\]]*)\]")
_AUTO_ESCALATE_RE = re.compile(r"escalate_when\s*:\s*(['\"])(.*?)\1", re.DOTALL)
_AUTO_BOUNDARY_RE = re.compile(r"boundary\s*:\s*(['\"])(.*?)\1", re.DOTALL)


def _lower_autonomy(block: SubstrateBlock, mod: Module) -> None:
    src = block.body
    m = _AUTONOMY_HEAD_RE.search(src)
    if not m:
        raise SyntaxFail("expected `autonomy { ... }`")
    brace = src.index("{", m.end() - 1)
    body, _ = _balanced_extract(src, brace)

    lm = _AUTO_LEVEL_RE.search(body)
    am = _AUTO_APPROVAL_RE.search(body)
    em = _AUTO_ESCALATE_RE.search(body)
    bm = _AUTO_BOUNDARY_RE.search(body)
    mod.autonomy = AutonomyDecl(
        level=lm.group(1) if lm else "propose",
        require_approval=[s.strip() for s in am.group(1).split(",") if s.strip()] if am else [],
        escalate_when=em.group(2).strip() if em else "",
        boundary=bm.group(2).strip() if bm else "",
    )


_BAYES_HEAD_RE = re.compile(r"bayesian\s*\{")
# var name: [v1, v2, ...] prior [p1, p2, ...]
_BAYES_VAR_RE = re.compile(
    r"var\s+(\w+)\s*:\s*\[([^\]]+)\]\s*prior\s*\[([^\]]+)\]"
)
# likelihood observed given latent { latent_v -> observed_v: p; ... }
_BAYES_LIK_HEAD_RE = re.compile(
    r"likelihood\s+(\w+)\s+given\s+(\w+)\s*\{"
)
_BAYES_LIK_ROW_RE = re.compile(
    r"(\w+)\s*->\s*(\w+)\s*:\s*([0-9.]+)"
)


def _lower_bayesian(block: SubstrateBlock, mod: Module) -> None:
    src = block.body
    m = _BAYES_HEAD_RE.search(src)
    if not m:
        raise SyntaxFail("expected `bayesian { ... }`")
    brace = src.index("{", m.end() - 1)
    body, _ = _balanced_extract(src, brace)

    decl = BayesianDeclSIR()
    for vm in _BAYES_VAR_RE.finditer(body):
        name = vm.group(1)
        values = [s.strip() for s in vm.group(2).split(",")]
        prior = [float(s.strip()) for s in vm.group(3).split(",")]
        if len(values) != len(prior):
            raise SyntaxFail(
                f"bayesian var {name!r}: values/prior length mismatch"
            )
        if abs(sum(prior) - 1.0) > 1e-6:
            raise SyntaxFail(
                f"bayesian var {name!r}: prior must sum to 1.0 (got {sum(prior):.4f})"
            )
        decl.variables.append(BayesianVarSpec(name=name, values=values, prior=prior))

    # Parse likelihood blocks
    cursor = 0
    while True:
        lm = _BAYES_LIK_HEAD_RE.search(body, cursor)
        if not lm:
            break
        observed = lm.group(1)
        latent = lm.group(2)
        l_brace = body.index("{", lm.end() - 1)
        l_body, end = _balanced_extract(body, l_brace)
        rows: dict[str, dict[str, float]] = {}
        for rm in _BAYES_LIK_ROW_RE.finditer(l_body):
            latent_v, observed_v, p_str = rm.group(1), rm.group(2), rm.group(3)
            rows.setdefault(latent_v, {})[observed_v] = float(p_str)
        decl.likelihoods.append(BayesianLikelihoodSpec(
            latent=latent, observed=observed, rows=rows,
        ))
        cursor = end

    mod.bayesian = decl


_CAUSAL_HEAD_RE = re.compile(r"causal\s+(\w+)\s*\{")
_CAUSAL_VAR_RE = re.compile(r"var\s+(\w+)\s*:\s*\w+")
_CAUSAL_EDGE_RE = re.compile(r"edge\s+(\w+)\s*->\s*(\w+)")


def _lower_causal(block: SubstrateBlock, mod: Module) -> None:
    src = block.body
    m = _CAUSAL_HEAD_RE.search(src)
    if not m:
        raise SyntaxFail("expected `causal Name { ... }`")
    name = m.group(1)
    brace = src.index("{", m.end() - 1)
    body, _ = _balanced_extract(src, brace)
    variables = [vm.group(1) for vm in _CAUSAL_VAR_RE.finditer(body)]
    edges = [(em.group(1), em.group(2)) for em in _CAUSAL_EDGE_RE.finditer(body)]
    # Verify every edge references declared variables
    declared = set(variables)
    for p, c in edges:
        if p not in declared or c not in declared:
            raise SyntaxFail(
                f"causal DAG {name!r}: edge {p}->{c} references undeclared var"
            )
    decl = CausalDAGDecl(name=name, variables=variables, edges=edges)
    # Cycle check — a directed cycle violates the DAG promise (Pearl 2009 §1.2).
    from ..causal import CausalDAG as _RuntimeDAG
    runtime_view = _RuntimeDAG(name=name, variables=variables, edges=edges)
    if runtime_view.has_cycle():
        raise SyntaxFail(
            f"causal DAG {name!r} has a cycle — DAGs must be acyclic (Pearl 2009 §1.2)"
        )
    mod.causal_dags[name] = decl


_METACOG_HEAD_RE = re.compile(r"metacognition\s*\{")
_METACOG_TEMP_RE = re.compile(r"temperature\s*:\s*([0-9.]+)")
_METACOG_BINS_RE = re.compile(r"n_bins\s*:\s*(\d+)")
_METACOG_TRACK_RE = re.compile(r"track_ece\s*:\s*(true|false)")


def _lower_metacognition(block: SubstrateBlock, mod: Module) -> None:
    src = block.body
    m = _METACOG_HEAD_RE.search(src)
    if not m:
        raise SyntaxFail("expected `metacognition { ... }`")
    brace = src.index("{", m.end() - 1)
    body, _ = _balanced_extract(src, brace)
    decl = MetacognitionDecl()
    tm = _METACOG_TEMP_RE.search(body)
    if tm:
        decl.temperature = float(tm.group(1))
    bm = _METACOG_BINS_RE.search(body)
    if bm:
        decl.n_bins = int(bm.group(1))
    km = _METACOG_TRACK_RE.search(body)
    if km:
        decl.track_ece = (km.group(1) == "true")
    mod.metacognition = decl


_EVOLVE_HEAD_RE = re.compile(r"evolve\s+(\w+)\s*\{")
_EVOLVE_POP_RE = re.compile(r"population\s*:\s*(\d+)")
_EVOLVE_GENS_RE = re.compile(r"generations\s*:\s*(\d+)")
_EVOLVE_MUTATE_RE = re.compile(r"mutate\s*:\s*\[([^\]]*)\]")
_EVOLVE_FITNESS_RE = re.compile(r"fitness\s*:\s*(\w+)")


def _lower_evolve(block: SubstrateBlock, mod: Module) -> None:
    src = block.body
    m = _EVOLVE_HEAD_RE.search(src)
    if not m:
        raise SyntaxFail("expected `evolve AgentName { ... }`")
    target = m.group(1)
    brace = src.index("{", m.end() - 1)
    body, _ = _balanced_extract(src, brace)
    pop = int(_EVOLVE_POP_RE.search(body).group(1)) if _EVOLVE_POP_RE.search(body) else 4
    gens = int(_EVOLVE_GENS_RE.search(body).group(1)) if _EVOLVE_GENS_RE.search(body) else 3
    fit_m = _EVOLVE_FITNESS_RE.search(body)
    fitness = fit_m.group(1) if fit_m else "eval_pass_rate"
    mut_m = _EVOLVE_MUTATE_RE.search(body)
    mutate_targets = (
        [s.strip() for s in mut_m.group(1).split(",") if s.strip()]
        if mut_m else ["prompts"]
    )
    mod.evolve = EvolveDecl(
        target_agent=target,
        population=pop,
        generations=gens,
        fitness=fitness,
        mutate_targets=mutate_targets,
    )


def _lower_knowledge(block: SubstrateBlock, mod: Module) -> None:
    src = block.body
    m = _KNOWLEDGE_HEAD_RE.search(src)
    if not m:
        raise SyntaxFail("expected `knowledge { schema Name(fields) }`")
    brace = src.index("{", m.end() - 1)
    body, _ = _balanced_extract(src, brace)
    persistent = block.attrs.get("persistent", "true").lower() != "false"
    for sm in _SCHEMA_DECL_RE.finditer(body):
        name, params_src = sm.groups()
        mod.knowledge_schemas[name] = KnowledgeSchemaDecl(
            name=name,
            fields=_parse_typed_params(params_src),
            persistent=persistent,
        )


_TRAINABLE_HEAD_RE = re.compile(r"trainable\s*\{")
_OPT_RE = re.compile(r"optimizer\s*:\s*(adam|sgd)(?:\s*\(\s*lr\s*=\s*([0-9.eE+-]+)\s*\))?")
_EPOCHS_RE = re.compile(r"epochs\s*:\s*(\d+)")
_LOSS_RE = re.compile(r"loss\s*:\s*(\w+)")
_BATCH_RE = re.compile(r"batch_size\s*:\s*(\d+)")


def _lower_neural(block: SubstrateBlock, mod: Module) -> None:
    src = block.body
    m = _MODEL_HEAD_RE.search(src)
    if not m:
        raise SyntaxFail("expected `model Name { ... }`")
    name = m.group(1)
    brace = src.index("{", m.end() - 1)
    body, body_end = _balanced_extract(src, brace)
    layers: list[dict] = []
    for raw in body.splitlines():
        line = raw.strip().rstrip(";")
        if not line or line.startswith("//"):
            continue
        # `linear in=784 out=128` or `relu` or `sigmoid` / `tanh`
        parts = line.split()
        kind = parts[0].lower()
        layer: dict = {"kind": kind}
        for kv in parts[1:]:
            if "=" in kv:
                k, _, v = kv.partition("=")
                layer[k.strip()] = int(v.strip()) if v.strip().lstrip("-").isdigit() else v.strip()
        layers.append(layer)
    decl = NeuralModelDecl(name=name, layers=layers)
    # Optional `trainable { ... }` clause after the model body.
    tail = src[body_end:]
    tm = _TRAINABLE_HEAD_RE.search(tail)
    if tm:
        t_brace = tail.index("{", tm.end() - 1)
        t_body, _ = _balanced_extract(tail, t_brace)
        decl.trainable = True
        opt_m = _OPT_RE.search(t_body)
        if opt_m:
            decl.optimizer = opt_m.group(1)
            if opt_m.group(2):
                decl.learning_rate = float(opt_m.group(2))
        ep_m = _EPOCHS_RE.search(t_body)
        if ep_m:
            decl.epochs = int(ep_m.group(1))
        ls_m = _LOSS_RE.search(t_body)
        if ls_m:
            decl.loss = ls_m.group(1)
        bs_m = _BATCH_RE.search(t_body)
        if bs_m:
            decl.batch_size = int(bs_m.group(1))
    mod.neural_models[name] = decl


def lower(pm: ParsedModule) -> Module:
    from .optimize import optimize as _optimize_sir
    mod = Module(name=pm.frontmatter.get("agent", "unnamed"))
    for block in pm.blocks:
        if block.substrate == "logic":
            _lower_logic(block, mod)
        elif block.substrate == "agent":
            _lower_agents_in_block(block, mod)
        elif block.substrate == "memory:workspace":
            _lower_workspace(block, mod)
        elif block.substrate == "prompt":
            _lower_prompt(block, mod)
        elif block.substrate == "tool":
            _lower_tool(block, mod)
        elif block.substrate == "neural":
            _lower_neural(block, mod)
        elif block.substrate == "memory:episodic":
            _lower_episodic(block, mod)
        elif block.substrate == "memory:semantic":
            _lower_knowledge(block, mod)
        elif block.substrate == "memory:procedural":
            _lower_procedural(block, mod)
        elif block.substrate == "traits":
            _lower_traits(block, mod)
        elif block.substrate == "intent":
            _lower_intent(block, mod)
        elif block.substrate == "ethics":
            _lower_ethics(block, mod)
        elif block.substrate == "autonomy":
            _lower_autonomy(block, mod)
        elif block.substrate == "evolve":
            _lower_evolve(block, mod)
        elif block.substrate == "metacognition":
            _lower_metacognition(block, mod)
        elif block.substrate == "causal":
            _lower_causal(block, mod)
        elif block.substrate == "bayesian":
            _lower_bayesian(block, mod)
        elif block.substrate == "policy":
            _lower_policy(block, mod)
        elif block.substrate == "eval":
            _lower_eval(block, mod)
        else:
            stub = Region(name=f"stub:{block.substrate}")
            stub.add(Node(
                op=Op.Const,
                attributes={"unsupported_substrate": block.substrate},
                substrate=block.substrate,
                provenance=block.span,
            ))
            mod.regions.append(stub)

    # Propagate frontmatter capabilities to every region of the named root agent
    # and any plans it owns. Child agents (spawned via Spawn) get caps at runtime.
    root_caps = set(pm.frontmatter.get("capabilities_requested") or [])
    root_agent = pm.frontmatter.get("agent")
    if root_caps and root_agent:
        for region in mod.regions:
            if region.name == f"agent:{root_agent}" or region.name.startswith("plan:") or region.name.startswith("fn:"):
                region.capabilities_in_scope |= root_caps

    # Run SIR optimization passes (constant folding + DCE) before handing off
    # to verify/emit/interp. Pure SIR-to-SIR transform; never changes behavior.
    _optimize_sir(mod)

    return mod
