# Tessera — Per-Substrate Citations

Every research-grounded substrate Tessera ships carries a primary
reference. This file is the canonical citation table — verifiable
DOIs / arXiv IDs / book ISBNs, the key claim each substrate
operationalizes, and what we explicitly do NOT claim from that work.

When the substrate vocabulary is shown via `tessera substrates`, each
entry links here.

## Calibration / metacognition

### `tsr:metacognition` (research A1)

- **Reference:** Guo, C., Pleiss, G., Sun, Y., Weinberger, K. Q. (2017).
  *On calibration of modern neural networks.* Proceedings of the 34th
  International Conference on Machine Learning (ICML).
  arXiv:1706.04599.
- **Operationalized claim:** A single scalar temperature T applied to
  logits before softmax is competitive with more complex calibration
  methods at reducing Expected Calibration Error (ECE).
- **What we do NOT claim:** Calibration on training distribution
  transfers to out-of-distribution inputs. Calibration ≠ accuracy.

### `tsr:bayesian` (research A2)

- **Reference:** Blei, D. M., Kucukelbir, A., McAuliffe, J. D. (2017).
  *Variational inference: a review for statisticians.* Journal of the
  American Statistical Association, 112(518), 859–877.
  arXiv:1601.00670.
- **Operationalized claim:** Discrete-variable Bayes update yields the
  exact posterior given prior and likelihood. The cited paper covers
  variational inference for the continuous follow-up.
- **What we do NOT claim:** Variational inference is shipped (the MVP
  uses exact discrete inference). The citation marks the planned
  direction.

## Cognitive architecture

### `tsr:gwt` extension to `memory:workspace` (research B1)

- **References:**
  - Baars, B. J. (1988). *A Cognitive Theory of Consciousness.*
    Cambridge University Press. ISBN 978-0521427432.
  - Dehaene, S. (2014). *Consciousness and the Brain.* Viking Press.
    ISBN 978-0670025435.
- **Operationalized claims:**
  - Baars: the global workspace has a limited capacity (attention
    bottleneck). We ship `gwt_bottleneck`.
  - Dehaene: ignition is a measurable broadcast event with bandwidth
    and timing. We ship `gwt:ignition` audit events with bandwidth,
    winner_salience, cycle index.
- **What we do NOT claim:** A workspace ignition event in software
  produces the biological ignition signature, nor that either
  resolves phenomenal consciousness.

### `tsr:rl_loop` (research B3) — planned

- **Reference:** Sutton, R. S., Barto, A. G. (2018). *Reinforcement
  Learning: An Introduction* (2nd ed.). MIT Press.
  ISBN 978-0262039246.

### `tsr:active_inference` (research B2) — planned

- **References:**
  - Friston, K. (2010). *The free-energy principle: a unified brain
    theory?* Nature Reviews Neuroscience, 11(2), 127–138.
  - Friston, K., FitzGerald, T., Rigoli, F., Schwartenbeck, P.,
    Pezzulo, G. (2017). *Active inference: a process theory.* Neural
    Computation, 29(1), 1–49.

## Consciousness-adjacent measurable substrates

### `tsr:iit` (research C1) — SHIPPED

- **References:**
  - Tononi, G. (2004). *An information integration theory of
    consciousness.* BMC Neuroscience, 5, 42.
  - Tononi, G., Boly, M., Massimini, M., Koch, C. (2016). *Integrated
    information theory: from consciousness to its physical substrate.*
    Nature Reviews Neuroscience, 17(7), 450–461.
  - Mediano, P. A. M., Rosas, F. E., Bor, D., Seth, A. K., Barrett,
    A. B. (2022). *The strength of weak integrated information theory.*
    Trends in Cognitive Sciences, 26(8), 646–655.
- **Operationalized claim (when shipped):** φ* approximations measure
  *information integration* — a structural property of the system's
  belief/intention graph. Mediano et al. (2022) explicitly call φ a
  "signature of dynamical complexity," not phenomenality.
- **What we do NOT claim:** φ > 0 means consciousness. IIT is one of
  many contested theories. Tessera's substrate refuses any block that
  draws the inference φ > 0 → conscious.

### `tsr:ast` (research C2)

- **References:**
  - Graziano, M. S. A. (2013). *Consciousness and the Social Brain.*
    Oxford University Press. ISBN 978-0199928644.
  - Graziano, M. S. A. (2019). *Rethinking Consciousness: a scientific
    theory of subjective experience.* W. W. Norton.
    ISBN 978-0393652611.
  - Graziano, M. S. A., Guterstam, A., Bio, B. J., Wilterson, A. I.
    (2020). *Toward a standard model of consciousness: reconciling the
    attention schema and global workspace theories.* Philosophy and
    the Mind Sciences, 1, II.5.
- **Operationalized claim:** Self-reports query an internal model of
  attention; the model can be accurate or drifted. The fidelity
  metric (`reports matching truth` / `total reports`) is the
  honesty signal.
- **What we do NOT claim:** Maintaining an attention schema produces
  subjective experience. The substrate ships the measure; the
  metaphysics is left for philosophy of mind.

### `tsr:tom` (research C3) — SHIPPED

- **References:**
  - Premack, D., Woodruff, G. (1978). *Does the chimpanzee have a
    theory of mind?* Behavioral and Brain Sciences, 1(4), 515–526.
  - Baker, C. L., Saxe, R., Tenenbaum, J. B. (2009). *Action
    understanding as inverse planning.* Cognition, 113(3), 329–349.
  - Rabinowitz, N. C., Perbet, F., Song, H. F., Zhang, C., Eslami,
    S. M. A., Botvinick, M. (2018). *Machine theory of mind.* ICML.
    arXiv:1802.07740.

### `tsr:welfare` (research C4) — SHIPPED

- **Reference:** Birch, J. (2020). *The search for invertebrate
  consciousness.* Noûs, 54(1), 133–155.
- **Operationalized claim (when shipped):** A markers-based behavioral
  commitment — given certain measurable signatures (φ thresholds, AST
  fidelity drops, broadcast-bandwidth collapse), gate the agent's
  inputs to avoid driving the markers below a declared threshold.
- **What we do NOT claim:** Welfare = moral status = consciousness.
  Welfare is the BEHAVIORAL commitment to act as if certain markers
  matter; the moral-status question is left to ethicists.

## Reasoning / inference

### `tsr:dual_process` (research 4.1) — SHIPPED (substrate)

- **References:**
  - Kahneman, D. (2011). *Thinking, Fast and Slow.* Farrar, Straus and Giroux.
  - Evans, J. St. B. T., Stanovich, K. E. (2013). *Dual-process theories of higher cognition: advancing the debate.* Perspectives on Psychological Science, 8(3), 223–241.
- **Operationalized claim:** A router can pick fast (cached / pattern-match) vs slow (deliberative) based on confidence + budget + irreversibility, with the chosen mode audit-emitted per action.
- **What we do NOT claim:** That cognition is genuinely discrete-dual. Evans & Stanovich (2013) acknowledge the abstraction's edges.

### `counterfactual(...)` (research 4.2) — SHIPPED (callable over tsr:causal)

- **References:**
  - Lewis, D. (1973). *Counterfactuals.* Harvard University Press.
  - Halpern, J. Y. (2016). *Actual Causality.* MIT Press.
  - Pearl, J. (2009). *Causality* (2nd ed.), Ch. 7.
- **Operationalized claim:** Deterministic structural counterfactuals via Pearl's ABDUCTION → ACTION → PREDICTION recipe over a declared causal DAG; inconsistent observations return (None, None) signalling non-identifiability.
- **What we do NOT claim:** Stochastic counterfactual identifiability. MVP is deterministic structural equations.

### `abductive(...)` (research 4.3) — SHIPPED (callable)

- **References:**
  - Peirce, C. S. (1903). *Pragmatism as a Principle and Method of Right Thinking.* Harvard lectures.
  - Lipton, P. (2004). *Inference to the Best Explanation* (2nd ed.). Routledge.
  - Douven, I. (2017). *Abduction.* Stanford Encyclopedia of Philosophy.
- **Operationalized claim:** Ranked hypotheses by posterior = prior × likelihood × parsimony; below-threshold winner returns None (anti-overconfidence).
- **What we do NOT claim:** That "best" captures Lipton's LOVELINESS dimension. We rank by likeliness only; loveliness is a follow-up.

### `analogy(...)` (research 4.4) — SHIPPED (callable)

- **References:**
  - Gentner, D. (1983). *Structure-mapping: a theoretical framework for analogy.* Cognitive Science, 7(2), 155–170.
  - Falkenhainer, B., Forbus, K. D., Gentner, D. (1989). *The structure-mapping engine.* Artificial Intelligence, 41(1), 1–63.
- **Operationalized claim:** A greedy structure-mapping search finds an object binding source→target maximizing relational overlap, with a systematicity bonus for higher-arity relations.
- **What we do NOT claim:** Surface/semantic similarity. The engine is symbolic over author-declared relations; it cannot infer relations from prose.

### `tsr:gricean` (research 4.5) — SHIPPED (substrate)

- **References:**
  - Grice, H. P. (1975). *Logic and conversation.* In Cole & Morgan (eds.), Syntax and Semantics 3: Speech Acts. Academic Press, 41–58.
- **Operationalized claim:** An outgoing message can be scored against the four maxims (quantity / quality / relation / manner) with length, evidence, topic, and repetition heuristics; gated maxims refuse on violation.
- **What we do NOT claim:** That the maxims are indefeasible. They are pluggable per agent — irony / indirection / deniability are real exceptions the heuristics don't model.

### `tsr:precaution` (research 4.7) — SHIPPED (substrate)

- **References:**
  - Hansson, S. O. (2003). *Ethical criteria of risk acceptance.* Erkenntnis, 59(3), 291–309.
  - Taleb, N. N. (2012). *Antifragile.* Random House. (Informal antifragility framing.)
- **Operationalized claim:** Under non-trivial tail probability of crossing a harm threshold — especially when irreversible — refuse the action even when expected value is positive; the burden of proof shifts onto the action.
- **What we do NOT claim:** That thresholds are objective. They are author-declared per domain; over-precaution paralyzes, under-precaution defeats the purpose.

### `tsr:moral_foundations` (research 4.9) — SHIPPED (substrate)

- **References:**
  - Haidt, J. (2012). *The Righteous Mind.* Pantheon.
  - Graham, J., Haidt, J., et al. (2013). *Moral Foundations Theory.* Advances in Experimental Social Psychology, 47, 55–130.
- **Operationalized claim:** Six weighted moral axes; an action scoring negative on a weighted axis (> 0.1) is refused — value pluralism rather than one-axis utility.
- **What we do NOT claim:** That MFT is the true structure of moral cognition. It is contested; the substrate treats it as a useful representation, with author-editable foundations.

### `tsr:hindsight` (research 4.10) — SHIPPED (substrate)

- **References:**
  - US Army Combined Arms Center (1993). *A Leader's Guide to After-Action Reviews.* TC 25-20.
  - Argyris, C., Schön, D. (1978). *Organizational Learning.* Addison-Wesley.
  - Fischhoff, B. (1975). *Hindsight ≠ foresight.* JEP: Human Perception and Performance, 1(3), 288–299.
- **Operationalized claim:** On plan completion, compare declared vs applied ethics + intended vs actual outcome; discrepancies become a learning signal feeding tsr:evolve fitness.
- **What we do NOT claim:** That retrospective judgment is unbiased — the review shows prior and posterior separately (Fischhoff) rather than collapsing them.

### `tsr:argumentative` (research 4.12) — SHIPPED (substrate)

- **References:**
  - Mercier, H., Sperber, D. (2011). *Why do humans reason?* Behavioral and Brain Sciences, 34(2), 57–74.
  - Mercier, H., Sperber, D. (2017). *The Enigma of Reason.* Harvard University Press.
- **Operationalized claim:** A critic pass argues against the proposed answer; the counter-argument's strength log-odds-downweights the proposer's confidence, and a below-threshold answer is refused.
- **What we do NOT claim:** That the critic is a complete adversary. The strength heuristic scores declared refutation markers; an LLM critic is the stronger follow-up.

### `tsr:causal` (research D1) — SHIPPED (substrate + callables)

- **References:**
  - Pearl, J. (2009). *Causality: Models, Reasoning, and Inference*
    (2nd ed.). Cambridge University Press. ISBN 978-0521895606.
  - Pearl, J., Mackenzie, D. (2018). *The Book of Why.* Basic Books.
    ISBN 978-0465097609.
- **Operationalized claim:** Pearl's backdoor criterion identifies
  whether the causal effect P(Y | do(X)) is computable from
  observational data, and finds the smallest admissible adjustment
  set Z.
- **What we do NOT claim:** The DAG matches reality. The substrate
  enforces the math given the declared DAG; verifying the DAG
  against the world is the author's job.

### `tsr:concept_formation` (research D2) — planned

- **References:**
  - Rosch, E. (1978). *Principles of categorization.* In Cognition and
    Categorization (Rosch & Lloyd, eds.). Lawrence Erlbaum.
  - Nosofsky, R. M. (1986). *Attention, similarity, and the
    identification-categorization relationship.* Journal of
    Experimental Psychology: General, 115(1), 39–57.

## Memory / retrieval

### `tsr:semantic_embedding` (research E1) — planned

- **References:**
  - Gao, T., Yao, X., Chen, D. (2021). *SimCSE: simple contrastive
    learning of sentence embeddings.* EMNLP 2021. arXiv:2104.08821.
  - Karpukhin, V., Oğuz, B., Min, S., Lewis, P., Wu, L., Edunov, S.,
    Chen, D., Yih, W. (2020). *Dense passage retrieval for open-domain
    question answering.* EMNLP 2020. arXiv:2004.04906.
  - Reimers, N., Gurevych, I. (2019). *Sentence-BERT: sentence
    embeddings using Siamese BERT-networks.* EMNLP 2019.
    arXiv:1908.10084.

## Philosophical anchors

These two are cited throughout `PHILOSOPHY.md`:

- Chalmers, D. J. (1995). *Facing up to the problem of consciousness.*
  Journal of Consciousness Studies, 2(3), 200–219.
- Block, N. (1995). *On a confusion about a function of consciousness.*
  Behavioral and Brain Sciences, 18(2), 227–247.
