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

### `tsr:dual_process` (research 4.1) — SHIPPED (engine; substrate decl pending)

- **References:**
  - Kahneman, D. (2011). *Thinking, Fast and Slow.* Farrar, Straus and Giroux.
  - Evans, J. St. B. T., Stanovich, K. E. (2013). *Dual-process theories of higher cognition: advancing the debate.* Perspectives on Psychological Science, 8(3), 223–241.
- **Operationalized claim:** A router can pick fast (cached / pattern-match) vs slow (deliberative) based on confidence + budget + irreversibility, with the chosen mode audit-emitted per action.
- **What we do NOT claim:** That cognition is genuinely discrete-dual. Evans & Stanovich (2013) acknowledge the abstraction's edges.

### `tsr:counterfactual` (research 4.2) — SHIPPED (engine; substrate decl pending)

- **References:**
  - Lewis, D. (1973). *Counterfactuals.* Harvard University Press.
  - Halpern, J. Y. (2016). *Actual Causality.* MIT Press.
  - Pearl, J. (2009). *Causality* (2nd ed.), Ch. 7.
- **Operationalized claim:** Deterministic structural counterfactuals via Pearl's ABDUCTION → ACTION → PREDICTION recipe over a declared causal DAG; inconsistent observations return (None, None) signalling non-identifiability.
- **What we do NOT claim:** Stochastic counterfactual identifiability. MVP is deterministic structural equations.

### `tsr:abductive` (research 4.3) — SHIPPED (engine; substrate decl pending)

- **References:**
  - Peirce, C. S. (1903). *Pragmatism as a Principle and Method of Right Thinking.* Harvard lectures.
  - Lipton, P. (2004). *Inference to the Best Explanation* (2nd ed.). Routledge.
  - Douven, I. (2017). *Abduction.* Stanford Encyclopedia of Philosophy.
- **Operationalized claim:** Ranked hypotheses by posterior = prior × likelihood × parsimony; below-threshold winner returns None (anti-overconfidence).
- **What we do NOT claim:** That "best" captures Lipton's LOVELINESS dimension. We rank by likeliness only; loveliness is a follow-up.



### `tsr:causal` (research D1)

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
