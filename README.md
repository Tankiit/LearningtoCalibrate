# RSUQ — Random-Set Uncertainty Quantification

One library, four papers. Frames are the interface; papers are thin drivers.

    rsuq/core/frame.py    FrameProtocol; FixedFrame (NeurIPS, Stage II,
                          Barycenter); ContextFrame (AAAI)
    rsuq/core/senses.py   sense inventory for ContextFrame (OD-1..OD-3 marked)
    rsuq/core/beliefs.py  mass heads, q, pignistics (per-position aware)
    rsuq/core/signals.py  W, H_tc, H(BetP), size-controlled W
    rsuq/diagnostics/     orthogonality gate, redundancy screen, smoking-gun
                          pairs, AUROC/sep-ratio/ECE
    papers/aaai_driver.py the controlled frame comparison (claims + ablations
                          pre-registered in-file)
    tests/test_core.py    identities I1-I8, pinned

Design rule: if a paper driver exceeds ~200 lines, the abstraction leaked —
move the logic into the library.
