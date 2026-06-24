"""RSUQ — Random-Set Uncertainty Quantification.

Shared machinery for the portfolio: frames (fixed + contextual), belief
readouts, uncertainty signals, and the diagnostic battery. Papers are thin
drivers in papers/.
"""
from .core.frame import FixedFrame, ContextFrame, FrameProtocol
from .core.beliefs import (MassHead, within_cluster_q, cluster_prob_mass,
                           pignistic, ranked_pignistic)
from .core.signals import (credal_width, token_choice_axis, pignistic_entropy,
                           all_signals)
from .rsnn_compat import (betp_matrix, final_betp,
                          groundtruth_belief_encode, mobius_inverse)
from .qa_data import load_qa_instances
from .train import train_mass_head
from .deferral import risk_coverage, deferral_battery, aurc_delta_ci
