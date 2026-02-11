"""
Chain factories for DeepThink QNN.
All chains are organized by purpose and can be imported from here.
"""
from .agent_chains import (
    get_input_spanner_chain,
    get_attribute_and_hard_request_generator_chain,
    get_seed_generation_chain,
    get_dense_spanner_chain,
)

from .synthesis_chains import (
    get_synthesis_chain,
    get_code_synthesis_chain,
    get_problem_decomposition_chain,
    get_problem_reframer_chain,
    get_opinion_synthesizer_chain,
)

from .utility_chains import (
    get_memory_summarizer_chain,
    get_perplexity_heuristic_chain,
    get_module_card_chain,
    get_code_detector_chain,
    get_request_is_code_chain,
    get_interrogator_chain,
    get_paper_formatter_chain,
    get_rag_chat_chain,
)

from .brainstorm_chains import (
    get_complexity_estimator_chain,
    get_expert_reflection_chain,
    get_opinion_synthesizer_chain,
    get_brainstorming_agent_chain,
    get_brainstorming_mirror_descent_chain,
    get_brainstorming_synthesis_chain,
    get_brainstorming_seed_chain,
    get_brainstorming_spanner_chain,
    get_problem_summarizer_chain,
    get_brainstorming_polisher_chain,
)

from .distillation_chains import (
    get_task_master_chain,
    get_seed_creator_chain,
    get_mirror_descent_chain,
    get_mixing_chain,
    get_followup_question_chain,
    DISTILLATION_ARCHETYPES,
)

__all__ = [
    # Agent chains
    'get_input_spanner_chain',
    'get_attribute_and_hard_request_generator_chain',
    'get_seed_generation_chain',
    'get_dense_spanner_chain',
    # Synthesis chains
    'get_synthesis_chain',
    'get_code_synthesis_chain',
    'get_problem_decomposition_chain',
    'get_problem_reframer_chain',
    'get_opinion_synthesizer_chain',
    # Utility chains
    'get_memory_summarizer_chain',
    'get_perplexity_heuristic_chain',
    'get_module_card_chain',
    'get_code_detector_chain',
    'get_request_is_code_chain',
    'get_interrogator_chain',
    'get_paper_formatter_chain',
    'get_rag_chat_chain',
    # Brainstorm chains
    'get_complexity_estimator_chain',
    'get_expert_reflection_chain',
    'get_brainstorming_agent_chain',
    'get_brainstorming_mirror_descent_chain',
    'get_brainstorming_synthesis_chain',
    'get_brainstorming_seed_chain',
    'get_brainstorming_spanner_chain',
    'get_problem_summarizer_chain',
    'get_brainstorming_polisher_chain',
    # Distillation chains
    'get_task_master_chain',
    'get_seed_creator_chain',
    'get_mirror_descent_chain',
    'get_mixing_chain',
    'get_followup_question_chain',
    'DISTILLATION_ARCHETYPES',
]

