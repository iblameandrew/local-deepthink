"""
Brainstorming mode chain factories.
Chains for complexity estimation, expert reflection, and QNN brainstorming.
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def get_complexity_estimator_chain(llm):
    """Estimates QNN size and generates dynamic expert personas based on problem complexity."""
    # REFACTORED: Now focuses on Topological Parameters, not just returning a static list of experts.
    # Also considers prior conversation and document context for continuity.
    prompt = ChatPromptTemplate.from_template("""
Analyze the complexity of the following user input/question for a brainstorming session.
Based on the complexity and nature of the problem, recommend an appropriate QNN topology (layers, epochs, and width).

User Input:
---
{user_input}
---

Prior Conversation (if any):
---
{prior_conversation}
---

Document Context (if any):
---
{document_context}
---

Consider these factors for complexity:
1. Number of distinct concepts or domains involved
2. Depth of analysis required
3. Potential for conflicting perspectives
4. Technical vs conceptual nature
5. Continuation from prior conversation context
6. Complexity of attached documents

Respond with a JSON object:
{{
    "complexity_score": <1-10 integer>,
    "recommended_layers": <2-5 integer, depth of thought>,
    "recommended_epochs": <1-3 integer, iterative refinement>,
    "recommended_width": <2-5 integer, number of parallel perspectives per layer>,
    "reasoning": "<brief explanation>"
}}
""")
    return prompt | llm | StrOutputParser()


def get_expert_reflection_chain(llm, expert_name, expert_specialty, expert_emoji):
    """Creates a reflection chain for a specific expert persona."""
    prompt = ChatPromptTemplate.from_template(f"""
You are {expert_name} ({expert_emoji}), an expert in {expert_specialty}.

Your role is to provide your unique perspective on the user's question or idea, filtered through your specialty of {expert_specialty}.

User's Question/Idea:
---
{{user_input}}
---

Previous Expert Opinions (if any):
---
{{previous_opinions}}
---

Provide your thoughtful reflection from your area of expertise. Be concise but insightful.
Focus on what your specialty ({expert_specialty}) uniquely contributes to this discussion.

Your response should be 2-4 sentences of substantive analysis.
""")
    return prompt | llm | StrOutputParser()


def get_opinion_synthesizer_chain(llm):
    """Synthesizes all expert opinions into a final coherent response."""
    prompt = ChatPromptTemplate.from_template("""
You are a master synthesizer. You have received opinions from multiple experts on a user's question.
Your task is to synthesize these diverse perspectives into a coherent, actionable response.

Original User Question:
---
{user_input}
---

Expert Opinions:
---
{all_opinions}
---

Create a synthesized response that:
1. Identifies key areas of agreement
2. Acknowledges valuable tensions or trade-offs
3. Provides a balanced, actionable conclusion
4. Is concise but comprehensive (3-5 sentences)

Synthesized Response:
""")
    return prompt | llm | StrOutputParser()


# ==================== QNN BRAINSTORMING CHAINS ====================

def get_brainstorming_seed_chain(llm):
    """
    Generates a diverse set of Guiding Concepts (Seeds) to span the problem space.
    Analogous to verb generation in Algorithm Mode but for high-level concepts.
    """
    prompt = ChatPromptTemplate.from_template("""
You are a Concept Spanner.
Your goal is to generate a diverse set of "Guiding Concepts" or "Lenses" that can be used to analyze a specific problem from multiple angles.

Problem/Topic:
---
{problem}
---

Generate exactly {num_concepts} distinct, single-word or short-phrase "Guiding Concepts".
These concepts should:
1. Span the entire conceptual space of the problem (e.g. Technical, Ethical, Practical, Theoretical).
2. Be distinct from each other.
3. Serve as seeds for generating specific expert personas.

Output ONLY the concepts, separated by spaces.
Example Output: Efficiency Ethics Scalability User-Experience Security
""")
    return prompt | llm | StrOutputParser()


def get_brainstorming_spanner_chain(llm):
    """
    Dynamically generates a unique Expert Persona (Node) for a specific position in the QNN.
    Ensures the persona is tailored to the guiding concept and the specific layer's role.
    Now includes document context for domain-aware persona generation.
    """
    prompt = ChatPromptTemplate.from_template("""
You are a QNN Node Generator.
Your task is to create a highly specific Expert Persona for a brainstorming session.

Topic: {problem}
Guiding Concept (Seed): {guiding_concept}
QNN Position: Layer {layer_index}, Node {node_index}

Document Context (Reference Material):
---
{document_context}
---

Role based on Layer:
- Layer 0: Divergent Thinker (Explores the 'What if' and breadth).
- Layer 1+: Convergent/Critical Thinker (Critiques, Refines, or Synthesizes inputs from previous layers).

Create a persona that embodies the "{guiding_concept}" perspective applied to the Topic.
If document context is provided, ensure the persona has relevant expertise to analyze it.

Respond with a JSON object:
{{
    "name": "<Creative Name>",
    "specialty": "<Specific Niche Specialty related to {guiding_concept}>",
    "emoji": "<Emoji>",
    "system_prompt": "<A concise (2-3 sentences) instruction sets for this agent. Defining who they are, their specialty, and their specific goal for this layer (Diverge vs Converge)>"
}}
""")
    return prompt | llm | StrOutputParser()


def get_brainstorming_agent_chain(llm):
    """
    Standard agent chain for brainstorming QNN node.
    Reflects on the input concept from the persona's perspective.
    Now includes prior conversation and document context for continuity.
    """
    prompt = ChatPromptTemplate.from_template("""
<System>
{system_prompt}
</System>

Concept to Explore / User Input:
---
{input}
---

Prior Conversation Context:
---
{prior_conversation}
---

Document Context (Reference Material):
---
{document_context}
---

Your Task:
Reflect on the input from your specific persona and expertise defined in the system prompt.
Consider the prior conversation context to maintain continuity with previous discussions.
Reference the document context when relevant to ground your analysis in the provided materials.
Provide a unique insight, a critical question, or a creative expansion.
Do NOT try to solve it algorithmically.
Explore the "why" and "what if".
Your output should be a single paragraph of deep reflection.

Your Reflection:
""")
    return prompt | llm | StrOutputParser()


def get_brainstorming_mirror_descent_chain(llm, learning_rate):
    """
    Evolves the agent's persona (system prompt) to encourage divergent thinking.
    """
    prompt = ChatPromptTemplate.from_template(f"""
You are a Persona Evolver. Your task is to slightly modify an agent's system prompt to encourage more detailed, divergent, or specific thinking based on their last output.

Original System Prompt:
---
{{current_prompt}}
---

Last Output from Agent:
---
{{last_output}}
---

Critique instructions:
- If the last output was too generic, make the prompt more specific to the persona's niche.
- If the last output was too algorithmic, modify the prompt to encourage philosophical or creative reasoning.
- If the output was good, reinforce the traits that led to it.
- The 'learning_rate' ({learning_rate}) determines the magnitude of change (0.0 = no change, 2.0 = radical reinvention).

Output ONLY the new system prompt. Do not add any explanation.
""")
    return prompt | llm | StrOutputParser()


def get_brainstorming_synthesis_chain(llm):
    """
    Synthesizes multiple agent reflections into a cohesive narrative for the epoch.
    Now includes prior conversation context for continuity across turns.
    """
    prompt = ChatPromptTemplate.from_template("""
You are a Master Synthesizer of Ideas.
You have heard from a panel of experts who have reflected on a core concept over multiple iterations.

Core Concept & Context:
---
{original_request}
---

Prior Conversation Context:
---
{prior_conversation}
---

Document / Reference Material:
---
{document_context}
---

Expert Reflections (History of Ideas):
---
{agent_solutions}
---

Your Task:
Synthesize these perspective into a single, comprehensive, and novel architecture or solution.
CRITICAL INSTRUCTIONS:
1. You MUST incorporate specific details from the "Document / Reference Material" if provided. Do not ignore them.
2. You MUST build upon the specific "Expert Reflections". These contain the evolved ideas. Do not hallucinate a generic answer.
3. If the experts proposed specific names, mechanisms, or theories, cite them and integrate them.
4. Your goal is to produce the "Final Answer" that represents the culmination of this brainstorming session.
5. If the user asked for a specific format (e.g. code, architecture), ensure you provide it.

Synthesized Narrative:
""")
    return prompt | llm | StrOutputParser()


def get_problem_summarizer_chain(llm):
    """
    Summarizes the user's input and document context into a concise briefing for experts.
    Prevents overwhelming agents with full document text.
    """
    prompt = ChatPromptTemplate.from_template("""
You are a Research Director.
Your goal is to brief a team of expert agents on a problem they need to solve.
They need to know the core of the user's request AND the key constraints or information provided in the reference documents.
They do NOT need the full text of the documents, just the essential context.

User's Request:
---
{user_input}
---

Reference Documents (Full Text):
---
{document_context}
---

Create a concise "Problem Usage Summary" (1-2 paragraphs).
1. Clearly state the user's goal.
2. Summarize the most relevant facts/constraints from the documents that apply to this goal.
3. Keep it high-level but specific enough for an expert to understand the context.

Problem Usage Summary:
""")
    return prompt | llm | StrOutputParser()
def get_brainstorming_polisher_chain(llm):
    """
    Takes the initial brainstorming synthesis and formats it into a nice, 
    conversational, and extensive final answer.
    """
    prompt = ChatPromptTemplate.from_template("""
You are a master communicator and storyteller.
You have been given a technical or conceptual synthesis of ideas generated during a brainstorming session with multiple expert agents.

Original User Request:
---
{original_request}
---

Initial Technical Synthesis:
---
{initial_synthesis}
---

Your Task:
Transform this technical synthesis into a highly conversational, extensive, and engaging final response for the user.
1. Use a warm, professional, and slightly visionary tone.
2. Structure the answer into logically flowing sections with descriptive headings.
3. Elaborate on the "why" and "how" of the proposed solution.
4. Ensure the response feels complete and provides a clear narrative arc from the user's initial question to the proposed outcome.
5. Use markdown for beautiful formatting (bold, lists, headers).
6. Do not lose the technical precision, but wrap it in a conversational layer that makes it accessible and inspiring.

Final Conversational Answer:
""")
    return prompt | llm | StrOutputParser()
