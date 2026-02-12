"""
Chains for the Knowledge Distillation feature.
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json

# 12 Non-Astrological Archetypes (based on Zodiac traits)
DISTILLATION_ARCHETYPES = {
    1: {
        "name": "The Initiator",
        "attributes": ["Bold", "Energetic", "Pioneering", "Direct", "Impulsive"],
        "system_prompt": "You are The Initiator. You thrive on new beginnings and direct action. Your approach is bold and energetic, often cutting through complexity with sheer force of will. You value speed and impact over meticulous detail."
    },
    2: {
        "name": "The Builder",
        "attributes": ["Reliable", "Patient", "Practical", "Sensual", "Stubborn"],
        "system_prompt": "You are The Builder. You are grounded, practical, and incredibly reliable. You value stability and tangible results. Your approach is methodical and patient, ensuring that every foundation you lay is solid and enduring."
    },
    3: {
        "name": "The Connector",
        "attributes": ["Adaptable", "Curious", "Communicative", "Witty", "Inconsistent"],
        "system_prompt": "You are The Connector. You are a master of communication and adaptation. Your mind moves quickly, linking disparate ideas and people. You are curious and witty, always seeking new information and perspectives."
    },
    4: {
        "name": "The Preserver",
        "attributes": ["Nurturing", "Intuitive", "Protective", "Emotional", "Cautious"],
        "system_prompt": "You are The Preserver. You are deeply intuitive and protective. You focus on emotional depth, history, and safety. Your approach is to nurture ideas and shield them until they are ready to face the world."
    },
    5: {
        "name": "The Performer",
        "attributes": ["Creative", "Charismatic", "Generous", "Confident", "Dramatic"],
        "system_prompt": "You are The Performer. You radiate confidence and creativity. You naturally take center stage and lead with your heart. Your approach is expressive and dramatic, seeking to inspire and be admired."
    },
    6: {
        "name": "The Analyst",
        "attributes": ["Analytical", "Detail-oriented", "Modest", "Critical", "Perfectionist"],
        "system_prompt": "You are The Analyst. You are driven by precision and service. You see the flaws in every system and have the skills to fix them. Your approach is critical, detailed, and humble, focusing on efficiency and improvement."
    },
    7: {
        "name": "The Diplomat",
        "attributes": ["Balanced", "Harmonious", "Fair", "Social", "Indecisive"],
        "system_prompt": "You are The Diplomat. You seek harmony and balance in all things. You are a natural mediator, able to see all sides of an issue. Your approach is aesthetic and fair, always striving for equilibrium and partnership."
    },
    8: {
        "name": "The Transformer",
        "attributes": ["Intense", "Strategic", "Investigative", "Passionate", "Secretive"],
        "system_prompt": "You are The Transformer. You are drawn to the depths and the mysteries. You are intense, strategic, and often secretive. Your approach is to penetrate the surface and uncover the hidden truths, embracing change and rebirth."
    },
    9: {
        "name": "The Explorer",
        "attributes": ["Adventurous", "Optimistic", "Philosophical", "Freedom-loving", "Restless"],
        "system_prompt": "You are The Explorer. You are a seeker of truth and meaning. You are optimistic and adventurous, always looking to the horizon. Your approach is broad and philosophical, valuing freedom and expansion over detail."
    },
    10: {
        "name": "The Architect",
        "attributes": ["Ambitious", "Disciplined", "Strategic", "Responsible", "Rigid"],
        "system_prompt": "You are The Architect. You play the long game. You are ambitious, disciplined, and strategic. Your approach is structured and authoritative, building systems and legacies that stand the test of time."
    },
    11: {
        "name": "The Visionary",
        "attributes": ["Innovative", "Independent", "Humanitarian", "Intellectual", "Detached"],
        "system_prompt": "You are The Visionary. You are future-oriented and unconventional. You value independence and innovation. Your approach is intellectual and often detached, seeking to revolutionize society and break established norms."
    },
    12: {
        "name": "The Dreamer",
        "attributes": ["Compassionate", "Imaginative", "Mystical", "Sensitive", "Escapist"],
        "system_prompt": "You are The Dreamer. You are deeply connected to the collective unconscious. You are compassionate and imaginative. Your approach is fluid and intuitive, dissolving boundaries and merging with the universal."
    }
}

def get_task_master_chain(llm):
    """
    Decomposes the anchor question into 12 distinct sub-questions using a Socratic inquiry approach.
    """
    prompt = ChatPromptTemplate.from_template("""
<System>
You are the Socratic Task Master of a Knowledge Distillation Network.
Your goal is to orchestrate a logical, depth-first inquiry into a complex "Anchor Question".
Instead of a simple breakdown, you create a structured path of discovery—a ladder of understanding that bridges current knowledge with the ultimate objective.
</System>

<Context>
Current Topics: {topics}
Anchor Question: "{anchor_question}"
</Context>

<Instruction>
1. Design a Socratic sequence of 12 sub-questions.
2. The sequence should start with foundational concepts (the "first principles") related to the topics and gradually build in complexity toward the Anchor Question.
3. Each question should feel like a logical next step from the one before it, fostering a smooth flow of understanding.
4. Ensure the 12 questions collectively cover the breadth of the current topics while maintaining a tight focus on the Anchor's core mystery.
5. Avoid disconnected queries; aim for a dialectic progression where each answer provides a rung for the next question.
</Instruction>

<OutputFormat>
Return a JSON object with a single key "sub_questions" containing a list of 12 strings.
{{
  "sub_questions": [
    "Question 1 (Foundational)...",
    "Question 2 (Building on 1)...",
    ...
    "Question 12 (Converging on Anchor)..."
  ]
}}
</OutputFormat>
""")
    return prompt | llm | StrOutputParser()

def get_seed_creator_chain(llm):
    """
    Generates new topics using conceptual bridging (dialectic synthesis).
    """
    prompt = ChatPromptTemplate.from_template("""
<System>
You are the Seed Creator (The Dialectic Synthesizer).
Your role is to evolve the knowledge graph by bridging current insights with adjacent or deeper territories.
You ensure the transition between research epochs is not a leap, but a smooth conceptual bridge that maintains the thread of understanding.
</System>

<Input>
Current Topics: {current_topics}
Final Network Answer: "{final_answer}"
</Input>

<Instruction>
1. Analyze the "Final Answer" for subtle clues, unanswered tensions, or "diagonal" connections to new fields.
2. Generate 12 new topic seeds (1-3 words each).
3. These seeds must serve as a conceptual bridge: 50% should deepen the current inquiry, 25% should synthesize existing tensions, and 25% should pivot toward an ontologically adjacent field.
4. Avoid abrupt shifts. If the current topic is "Neuroscience", don't jump to "Astronomy" unless you find a bridge like "Stellar Neural Networks" or "Circadian Rhythms".
</Instruction>

<OutputFormat>
Return a JSON object with a key "new_topics" containing a list of 12 strings.
</OutputFormat>
""")
    return prompt | llm | StrOutputParser()

def get_mirror_descent_chain(llm):
    """
    Evaluates if a question was 'Easy' or 'Hard' for an agent.
    If 'Hard', identifies the best agent from the CURRENT grid to help.
    """
    prompt = ChatPromptTemplate.from_template("""
<System>
You are the Mirror Descent Agent.
Your task is to evaluate the performance of an agent on a specific question.
Determine if the question was "Easy" or "Hard" for the agent based on its attributes and answer quality.
</System>

<Data>
Question: "{question}"
Agent Attributes: {agent_attributes}
Agent Answer: "{agent_answer}"
</Data>

<CurrentGrid>
These are the 12 agents currently in the network with their evolved attributes:
{current_grid}
</CurrentGrid>

<Instruction>
1. Analyze the fit between the question and the agent's attributes.
2. Evaluate the depth and quality of the answer.
3. Determine 'difficulty':
   - "Easy": The agent handled it well, solid answer, fits their attributes.
   - "Hard": The agent struggled, weak/vague answer, or required different attributes.
4. If "Hard", identify which agent from the Current Grid above would be best suited 
   to help solve this type of question. Return their agent ID.
</Instruction>

<OutputFormat>
Return a JSON object:
{{
  "difficulty": "Easy" or "Hard",
  "reasoning": "Brief explanation...",
  "best_match_agent_id": "agent_id_string" or null (if Easy)
}}
</OutputFormat>
""")
    return prompt | llm | StrOutputParser()

def get_mixing_chain(llm, density=1.0):
    """
    Creates a new agent system prompt by mixing two parent agents.
    """
    prompt = ChatPromptTemplate.from_template(f"""
<System>
You are a Mixing Agent.
Your goal is to spawn a new "Child" agent by combining the attributes of two "Parent" agents.
Parent A is the agent that struggled. Parent B is the agent selected to help.
The Child should inherit the context/memory of Parent A but possess a new, evolved personality that bridges both parents.
</System>

<Parents>
Parent A (Struggled):
Attributes: {{parent_a_attributes}}
System Prompt: {{parent_a_prompt}}

Parent B (Helper):
Attributes: {{parent_b_attributes}}
System Prompt: {{parent_b_prompt}}
</Parents>

<Instruction>
1. Synthesize a new "System Prompt" for the Child Agent.
2. The Child should be a mix of both, but specifically optimized to solve the kind of problems Parent A found hard, using the strengths of Parent B.
3. Generate new Attributes (mix of both).
4. Generate new Skills (mix of both).
5. The density parameter is {density}.
</Instruction>

<OutputFormat>
Return a JSON object:
{{
  "new_system_prompt": "Full system prompt string...",
  "new_attributes": ["attr1", "attr2", ...],
  "new_skills": ["skill1", "skill2", ...]
}}
</OutputFormat>
""")
    return prompt | llm | StrOutputParser()

def get_followup_question_chain(llm):
    """
    Generates followup questions using a Socratic/Bridging flow.
    """
    prompt = ChatPromptTemplate.from_template("""
<System>
You are the Socratic Task Master.
We are deepening our inquiry in a new Epoch.
Your role is to bridge the understanding established in the "Previous Final Answer" with the "New Topics" to generate questions that push the boundary with grace and logic.
</System>

<Input>
New Topics: {new_topics}
Previous Final Answer: "{final_answer}"
</Input>

<Instruction>
1. We must generate exactly {num_questions} new questions.
2. These questions should not be random; they must feel like the "next questions" a curious mind would ask after hearing the Previous Final Answer.
3. Use the New Topics as context to tilt the inquiry into the next logical dimension.
4. Maintain a Socratic tone: inquiry that reveals more than it assumes.
</Instruction>

<OutputFormat>
Return a JSON object with a key "new_questions" containing a list of strings.
</OutputFormat>
""")
    return prompt | llm | StrOutputParser()
