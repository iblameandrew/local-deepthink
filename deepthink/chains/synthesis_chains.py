"""
Synthesis and problem decomposition chain factories.
These chains handle solution synthesis and problem reframing.
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def get_synthesis_chain(llm):
    """Creates a chain that synthesizes multiple agent solutions into one."""
    prompt = ChatPromptTemplate.from_template("""
You are a synthesis agent. Your role is to combine the solutions from multiple agents into a single, coherent, and comprehensive answer.
You will receive a list of JSON objects, each representing a solution from a different agent.
Your task is to synthesize these solutions, considering the original problem, and produce a final answer in the same JSON format.

Original Problem: {original_request}

Current problem:
---
{current_problem}
---

Agent Solutions:
{agent_solutions}

Synthesize the final solution:
""")
    return prompt | llm | StrOutputParser()


def get_code_synthesis_chain(llm):
    """Creates a chain that synthesizes code solutions from multiple agents."""
    prompt = ChatPromptTemplate.from_template("""
You are an expert code synthesis agent. Your role is to combine multiple code snippets and solutions from different agents into a single, cohesive, and runnable application.
You must analyze the different approaches, merge them logically, handle potential conflicts, and produce a final, well-structured code file.
The final output should be a single block of code.

Consider the following existing modules that have already been successfully built. You can use their interfaces in your solution.
---
Existing Modules Context:
{synthesis_context}
---

Original Problem: {original_request}
                                            
Current problem:
---
{current_problem}
---

Agent Solutions (containing code snippets):
{agent_solutions}

Synthesize the final, complete code application. 
CRITICAL: The final output must be ONLY the runnable code, wrapped in triple backticks with a "python" language identifier (e.g., ```python\n[CODE HERE]\n```). Do not include any other text or explanations before or after the code block.

Final code application:
""")
    return prompt | llm | StrOutputParser()


def get_problem_decomposition_chain(llm):
    """Breaks down a complex problem into smaller subproblems."""
    prompt = ChatPromptTemplate.from_template("""
You are a master strategist and problem decomposer. Your task is to break down a complex, high-level problem into a series of smaller, more manageable, and granular subproblems.
You will be given a main problem and the total number of subproblems to generate.
Each subproblem should represent a distinct line of inquiry, a specific component to be developed, or a unique perspective to be explored, which, when combined, will contribute to solving the main problem.
If the problem is code, you must generate a list of subproblems that are specifications for code snippets that can be used to solve the main problem.

The output must be a JSON object with a single key "sub_problems", which is a list of strings. The list must contain exactly {num_sub_problems} unique subproblems.

Main Problem: "{problem}"
Total number of subproblems to generate: {num_sub_problems}

Generate the JSON object:
""")
    return prompt | llm | StrOutputParser()


def get_problem_reframer_chain(llm):
    """Reframes problems based on previous solutions to drive iteration."""
    prompt = ChatPromptTemplate.from_template("""
You are a strategic problem re-framer. You have been informed that an AI agent team has made a significant breakthrough on a problem.

Your task is to formulate wether the team needs to deepen with a more complex solution, or widen and perfect the current one, referring previous solutions with the new one. 
                                            
If the problem statement is code, and the current solution could be more modular and granular, you must set the next goal to improve the current solution with more breath instead of synthesis
                                              
- if however the solution is already up to a good standard of breadth, you must set a next step that builds upon the success of the solution but stil grounded in the original problem  trying to achieve the same goal.

The new problem should represent the "next logical step"; a more ambitious goal, or a more grounded approach if the solution is still not granular enough. Compare previous solutions to current ones, check if references and imports to previous work are appropiately used and decide the next step.

Original Problem:
---
{original_request}
---

Current problem:
---
{current_problem}
---
Previous Solution:                            
---
{previous_solution}
---

Previous solutions documentation:
                                              
{module_cards}

The Breakthrough Solution:
---
{final_solution}
---
                                              

Your output must be a JSON object with a single key: "new_problem".

Formulate the new, more advanced problem:
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
