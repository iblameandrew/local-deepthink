
import asyncio
import copy
import json
import logging
import os
import time
from typing import List, Dict, Any, Optional
from langchain_core.messages import SystemMessage, HumanMessage
from deepthink.chains import (
    get_task_master_chain,
    get_seed_creator_chain,
    get_mirror_descent_chain,
    get_mixing_chain,
    get_followup_question_chain,
    DISTILLATION_ARCHETYPES
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DistillationAgent:
    """A single agent in the distillation topology."""

    def __init__(self, agent_id: str, archetype_id: int, system_prompt: str,
                 attributes: List[str], skills: List[str] = None):
        self.id = agent_id
        self.archetype_id = archetype_id
        self.system_prompt = system_prompt
        self.attributes = attributes
        self.skills = skills or []
        self.history: List[Dict[str, str]] = []       # [{question, answer}, ...]
        self.current_question = ""
        self.difficulty_history: List[str] = []        # "Easy" or "Hard"
        self.context_memory = ""                       # Per-agent inherited context, max 100k tokens
        self.inherited_from: Optional[str] = None      # Parent agent id if this is a child
        self.solved_parent_question = False             # Track if child solved parent's hard question

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "archetype_id": self.archetype_id,
            "system_prompt": self.system_prompt,
            "attributes": self.attributes,
            "skills": self.skills,
            "difficulty_history": self.difficulty_history,
            "current_question": self.current_question,
            "inherited_from": self.inherited_from,
            "history_length": len(self.history),
            "context_memory_chars": len(self.context_memory),
        }

    def deep_copy_state(self) -> dict:
        """Full snapshot for archiving."""
        return {
            "id": self.id,
            "archetype_id": self.archetype_id,
            "system_prompt": self.system_prompt,
            "attributes": list(self.attributes),
            "skills": list(self.skills),
            "current_question": self.current_question,
            "difficulty_history": list(self.difficulty_history),
            "inherited_from": self.inherited_from,
            "history": list(self.history),
        }


class DistillationGraph:
    """
    Knowledge Distillation Graph.
    
    Topology: 1x2x2x2x2x2x1 (12 agents, 7 layers, no synthesis node).
    Each epoch: TaskMaster → ForwardPass → MirrorDescent → SeedCreator → Followup.
    Runs until token_budget is exhausted.
    """

    CONTEXT_MEMORY_MAX_CHARS = 400_000  # ~100k tokens at ~4 chars/token

    def __init__(self, llm, topics: List[str], anchor_question: str,
                 token_budget: int = 1_000_000, debug_mode: bool = False,
                 output_dir: str = "distillation_output"):
        self.llm = llm
        self.topics = topics
        self.anchor_question = anchor_question
        self.token_budget = token_budget
        self.debug_mode = debug_mode
        self.is_running = True  # Can be set to False externally to stop

        # Token accounting (input + output separately)
        self.total_input_tokens = 0
        self.total_output_tokens = 0

        self.epochs_run = 0
        self.topology_structure = [1, 2, 2, 2, 2, 2, 1]  # 1x2x2x2x2x2x1
        self.layers: List[List[DistillationAgent]] = []
        self.distilled_data: List[dict] = []  # QA pairs — the main product
        self.topology_archive: List[dict] = []  # Snapshot per epoch
        self.log_queue = asyncio.Queue()
        self.final_answer = ""

        # Real-time output file
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.dataset_path = os.path.join(self.output_dir, "distilled_dataset.json")
        self.topology_archive_path = os.path.join(self.output_dir, "topology_archive.json")

        self._initialize_graph()

    # ------------------------------------------------------------------
    # INITIALIZATION
    # ------------------------------------------------------------------

    def _initialize_graph(self):
        """Create 12 agents from the zodiac-derived archetypes across 7 layers."""
        archetype_keys = list(DISTILLATION_ARCHETYPES.keys())
        agent_counter = 0

        for layer_idx, num_agents in enumerate(self.topology_structure):
            layer_agents = []
            for slot in range(num_agents):
                if agent_counter < len(archetype_keys):
                    arch_id = archetype_keys[agent_counter]
                    arch = DISTILLATION_ARCHETYPES[arch_id]
                    agent = DistillationAgent(
                        agent_id=f"cnt_{layer_idx}_{slot}",
                        archetype_id=arch_id,
                        system_prompt=arch["system_prompt"],
                        attributes=list(arch["attributes"]),
                        skills=list(arch.get("skills", [])),
                    )
                    layer_agents.append(agent)
                    agent_counter += 1
            self.layers.append(layer_agents)

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------

    async def log(self, message: str):
        await self.log_queue.put(message)

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough estimate: 1 token ≈ 4 characters."""
        return len(text) // 4

    def _count_tokens(self, input_text: str, output_text: str):
        """Track input and output tokens separately."""
        self.total_input_tokens += self._estimate_tokens(input_text)
        self.total_output_tokens += self._estimate_tokens(output_text)

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    def _trim_context_memory(self, text: str) -> str:
        """Enforce the 100k-token (~400k char) cap per agent context."""
        if len(text) > self.CONTEXT_MEMORY_MAX_CHARS:
            return text[-self.CONTEXT_MEMORY_MAX_CHARS:]
        return text

    def _flat_agents(self) -> List[DistillationAgent]:
        return [a for layer in self.layers for a in layer]

    def _build_current_grid_description(self) -> str:
        """Build a text description of all 12 CURRENT agents for mirror descent."""
        lines = []
        for agent in self._flat_agents():
            lines.append(
                f"  {agent.archetype_id}. Agent {agent.id}: "
                f"Attributes=[{', '.join(agent.attributes)}], "
                f"Skills=[{', '.join(agent.skills)}]"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # ARCHIVING
    # ------------------------------------------------------------------

    def _archive_topology(self):
        """Deep-copy the current topology for historical record."""
        snapshot = {
            "epoch": self.epochs_run,
            "topics": list(self.topics),
            "layers": [
                [agent.deep_copy_state() for agent in layer]
                for layer in self.layers
            ],
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
        }
        self.topology_archive.append(snapshot)
        # Write archive to disk
        try:
            with open(self.topology_archive_path, "w", encoding="utf-8") as f:
                json.dump(self.topology_archive, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to write topology archive: {e}")

    def _write_dataset_to_file(self):
        """Write the distilled QA dataset to disk in real-time."""
        try:
            with open(self.dataset_path, "w", encoding="utf-8") as f:
                json.dump({
                    "anchor_question": self.anchor_question,
                    "total_epochs": self.epochs_run,
                    "total_input_tokens": self.total_input_tokens,
                    "total_output_tokens": self.total_output_tokens,
                    "total_tokens": self.total_tokens,
                    "qa_pairs": self.distilled_data,
                }, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to write dataset: {e}")

    # ------------------------------------------------------------------
    # PERPLEXITY HEURISTIC
    # ------------------------------------------------------------------

    def _compute_perplexity_heuristic(self) -> float:
        """
        Compute a perplexity-like diversity score for the current epoch.
        Higher = more agents struggling (more exploration happening).
        Lower = agents handling questions well (convergence).
        """
        flat = self._flat_agents()
        if not flat:
            return 0.0
        hard_count = sum(
            1 for a in flat
            if a.difficulty_history and a.difficulty_history[-1] == "Hard"
        )
        # Ratio of hard agents = exploration pressure
        # Scale to a "perplexity-like" range for visualization
        ratio = hard_count / len(flat)
        return round(ratio * 100, 2)  # 0-100 scale

    # ------------------------------------------------------------------
    # EPOCH: MAIN ENTRY POINT
    # ------------------------------------------------------------------

    async def run_epoch(self) -> bool:
        """
        Run a single epoch: TaskMaster → ForwardPass → MirrorDescent → SeedCreator.
        Returns True if should continue, False if budget exhausted or stopped.
        """
        if not self.is_running:
            await self.log("Distillation stopped by user.")
            return False

        self.epochs_run += 1
        await self.log(f"━━━ Starting Epoch {self.epochs_run} ━━━")

        # Archive topology BEFORE mutations
        self._archive_topology()

        flat_agents = self._flat_agents()

        # ── STEP 1: Task Master / Question Assignment ──
        await self._step_assign_questions(flat_agents)

        # ── STEP 2: Feed-Forward Pass ──
        await self._step_forward_pass()

        # ── STEP 3: Mirror Descent ──
        await self._step_mirror_descent(flat_agents)

        # ── STEP 4: Synthesize Final Answer ──
        await self._step_synthesize_final_answer(flat_agents)

        # ── STEP 5: Seed Creator + Followup Questions ──
        await self._step_seed_and_followup(flat_agents)

        # ── Write dataset to file in real-time ──
        self._write_dataset_to_file()

        # ── Check budget ──
        await self.log(
            f"Epoch {self.epochs_run} done. "
            f"Tokens: {self.total_tokens:,} / {self.token_budget:,} "
            f"(in: {self.total_input_tokens:,}, out: {self.total_output_tokens:,})"
        )

        if self.total_tokens >= self.token_budget:
            await self.log("✓ Token budget exhausted. Distillation complete.")
            return False

        return self.is_running

    # ------------------------------------------------------------------
    # STEP 1: QUESTION ASSIGNMENT
    # ------------------------------------------------------------------

    async def _step_assign_questions(self, flat_agents: List[DistillationAgent]):
        """
        Epoch 1: Task Master breaks anchor into 12 sub-questions.
        Epoch 2+: Questions were already assigned at end of previous epoch.
                  Agents with current_question keep theirs (Hard agents).
                  Agents without got new questions from the Seed/Followup step.
        """
        if self.epochs_run == 1:
            await self.log(f"Task Master decomposing anchor: {self.anchor_question}")
            sub_questions = await self._invoke_task_master()

            for i, agent in enumerate(flat_agents):
                if i < len(sub_questions):
                    agent.current_question = sub_questions[i]
                else:
                    agent.current_question = sub_questions[-1] if sub_questions else self.anchor_question

        # On epoch 2+, questions are already assigned by _step_seed_and_followup
        # Just log current assignments
        for agent in flat_agents:
            name = DISTILLATION_ARCHETYPES.get(agent.archetype_id, {}).get("name", "Mixed")
            await self.log(f"  Agent {agent.id} ({name}): {agent.current_question[:60]}...")

    async def _invoke_task_master(self) -> List[str]:
        if self.debug_mode:
            questions = [f"Debug Q{i}: {self.anchor_question} — aspect {i}" for i in range(12)]
            await self.log("DEBUG: Generated 12 mock sub-questions.")
            return questions

        chain = get_task_master_chain(self.llm)
        try:
            input_text = f"topics: {', '.join(self.topics)}, anchor: {self.anchor_question}"
            result = await chain.ainvoke({
                "topics": ", ".join(self.topics),
                "anchor_question": self.anchor_question,
            })
            self._count_tokens(input_text, result)
            parsed = json.loads(result)
            return parsed.get("sub_questions", [f"Sub-question {i}" for i in range(12)])
        except Exception as e:
            await self.log(f"Task Master error: {e}")
            return [f"Sub-question {i} about {self.anchor_question}" for i in range(12)]

    # ------------------------------------------------------------------
    # STEP 2: FEED-FORWARD PASS
    # ------------------------------------------------------------------

    async def _step_forward_pass(self):
        """
        Process agents layer by layer. Each layer receives the output of the
        previous layer as context. Each agent processes its own sub-question
        within the global anchor as grand objective.
        """
        await self.log("▶ Feed-Forward Pass...")
        previous_layer_output = ""

        for layer_idx, layer in enumerate(self.layers):
            tasks = [self._process_agent(agent, previous_layer_output) for agent in layer]
            results = await asyncio.gather(*tasks)

            layer_text = [r for r in results if r]
            previous_layer_output = "\n---\n".join(layer_text)

            # Trim if extremely long (keep it manageable for next layer)
            if len(previous_layer_output) > 80_000:
                previous_layer_output = previous_layer_output[-80_000:]

            await self.log(f"  Layer {layer_idx} ({len(layer)} agents) complete.")

    async def _process_agent(self, agent: DistillationAgent, layer_context: str) -> str:
        """
        Invoke LLM for a single agent. Combines:
        - Agent's per-agent context memory (inherited, max 100k tokens)
        - Layer context (output from previous layer)
        - Global anchor question
        - Agent's own sub-question
        """
        if self.debug_mode:
            import random
            await asyncio.sleep(0.1)
            words = ["epoch", "optimization", "gradient", "descent", "neural",
                     "flux", "tensor", "backprop", "latent", "space", "entropy",
                     "manifold", "convergence", "divergence", "resonance"]
            content = " ".join([random.choice(words) for _ in range(30)])
            content = f"[DEBUG] {content}"

            agent.history.append({"question": agent.current_question, "answer": content})
            self.distilled_data.append({
                "epoch": self.epochs_run,
                "agent_id": agent.id,
                "archetype_id": agent.archetype_id,
                "question": agent.current_question,
                "answer": content,
            })
            self._count_tokens(agent.current_question, content)
            # Update agent context memory
            agent.context_memory = self._trim_context_memory(
                agent.context_memory + f"\n[Epoch {self.epochs_run}] Q: {agent.current_question}\nA: {content}\n"
            )
            return content

        prompt = f"""<Context>
Grand Objective / Anchor Question: {self.anchor_question}

Your Accumulated Memory:
{agent.context_memory[-20000:] if agent.context_memory else '(First epoch — no prior memory.)'}

Previous Layer Context:
{layer_context[:20000] if layer_context else '(First layer — no prior context.)'}
</Context>

<Task>
Your specific sub-question: {agent.current_question}
</Task>

<Instruction>
Answer your sub-question deeply. Integrate your accumulated memory and the previous
layer's context where relevant. Maintain your unique perspective and attributes.
Be thorough and analytical.
</Instruction>"""

        try:
            messages = [
                SystemMessage(content=agent.system_prompt),
                HumanMessage(content=prompt),
            ]
            response = await self.llm.ainvoke(messages)
            content = response.content

            # Count both input and output tokens
            input_text = agent.system_prompt + prompt
            self._count_tokens(input_text, content)

            # Update agent state
            agent.history.append({"question": agent.current_question, "answer": content})
            self.distilled_data.append({
                "epoch": self.epochs_run,
                "agent_id": agent.id,
                "archetype_id": agent.archetype_id,
                "question": agent.current_question,
                "answer": content,
            })
            # Update per-agent context memory (capped at 100k tokens)
            agent.context_memory = self._trim_context_memory(
                agent.context_memory + f"\n[Epoch {self.epochs_run}] Q: {agent.current_question}\nA: {content}\n"
            )
            return content
        except Exception as e:
            await self.log(f"Error processing agent {agent.id}: {e}")
            return ""

    # ------------------------------------------------------------------
    # STEP 3: MIRROR DESCENT
    # ------------------------------------------------------------------

    async def _step_mirror_descent(self, flat_agents: List[DistillationAgent]):
        """
        Evaluate each agent's performance. If Hard, find the best match from
        the CURRENT grid (not static archetypes) and spawn a child.
        """
        await self.log("◀ Mirror Descent Pass...")

        current_grid_description = self._build_current_grid_description()

        for agent in flat_agents:
            if not self.is_running:
                return

            try:
                eval_result = await self._evaluate_agent(agent, current_grid_description)
                difficulty = eval_result.get("difficulty", "Easy")
                agent.difficulty_history.append(difficulty)
                await self.log(f"  Agent {agent.id}: {difficulty}")

                if difficulty == "Hard":
                    await self._handle_hard_agent(agent, eval_result, flat_agents)
                else:
                    # Easy — check if this is a child that just solved its parent's question
                    if agent.inherited_from:
                        agent.solved_parent_question = True
                        await self.log(f"  ✓ Child {agent.id} solved inherited question!")
                    # Will get a new question in the Seed/Followup step
                    agent.current_question = ""

            except Exception as e:
                await self.log(f"Mirror Descent error for {agent.id}: {e}")

    async def _evaluate_agent(self, agent: DistillationAgent, grid_description: str) -> dict:
        if self.debug_mode:
            import random
            difficulty = "Hard" if random.random() > 0.5 else "Easy"
            if difficulty == "Hard":
                candidates = [a for a in self._flat_agents() if a.id != agent.id]
                helper = random.choice(candidates) if candidates else None
                return {
                    "difficulty": "Hard",
                    "best_match_agent_id": helper.id if helper else None,
                    "reasoning": "Debug mode forced Hard.",
                }
            return {"difficulty": "Easy", "reasoning": "Debug mode forced Easy."}

        chain = get_mirror_descent_chain(self.llm)
        input_data = {
            "question": agent.current_question,
            "agent_attributes": ", ".join(agent.attributes),
            "agent_answer": agent.history[-1]["answer"] if agent.history else "",
            "current_grid": grid_description,
        }
        input_text = json.dumps(input_data)
        result_json = await chain.ainvoke(input_data)
        self._count_tokens(input_text, result_json)
        return json.loads(result_json)

    async def _handle_hard_agent(self, agent: DistillationAgent,
                                  eval_result: dict, flat_agents: List[DistillationAgent]):
        """Spawn a child from the struggling agent + the best-match helper."""
        helper_agent_id = eval_result.get("best_match_agent_id")
        # Fallback: try archetype-based lookup
        helper_archetype_id = eval_result.get("best_match_archetype_id")

        # Find the actual helper agent in the current grid
        helper_agent = None
        if helper_agent_id:
            helper_agent = next((a for a in flat_agents if a.id == helper_agent_id), None)
        if not helper_agent and helper_archetype_id:
            # Fallback to archetype match
            helper_agent = next(
                (a for a in flat_agents if a.archetype_id == helper_archetype_id and a.id != agent.id),
                None,
            )
        if not helper_agent:
            await self.log(f"  No helper found for {agent.id}, keeping agent.")
            return

        await self.log(f"  Spawning child from {agent.id} + {helper_agent.id}")

        mix_result = await self._mix_agents(agent, helper_agent)

        # Replace agent in-place: child inherits parent's context memory
        old_context = agent.context_memory
        agent.inherited_from = agent.id + f"_epoch{self.epochs_run}"
        agent.system_prompt = mix_result.get("new_system_prompt", agent.system_prompt)
        agent.attributes = mix_result.get("new_attributes", agent.attributes)
        agent.skills = mix_result.get("new_skills", agent.skills)
        agent.context_memory = self._trim_context_memory(old_context)
        agent.solved_parent_question = False
        # Hard agent keeps the same question for the next epoch

    async def _mix_agents(self, parent_a: DistillationAgent,
                           parent_b: DistillationAgent) -> dict:
        if self.debug_mode:
            return {
                "new_system_prompt": f"Debug Child of {parent_a.id} and {parent_b.id}",
                "new_attributes": parent_a.attributes[:2] + parent_b.attributes[:2],
                "new_skills": parent_a.skills[:1] + parent_b.skills[:1],
            }

        chain = get_mixing_chain(self.llm)
        input_data = {
            "parent_a_attributes": ", ".join(parent_a.attributes),
            "parent_a_prompt": parent_a.system_prompt,
            "parent_b_attributes": ", ".join(parent_b.attributes),
            "parent_b_prompt": parent_b.system_prompt,
        }
        input_text = json.dumps(input_data)
        result_json = await chain.ainvoke(input_data)
        self._count_tokens(input_text, result_json)
        return json.loads(result_json)

    # ------------------------------------------------------------------
    # STEP 4: SYNTHESIZE FINAL ANSWER
    # ------------------------------------------------------------------

    async def _step_synthesize_final_answer(self, flat_agents: List[DistillationAgent]):
        """
        Combine all 12 agents' latest answers into a coherent final answer.
        This is NOT a synthesis node (spec says no synthesis node) — it's a
        simple concatenation for the Seed Creator to analyze.
        """
        parts = []
        for agent in flat_agents:
            if agent.history:
                last = agent.history[-1]
                parts.append(f"[Agent {agent.id}] Q: {last['question']}\nA: {last['answer']}")
        self.final_answer = "\n\n---\n\n".join(parts)
        # Truncate for downstream chains (to avoid blowing context windows)
        if len(self.final_answer) > 40_000:
            self.final_answer = self.final_answer[-40_000:]

    # ------------------------------------------------------------------
    # STEP 5: SEED CREATOR + FOLLOWUP QUESTIONS
    # ------------------------------------------------------------------

    async def _step_seed_and_followup(self, flat_agents: List[DistillationAgent]):
        """
        Seed Creator: generate 12 ontologically close new topics.
        Followup: generate new questions for agents that had it "Easy".
        """
        await self.log("🌱 Seed Creator + Followup Questions...")

        # Generate new topics
        new_topics = await self._invoke_seed_creator()
        self.topics = new_topics
        await self.log(f"  Evolved topics: {new_topics[:3]}{'...' if len(new_topics) > 3 else ''}")

        # Identify agents needing new questions
        # Easy agents: current_question was cleared
        # Also: child agents that solved their parent's question
        agents_needing_questions = []
        for agent in flat_agents:
            if not agent.current_question:
                agents_needing_questions.append(agent)
            elif agent.solved_parent_question:
                agent.current_question = ""
                agent.solved_parent_question = False
                agents_needing_questions.append(agent)

        if agents_needing_questions:
            new_questions = await self._invoke_followup(len(agents_needing_questions))
            for i, agent in enumerate(agents_needing_questions):
                if i < len(new_questions):
                    agent.current_question = new_questions[i]
                else:
                    agent.current_question = f"Deepen your analysis of: {new_topics[i % len(new_topics)] if new_topics else self.anchor_question}"
            await self.log(f"  Assigned {len(agents_needing_questions)} new questions.")

    async def _invoke_seed_creator(self) -> List[str]:
        if self.debug_mode:
            return [f"Debug Topic {i}" for i in range(12)]

        chain = get_seed_creator_chain(self.llm)
        truncated_answer = self.final_answer[:4000]
        input_data = {
            "current_topics": ", ".join(self.topics),
            "final_answer": truncated_answer,
        }
        input_text = json.dumps(input_data)
        try:
            result = await chain.ainvoke(input_data)
            self._count_tokens(input_text, result)
            return json.loads(result).get("new_topics", self.topics)
        except Exception as e:
            await self.log(f"Seed Creator error: {e}")
            return self.topics

    async def _invoke_followup(self, count: int) -> List[str]:
        if self.debug_mode:
            return [f"Debug Followup Q{i}" for i in range(count)]

        chain = get_followup_question_chain(self.llm)
        truncated_answer = self.final_answer[:4000]
        input_data = {
            "new_topics": ", ".join(self.topics),
            "final_answer": truncated_answer,
            "num_questions": count,
        }
        input_text = json.dumps(input_data)
        try:
            result = await chain.ainvoke(input_data)
            self._count_tokens(input_text, result)
            return json.loads(result).get("new_questions", [])
        except Exception as e:
            await self.log(f"Followup error: {e}")
            return []
