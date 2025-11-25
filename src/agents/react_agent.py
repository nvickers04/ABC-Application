# src/agents/react_agent.py
# Purpose: Memory-enhanced ReAct (Reasoning + Acting) agent implementation using LangChain.
# This agent uses the ReAct pattern to reason about problems and take actions using available tools.
# Enhanced with memory capabilities to learn from past reasoning patterns and tool effectiveness.
# Extends BaseAgent to integrate with the existing agent framework.

import logging
from typing import Dict, Any, List, Optional
import asyncio
from datetime import datetime
import json
import hashlib
from collections import defaultdict

from .base import BaseAgent

logger = logging.getLogger(__name__)

class ReActAgent(BaseAgent):
    """
    Memory-enhanced ReAct (Reasoning + Acting) agent that uses LangChain's ReAct pattern.
    This agent can reason about complex problems and use tools to gather information and take actions.
    Enhanced with memory capabilities to learn from past reasoning patterns, tool effectiveness,
    and problem-solving strategies to continuously improve performance.

    Key memory features:
    - Remembers successful reasoning patterns and applies them to similar problems
    - Tracks tool effectiveness and recommends best tools for specific tasks
    - Learns from interaction history to optimize future performance
    - Maintains reasoning trajectories for analysis and improvement
    """

    def __init__(self, role: str = "react", config_paths: Optional[Dict[str, str]] = None,
                 prompt_paths: Optional[Dict[str, str]] = None, tools: Optional[List[Any]] = None,
                 a2a_protocol: Any = None, max_iterations: int = 5):
        """
        Initialize the ReAct agent.

        Args:
            role: Agent role identifier
            config_paths: Configuration file paths
            prompt_paths: Prompt template paths
            tools: List of tools available to the agent
            a2a_protocol: Inter-agent communication protocol
            max_iterations: Maximum reasoning iterations before stopping
        """
        super().__init__(role, config_paths or {}, prompt_paths or {}, tools or [], a2a_protocol)
        self.max_iterations = max_iterations
        self.react_available = self._check_react_availability()

        # Initialize memory structures for learning and improvement
        self._initialize_react_memory()

    def _initialize_react_memory(self):
        """Initialize memory structures for learning from past ReAct interactions."""
        self.react_memory = {
            # Track successful reasoning patterns
            'successful_patterns': {},
            # Track tool effectiveness
            'tool_effectiveness': defaultdict(lambda: {'success_count': 0, 'failure_count': 0, 'avg_time': 0}),
            # Track problem types and solutions
            'problem_solutions': {},
            # Track reasoning trajectories
            'reasoning_trajectories': [],
            # Performance metrics
            'performance_stats': {
                'total_interactions': 0,
                'successful_solutions': 0,
                'average_iterations': 0,
                'tool_usage_patterns': {}
            }
        }

        # Load existing memory if available
        self._load_react_memory()

    def _load_react_memory(self):
        """Load existing ReAct memory from persistent storage."""
        try:
            # Try to load from agent's memory persistence
            if hasattr(self, 'memory_persistence') and self.memory_persistence:
                memory_data = self.memory_persistence.load_agent_memory(f"{self.role}_react_memory")
                if memory_data:
                    self.react_memory.update(memory_data)
                    logger.info(f"Loaded ReAct memory for {self.role} agent")
        except Exception as e:
            logger.warning(f"Could not load ReAct memory: {e}")

    def _save_react_memory(self):
        """Save ReAct memory to persistent storage."""
        try:
            if hasattr(self, 'memory_persistence') and self.memory_persistence:
                self.memory_persistence.save_agent_memory(f"{self.role}_react_memory", self.react_memory)
        except Exception as e:
            logger.warning(f"Could not save ReAct memory: {e}")

    def _check_react_availability(self) -> bool:
        """Check if LangChain ReAct components are available."""
        try:
            # Try importing the components we need
            import langchain.agents  # noqa: F401
            import langchain.prompts  # noqa: F401
            return True
        except ImportError:
            logger.warning("LangChain ReAct components not available - ReAct agent will use fallback logic")
            return False

    async def process_input(self, input_data: Any) -> Dict[str, Any]:
        """
        Process input using memory-enhanced ReAct pattern: Reason about the problem, then take actions.
        Uses memory to learn from past interactions and improve performance.

        Args:
            input_data: Input data containing the task or query

        Returns:
            Dict containing the reasoning process and final result
        """
        if not isinstance(input_data, dict):
            input_data = {"query": str(input_data), "context": {}}

        query = input_data.get("query", "")
        context = input_data.get("context", {})

        logger.info(f"Memory-enhanced ReAct Agent processing query: {query[:100]}...")

        # Create problem signature for memory lookup
        problem_signature = self._create_problem_signature(query, context)

        # Check if we have a known solution pattern
        known_solution = self._find_known_solution(problem_signature)
        if known_solution:
            logger.info("Using known solution pattern from memory")
            return await self._apply_known_solution(known_solution, query, context)

        # Use LangChain ReAct if available
        if self.react_available and self.llm:
            result = await self._process_with_langchain_react(query, context)
        else:
            # Fallback to custom ReAct implementation
            result = await self._process_with_custom_react(query, context)

        # Learn from this interaction
        await self._learn_from_interaction(query, context, result, problem_signature)

        return result

    async def _process_with_langchain_react(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process using LangChain's ReAct agent."""
        try:
            # Try to import LangChain components dynamically
            langchain_agents = __import__('langchain.agents', fromlist=['AgentExecutor', 'create_react_agent'])
            langchain_prompts = __import__('langchain.prompts', fromlist=['PromptTemplate'])

            AgentExecutor = getattr(langchain_agents, 'AgentExecutor', None)
            create_react_agent = getattr(langchain_agents, 'create_react_agent', None)
            PromptTemplate = getattr(langchain_prompts, 'PromptTemplate', None)

            if not all([AgentExecutor, create_react_agent, PromptTemplate]):
                raise ImportError("Required LangChain ReAct components not found")

            # Create ReAct prompt template
            react_template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}"""

            prompt = PromptTemplate.from_template(react_template)  # type: ignore

            # Create ReAct agent
            agent = create_react_agent(self.llm, self.tools, prompt)  # type: ignore

            # Create agent executor
            agent_executor = AgentExecutor(  # type: ignore
                agent=agent,
                tools=self.tools,
                verbose=True,
                max_iterations=self.max_iterations,
                handle_parsing_errors=True
            )

            # Execute the agent
            result = await agent_executor.ainvoke({"input": query})

            return {
                "success": True,
                "result": result.get("output", ""),
                "intermediate_steps": result.get("intermediate_steps", []),
                "method": "langchain_react",
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"LangChain ReAct processing failed: {e}")
            return await self._process_with_custom_react(query, context)

    async def _process_with_custom_react(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback custom ReAct implementation when LangChain ReAct is not available."""
        logger.info("Using custom ReAct implementation")

        reasoning_trace = []
        current_context = context.copy()

        for iteration in range(self.max_iterations):
            # Thought phase: Reason about what to do
            thought = await self._generate_thought(query, current_context, reasoning_trace)

            # Check if we have enough information to answer
            if thought.get("conclusion_reached", False):
                return {
                    "success": True,
                    "result": thought.get("final_answer", ""),
                    "reasoning_trace": reasoning_trace,
                    "method": "custom_react",
                    "iterations_used": iteration + 1,
                    "timestamp": datetime.now().isoformat()
                }

            # Action phase: Decide what action to take
            action = await self._decide_action(thought, self.tools)

            if not action:
                # No action needed or available
                break

            # Execute action
            observation = await self._execute_action(action)

            # Update context with observation
            current_context["last_observation"] = observation
            current_context["iteration"] = iteration + 1

            # Record this step
            reasoning_trace.append({
                "iteration": iteration + 1,
                "thought": thought,
                "action": action,
                "observation": observation
            })

        # If we reach max iterations, provide best effort answer
        final_answer = await self._generate_final_answer(query, current_context, reasoning_trace)

        return {
            "success": True,
            "result": final_answer,
            "reasoning_trace": reasoning_trace,
            "method": "custom_react",
            "iterations_used": self.max_iterations,
            "max_iterations_reached": True,
            "timestamp": datetime.now().isoformat()
        }

    async def _generate_thought(self, query: str, context: Dict[str, Any],
                               reasoning_trace: List[Dict]) -> Dict[str, Any]:
        """Generate a reasoning thought about the current state, enhanced with memory insights."""
        if not self.llm:
            return {
                "thought": "No LLM available for reasoning",
                "conclusion_reached": False,
                "needs_action": False
            }

        # Get memory insights
        memory_insights = self._get_memory_insights(query, context, reasoning_trace)
        tool_recommendations = self.get_tool_recommendations(query, context)

        # Build enhanced prompt with memory insights
        prompt = f"""You are a memory-enhanced ReAct agent. Analyze the current situation and decide what to do next.

Query: {query}

Current Context: {context}

Previous Reasoning Steps: {len(reasoning_trace)}

Memory Insights:
{memory_insights}

Recommended Tools (based on past success): {', '.join(tool_recommendations) if tool_recommendations else 'None available'}

Think step by step:
1. What do I know so far?
2. What information do I still need?
3. What action should I take next? Consider past successful patterns.
4. Do I have enough information to answer the query?

Respond with a JSON-like structure:
- thought: Your reasoning (include memory insights if relevant)
- conclusion_reached: true/false
- needs_action: true/false
- action_type: what kind of action if needed
- confidence: 0-1 scale
- recommended_tools: list of suggested tools based on memory"""

        try:
            response = await self.llm.ainvoke(prompt)
            thought_text = response.content if hasattr(response, 'content') else str(response)

            # Parse the response (simplified parsing)
            thought = {
                "thought": thought_text,
                "conclusion_reached": "conclusion_reached" in thought_text.lower() and "true" in thought_text.lower(),
                "needs_action": "needs_action" in thought_text.lower() and "true" in thought_text.lower(),
                "action_type": "tool_use" if "tool" in thought_text.lower() else "reasoning",
                "confidence": 0.5,
                "memory_enhanced": True
            }

            return thought

        except Exception as e:
            logger.error(f"Thought generation failed: {e}")
            return {
                "thought": f"Error in reasoning: {str(e)}",
                "conclusion_reached": False,
                "needs_action": True,
                "action_type": "tool_use",
                "confidence": 0.1,
                "memory_enhanced": False
            }

    async def _decide_action(self, thought: Dict[str, Any], available_tools: List[Any]) -> Optional[Dict[str, Any]]:
        """Decide what action to take based on the thought, enhanced with memory insights."""
        if not thought.get("needs_action", False):
            return None

        if not available_tools:
            return None

        try:
            # Get tool recommendations from memory
            current_query = thought.get("query", "")
            current_context = thought.get("context", {})

            recommended_tools = self.get_tool_recommendations(current_query, current_context)

            # Score tools based on effectiveness and recommendations
            tool_scores = {}
            for tool in available_tools:
                tool_name = getattr(tool, 'name', str(type(tool).__name__).lower())
                score = 0

                # Base score from memory effectiveness
                if tool_name in self.react_memory['tool_effectiveness']:
                    stats = self.react_memory['tool_effectiveness'][tool_name]
                    total_uses = stats['success_count'] + stats['failure_count']
                    if total_uses > 0:
                        success_rate = stats['success_count'] / total_uses
                        score += success_rate * 50  # Weight success rate heavily

                # Bonus for being recommended
                if tool_name in recommended_tools:
                    score += 25

                # Recency bonus (prefer recently used successful tools)
                # This would be more sophisticated in a full implementation

                tool_scores[tool] = score

            # Select the highest scoring tool
            best_tool = max(tool_scores.keys(), key=lambda t: tool_scores[t])

            # Create action with intelligent input generation
            action_input = await self._generate_action_input(best_tool, thought)

            return {
                "tool": best_tool,
                "action_input": action_input,
                "confidence": tool_scores[best_tool] / 100.0,  # Normalize to 0-1
                "reasoning": f"Selected {getattr(best_tool, 'name', 'tool')} based on memory insights"
            }

        except Exception as e:
            logger.warning(f"Error in intelligent action selection: {e}")
            # Fallback to simple selection
            tool = available_tools[0]
            return {
                "tool": tool,
                "action_input": {"query": "Gather relevant information"}
            }

    async def _generate_action_input(self, tool: Any, thought: Dict[str, Any]) -> Dict[str, Any]:
        """Generate appropriate input for the selected tool based on context and memory."""
        try:
            tool_name = getattr(tool, 'name', str(type(tool).__name__).lower())

            # Use memory of successful tool usage patterns
            successful_uses = []
            for pattern in self.react_memory['successful_patterns'].values():
                trace = pattern.get('result', {}).get('reasoning_trace', [])
                for step in trace:
                    if ('action' in step and 'tool' in step['action'] and
                        getattr(step['action']['tool'], 'name', str(type(step['action']['tool']).__name__)) == tool_name):
                        successful_uses.append(step['action'].get('action_input', {}))

            # If we have successful usage patterns, adapt them
            if successful_uses:
                # Use the most common pattern (simplified - would be more sophisticated)
                common_input = successful_uses[0].copy()

                # Adapt to current thought
                if "query" in thought:
                    common_input["query"] = thought["query"]

                return common_input

            # Default input generation based on tool type
            if "search" in tool_name.lower() or "retrieval" in tool_name.lower():
                return {"query": thought.get("query", "Search for relevant information")}
            elif "analysis" in tool_name.lower():
                return {"data": thought.get("context", {}), "analysis_type": "comprehensive"}
            elif "calculation" in tool_name.lower():
                return {"parameters": thought.get("context", {})}
            else:
                return {"query": thought.get("query", "Perform task")}

        except Exception as e:
            logger.warning(f"Error generating action input: {e}")
            return {"query": "Gather relevant information"}

    async def _execute_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the chosen action."""
        try:
            tool = action.get("tool")
            action_input = action.get("action_input", {})

            if tool is None:
                return {"success": False, "error": "No tool provided"}

            if hasattr(tool, 'ainvoke') and callable(getattr(tool, 'ainvoke', None)):
                # LangChain tool
                result = await tool.ainvoke(action_input)
            elif hasattr(tool, '__call__') and callable(tool):
                # Regular function
                if asyncio.iscoroutinefunction(tool):
                    result = await tool(**action_input)
                else:
                    result = tool(**action_input)
            else:
                result = {"error": "Tool not callable"}

            return {"success": True, "result": result}

        except Exception as e:
            logger.error(f"Action execution failed: {e}")
            return {"success": False, "error": str(e)}

    async def _generate_final_answer(self, query: str, context: Dict[str, Any],
                                   reasoning_trace: List[Dict]) -> str:
        """Generate a final answer based on all reasoning and observations."""
        if not self.llm:
            return "No LLM available to generate final answer"

        # Build summary of reasoning process
        summary = f"Query: {query}\n\nReasoning Process:\n"
        for step in reasoning_trace[-3:]:  # Last 3 steps
            summary += f"- Thought: {step['thought'].get('thought', '')[:100]}...\n"
            summary += f"- Action: {step['action']}\n"
            summary += f"- Observation: {step['observation']}\n\n"

        prompt = f"""Based on the reasoning process below, provide a final answer to the query.

{summary}

Final Answer:"""

        try:
            response = await self.llm.ainvoke(prompt)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            return f"Error generating final answer: {str(e)}"

    def _get_memory_insights(self, query: str, context: Dict[str, Any], reasoning_trace: List[Dict]) -> str:
        """Get relevant insights from memory to enhance reasoning."""
        try:
            insights = []

            # Check for similar past queries
            query_words = set(query.lower().split())
            similar_patterns = []

            for signature, pattern in self.react_memory['successful_patterns'].items():
                pattern_words = set(pattern.get('query', '').lower().split())
                similarity = len(query_words.intersection(pattern_words)) / len(query_words.union(pattern_words))

                if similarity > 0.4:  # 40% word overlap
                    similar_patterns.append((similarity, pattern))

            # Sort by similarity and take top 3
            similar_patterns.sort(reverse=True)
            similar_patterns = similar_patterns[:3]

            if similar_patterns:
                insights.append(f"Found {len(similar_patterns)} similar past queries:")
                for similarity, pattern in similar_patterns:
                    success_rate = "successful" if pattern.get('result', {}).get('success') else "unsuccessful"
                    iterations = pattern.get('result', {}).get('iterations_used', 0)
                    insights.append(f"  - {success_rate} solution in {iterations} iterations (similarity: {similarity:.2f})")

            # Tool effectiveness insights
            tool_stats = []
            for tool_name, stats in self.react_memory['tool_effectiveness'].items():
                total_uses = stats['success_count'] + stats['failure_count']
                if total_uses > 0:
                    success_rate = stats['success_count'] / total_uses
                    tool_stats.append((tool_name, success_rate, total_uses))

            tool_stats.sort(key=lambda x: x[1], reverse=True)  # Sort by success rate

            if tool_stats:
                insights.append("Tool effectiveness (based on past usage):")
                for tool_name, success_rate, total_uses in tool_stats[:5]:  # Top 5
                    insights.append(f"  - {tool_name}: {success_rate:.1%} success rate ({total_uses} uses)")

            # Performance insights
            stats = self.react_memory['performance_stats']
            if stats['total_interactions'] > 0:
                success_rate = stats['successful_solutions'] / stats['total_interactions']
                avg_iterations = stats['average_iterations']
                insights.append(f"Overall performance: {success_rate:.1%} success rate, {avg_iterations:.1f} avg iterations")

            return "\n".join(insights) if insights else "No relevant memory insights available."

        except Exception as e:
            logger.warning(f"Error getting memory insights: {e}")
            return "Memory insights unavailable due to error."

    def _create_problem_signature(self, query: str, context: Dict[str, Any]) -> str:
        """Create a signature for the problem to enable memory lookup."""
        # Create a hash of the query and key context elements
        signature_data = {
            'query': query.lower().strip(),
            'context_keys': sorted(list(context.keys())),
            'context_types': {k: type(v).__name__ for k, v in context.items()}
        }

        signature_str = json.dumps(signature_data, sort_keys=True)
        return hashlib.md5(signature_str.encode()).hexdigest()

    def _find_known_solution(self, problem_signature: str) -> Optional[Dict[str, Any]]:
        """Find a known solution for this problem signature."""
        return self.react_memory['successful_patterns'].get(problem_signature)

    async def _apply_known_solution(self, known_solution: Dict[str, Any], query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply a known solution pattern to the current problem."""
        try:
            # Adapt the known solution to current context
            adapted_solution = known_solution.copy()

            # Update timestamps and metadata
            adapted_solution['applied_at'] = datetime.now().isoformat()
            adapted_solution['original_query'] = query
            adapted_solution['method'] = 'memory_applied'

            # Track usage
            if 'usage_count' not in adapted_solution:
                adapted_solution['usage_count'] = 0
            adapted_solution['usage_count'] += 1

            # Update memory
            signature = self._create_problem_signature(query, context)
            self.react_memory['successful_patterns'][signature] = adapted_solution

            logger.info(f"Applied known solution pattern (used {adapted_solution['usage_count']} times)")
            return adapted_solution

        except Exception as e:
            logger.error(f"Error applying known solution: {e}")
            return {"error": f"Failed to apply known solution: {str(e)}"}

    async def _learn_from_interaction(self, query: str, context: Dict[str, Any],
                                    result: Dict[str, Any], problem_signature: str):
        """Learn from the interaction to improve future performance."""
        try:
            # Update performance stats
            self.react_memory['performance_stats']['total_interactions'] += 1

            # Check if this was a successful interaction
            success = result.get('success', False) or 'error' not in result

            if success:
                self.react_memory['performance_stats']['successful_solutions'] += 1

                # Store successful pattern
                pattern_data = {
                    'query': query,
                    'context': context,
                    'result': result,
                    'learned_at': datetime.now().isoformat(),
                    'usage_count': 1,
                    'iterations_used': result.get('iterations_used', 0)
                }

                self.react_memory['successful_patterns'][problem_signature] = pattern_data

            # Update tool effectiveness
            if 'reasoning_trace' in result:
                for step in result['reasoning_trace']:
                    if 'action' in step and 'observation' in step:
                        action = step['action']
                        observation = step['observation']

                        if isinstance(action, dict) and 'tool' in action:
                            tool_name = getattr(action['tool'], 'name', str(type(action['tool']).__name__))
                            success = observation.get('success', False)

                            tool_stats = self.react_memory['tool_effectiveness'][tool_name]
                            if success:
                                tool_stats['success_count'] += 1
                            else:
                                tool_stats['failure_count'] += 1

            # Store reasoning trajectory for analysis
            trajectory = {
                'query': query,
                'signature': problem_signature,
                'trace': result.get('reasoning_trace', []),
                'success': success,
                'timestamp': datetime.now().isoformat(),
                'iterations': result.get('iterations_used', 0)
            }

            self.react_memory['reasoning_trajectories'].append(trajectory)

            # Keep only last 100 trajectories
            if len(self.react_memory['reasoning_trajectories']) > 100:
                self.react_memory['reasoning_trajectories'] = self.react_memory['reasoning_trajectories'][-100:]

            # Update average iterations
            total_iterations = sum(t.get('iterations', 0) for t in self.react_memory['reasoning_trajectories'])
            self.react_memory['performance_stats']['average_iterations'] = total_iterations / len(self.react_memory['reasoning_trajectories'])

            # Save memory periodically
            if self.react_memory['performance_stats']['total_interactions'] % 10 == 0:
                self._save_react_memory()

        except Exception as e:
            logger.warning(f"Error learning from interaction: {e}")

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about the ReAct agent's memory and learning."""
        return {
            'total_interactions': self.react_memory['performance_stats']['total_interactions'],
            'successful_solutions': self.react_memory['performance_stats']['successful_solutions'],
            'success_rate': (self.react_memory['performance_stats']['successful_solutions'] /
                           max(1, self.react_memory['performance_stats']['total_interactions'])),
            'average_iterations': self.react_memory['performance_stats']['average_iterations'],
            'known_patterns': len(self.react_memory['successful_patterns']),
            'tool_effectiveness': dict(self.react_memory['tool_effectiveness']),
            'stored_trajectories': len(self.react_memory['reasoning_trajectories'])
        }

    def get_tool_recommendations(self, query: str, context: Dict[str, Any]) -> List[str]:
        """Get tool recommendations based on past successful patterns."""
        try:
            # Find similar past queries
            query_words = set(query.lower().split())
            recommendations = []

            for pattern in self.react_memory['successful_patterns'].values():
                pattern_words = set(pattern.get('query', '').lower().split())
                similarity = len(query_words.intersection(pattern_words)) / len(query_words.union(pattern_words))

                if similarity > 0.3:  # 30% word overlap
                    # Extract tools used in this pattern
                    trace = pattern.get('result', {}).get('reasoning_trace', [])
                    for step in trace:
                        if 'action' in step and 'tool' in step['action']:
                            tool = step['action']['tool']
                            tool_name = getattr(tool, 'name', str(type(tool).__name__))
                            if tool_name not in recommendations:
                                recommendations.append(tool_name)

            return recommendations[:5]  # Top 5 recommendations

        except Exception as e:
            logger.warning(f"Error getting tool recommendations: {e}")
            return []