import os
from typing import Dict, Any, List, TypedDict, Annotated, Sequence
from operator import add
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END

from schemas.events import TaskSpec, RawEvent
from agents.registry import AgentRegistry

load_dotenv()


# ---------------------------------------------------------------------------
# LangGraph State Schema
# ---------------------------------------------------------------------------
class WorkflowState(TypedDict):
    task_id: str
    description: str
    initial_input: Dict[str, Any]
    plan: str
    tool_result: str
    final_output: str
    events: Annotated[List[Dict[str, Any]], add]
    step_counter: int


# ---------------------------------------------------------------------------
# Multi-Agent Workflow Engine
# ---------------------------------------------------------------------------
class MultiAgentWorkflow:
    def __init__(self, model_name: str = "openai/gpt-oss-20b", temperature: float = 0.0):
        self.registry = AgentRegistry()
        api_key = os.getenv("GROQ_API_KEY")
        
        # Fallback dummy responses if key is missing/unassigned during quick testing
        self.llm = ChatGroq(
            groq_api_key=api_key or "gsk_dummy",
            model_name=model_name,
            temperature=temperature,
            max_retries=6
        ) if api_key else None
        
        self.graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(WorkflowState)

        # Define 3 core nodes
        builder.add_node("planner", self._planner_node)
        builder.add_node("tool_agent", self._tool_agent_node)
        builder.add_node("executor", self._executor_node)

        # Wire graph flow: START -> planner -> tool_agent -> executor -> END
        builder.set_entry_point("planner")
        builder.add_edge("planner", "tool_agent")
        builder.add_edge("tool_agent", "executor")
        builder.add_edge("executor", END)

        return builder.compile()

    # -----------------------------------------------------------------------
    # Node 1: Planner
    # -----------------------------------------------------------------------
    def _planner_node(self, state: WorkflowState) -> Dict[str, Any]:
        step = state.get("step_counter", 0) + 1
        agent_spec = self.registry.get_agent("planner")

        prompt = (
            f"You are {agent_spec.role}. Description: {agent_spec.description}.\n"
            f"Task Description: {state['description']}\n"
            f"Task Inputs: {state['initial_input']}\n"
            f"Provide a concise 2-step execution plan."
        )

        if self.llm:
            res = self.llm.invoke([SystemMessage(content=prompt)]).content
        else:
            res = f"Plan for {state['description']}: 1. Search doc 2. Summarize."

        event = RawEvent(
            task_id=state["task_id"],
            agent_id="planner",
            step_id=step,
            event_type="thought",
            content=str(res),
            success=True
        )

        return {
            "plan": str(res),
            "events": [event.model_dump()],
            "step_counter": step
        }

    # -----------------------------------------------------------------------
    # Node 2: Tool Agent
    # -----------------------------------------------------------------------
    def _tool_agent_node(self, state: WorkflowState) -> Dict[str, Any]:
        step = state["step_counter"] + 1
        agent_spec = self.registry.get_agent("tool_agent")
        tool_spec = self.registry.get_tool("doc_search")

        # Mock tool call output for rapid execution
        query = f"Query based on plan: {state['plan'][:50]}"
        tool_output = {
            "result": f"Document context found for task '{state['task_id']}': [Verified relevant reference data]."
        }

        event = RawEvent(
            task_id=state["task_id"],
            agent_id="tool_agent",
            step_id=step,
            event_type="tool_call",
            content=f"Executed tool {tool_spec.name}",
            tool_name=tool_spec.name,
            tool_input={"query": query},
            tool_output=tool_output,
            success=True
        )

        return {
            "tool_result": tool_output["result"],
            "events": [event.model_dump()],
            "step_counter": step
        }

    # -----------------------------------------------------------------------
    # Node 3: Executor / Aggregator
    # -----------------------------------------------------------------------
    def _executor_node(self, state: WorkflowState) -> Dict[str, Any]:
        step = state["step_counter"] + 1
        agent_spec = self.registry.get_agent("executor")

        prompt = (
            f"You are {agent_spec.role}.\n"
            f"Original Task: {state['description']}\n"
            f"Plan Used: {state['plan']}\n"
            f"Tool Output: {state['tool_result']}\n"
            f"Synthesize the final answer clearly."
        )

        if self.llm:
            res = self.llm.invoke([SystemMessage(content=prompt)]).content
        else:
            res = f"Final Answer synthesized from tool output: {state['tool_result']}"

        event = RawEvent(
            task_id=state["task_id"],
            agent_id="executor",
            step_id=step,
            event_type="final_answer",
            content=str(res),
            success=True
        )

        return {
            "final_output": str(res),
            "events": [event.model_dump()],
            "step_counter": step
        }

    # -----------------------------------------------------------------------
    # Public API for Tasks & Sampling
    # -----------------------------------------------------------------------
    def run_task(self, task: TaskSpec) -> List[RawEvent]:
        initial_state: WorkflowState = {
            "task_id": task.task_id,
            "description": task.description,
            "initial_input": task.initial_input,
            "plan": "",
            "tool_result": "",
            "final_output": "",
            "events": [],
            "step_counter": 0
        }

        final_state = self.graph.invoke(initial_state)
        return [RawEvent(**evt) for evt in final_state["events"]]

    def run_task_k_times(self, task: TaskSpec, k: int = 3, temperature: float = 0.7) -> List[List[RawEvent]]:
        """Hook required by Person 3 to compute metric C (Consistency) over K runs."""
        trajectories = []
        # Temporarily adjust temperature for sampling diversity
        original_temp = self.llm.temperature if self.llm else 0.0
        if self.llm:
            self.llm.temperature = temperature

        for _ in range(k):
            trajectories.append(self.run_task(task))

        if self.llm:
            self.llm.temperature = original_temp

        return trajectories