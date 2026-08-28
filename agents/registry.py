import yaml
from typing import Dict, List, Any, Optional
from pydantic import BaseModel
from schemas.events import TaskSpec, RawEvent


class AgentSpec(BaseModel):
    agent_id: str
    role: str
    description: str
    allowed_tools: List[str]


class ToolSpec(BaseModel):
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]


class AgentRegistry:
    def __init__(self, config_path: str = "config/agents.yaml"):
        self.agents: Dict[str, AgentSpec] = {}
        self.tools: Dict[str, ToolSpec] = {}
        self.load_from_yaml(config_path)

    def register_agent(self, agent_id: str, role: str, description: str, allowed_tools: List[str]) -> None:
        self.agents[agent_id] = AgentSpec(
            agent_id=agent_id,
            role=role,
            description=description,
            allowed_tools=allowed_tools
        )

    def register_tool(self, tool_name: str, description: str, input_schema: Dict[str, Any], output_schema: Dict[str, Any]) -> None:
        self.tools[tool_name] = ToolSpec(
            name=tool_name,
            description=description,
            input_schema=input_schema,
            output_schema=output_schema
        )

    def get_agent(self, agent_id: str) -> AgentSpec:
        if agent_id not in self.agents:
            raise KeyError(f"Agent '{agent_id}' is not registered in the system.")
        return self.agents[agent_id]

    def get_tool(self, tool_name: str) -> ToolSpec:
        if tool_name not in self.tools:
            raise KeyError(f"Tool '{tool_name}' is not registered in the system.")
        return self.tools[tool_name]

    def load_from_yaml(self, config_path: str) -> None:
        try:
            with open(config_path, "r") as f:
                data = yaml.safe_load(f) or {}

            for agent_id, info in data.get("agents", {}).items():
                self.register_agent(
                    agent_id=agent_id,
                    role=info.get("role", ""),
                    description=info.get("description", ""),
                    allowed_tools=info.get("allowed_tools", [])
                )

            for tool_name, info in data.get("tools", {}).items():
                self.register_tool(
                    tool_name=tool_name,
                    description=info.get("description", ""),
                    input_schema=info.get("input_schema", {}),
                    output_schema=info.get("output_schema", {})
                )
        except FileNotFoundError:
            print(f"Warning: Config file '{config_path}' not found. Starting with an empty registry.")