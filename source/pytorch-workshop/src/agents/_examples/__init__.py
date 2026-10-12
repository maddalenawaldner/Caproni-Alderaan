from src.agents._examples._basic_reflection import basic_reflection
from src.agents._examples._chatbot import chatbot
from src.agents._examples._code_assistant import code_assistant
from src.agents._examples._network import network
from src.agents._examples._plan_and_execute import plan_and_execute
from src.agents._examples._prompt_generator import prompt_generator
from src.agents._examples._reasoning_without_observation import reasoning_without_observation
from src.agents._examples._reflexion import reflexion
from src.agents._examples._research_assistant import research_assistant
from src.agents._examples._supervisor import supervisor
from src.agents._examples._tree_of_thoughts import tree_of_thoughts

EXAMPLE_AGENTS = {
    "basic_reflection": basic_reflection,
    "chatbot": chatbot,
    "code_assistant": code_assistant,
    "network": network,
    "plan_and_execute": plan_and_execute,
    "prompt_generator": prompt_generator,
    "reasoning_without_observation": reasoning_without_observation,
    "reflexion": reflexion,
    "research_assistant": research_assistant,
    "supervisor": supervisor,
    "tree_of_thoughts": tree_of_thoughts,
}
