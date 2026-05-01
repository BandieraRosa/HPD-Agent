from langgraph.graph import StateGraph, END

from src.core.enums import TaskDifficulty
from src.core.state import AgentState
from src.nodes import first_level_assessment, direct_answer, scheduler_node, synthesizer, reviewer


def _route_after_assessment(state: AgentState) -> str:
    """Conditional edge: simple → direct_answer, complex → coordinator."""
    difficulty = state.get("analysis")
    if difficulty == TaskDifficulty.SIMPLE:
        return "direct_answer"
    return "coordinator"


def _route_after_review(state: AgentState) -> str:
    """Conditional edge: reviewer decides next step."""
    decision = state.get("review_decision")
    if decision == "re-execute":
        return "scheduler_node"
    elif decision == "add_tasks":
        return "coordinator"
    return "synthesizer"


def build_graph() -> StateGraph:
    """Assemble the agent graph.

    Simple path:       assessment → direct_answer → END
    Complex path:     assessment → coordinator → scheduler_node → reviewer
                                                          ↑            |
                                                          └────────────┘
                                                    (re-execute / add_tasks)
                                                      → synthesizer → END

    Each node has a single, clear responsibility:
      - coordinator:  LLM planning — decompose into DAG (or re-plan)
      - scheduler:    Kahn orchestration — execute sub-tasks in parallel
      - reviewer:     Quality review — approve, re-execute, or request new tasks
      - synthesizer:  Build streaming prompt from all results
    """
    from src.agents import coordinator

    graph = StateGraph(AgentState)

    graph.add_node("first_level_assessment", first_level_assessment)
    graph.add_node("direct_answer", direct_answer)
    graph.add_node("coordinator", coordinator)
    graph.add_node("scheduler_node", scheduler_node)
    graph.add_node("reviewer", reviewer)
    graph.add_node("synthesizer", synthesizer)

    graph.set_entry_point("first_level_assessment")

    graph.add_conditional_edges(
        source="first_level_assessment",
        path=_route_after_assessment,
        path_map={"direct_answer": "direct_answer", "coordinator": "coordinator"},
    )

    graph.add_edge("direct_answer", END)
    graph.add_edge("coordinator", "scheduler_node")
    graph.add_edge("scheduler_node", "reviewer")
    graph.add_conditional_edges(
        source="reviewer",
        path=_route_after_review,
        path_map={
            "synthesizer": "synthesizer",
            "scheduler_node": "scheduler_node",
            "coordinator": "coordinator",
        },
    )
    graph.add_edge("synthesizer", END)

    return graph
