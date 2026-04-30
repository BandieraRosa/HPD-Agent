from src.core.models import AssessmentResult, TaskOutput
from src.core.state import AgentState
from src.core.observability import get_tracer, TokenTrackerCallback
from src.llm import get_structured_llm, ASSESSMENT_PROMPT


async def first_level_assessment(state: AgentState) -> AgentState:
    """Classify the user's query into simple | complex and write analysis to state."""
    tracer = get_tracer()
    with tracer.span("first_level_assessment") as span_id:
        classifier = get_structured_llm(AssessmentResult)
        prompt = ASSESSMENT_PROMPT.format(query=state["input"])
        result: AssessmentResult = await classifier.ainvoke(prompt)

        tin, tout, model = TokenTrackerCallback.snapshot()
        tracer.record_tokens(span_id, tokens_in=tin, tokens_out=tout, model=model)

    output = TaskOutput(
        node="first_level_assessment",
        result={
            "difficulty": result.difficulty.value,
            "reasoning": result.reasoning,
        },
    )

    return {
        "analysis": result.difficulty,
        "outputs": [*state.get("outputs", []), output],
        "parent_span_id": span_id,
    }
