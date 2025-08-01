import asyncio
from enrichment_agent.state import InputState, OutputState, State
from enrichment_agent.graph import graph

async def main():
    # Set up a minimal input state
    input_state = InputState(
        topic="Test Document",
        extraction_schema={},  # or your schema if needed
        document_path="inputs/MA_Nepal_2020.pdf",  # Use your test PDF
    )

    # Run the graph (this will trigger document_analysis as the first node)
    output: OutputState = await graph.ainvoke(input_state)

    # Print the resulting state for inspection
    print("==== OUTPUT STATE ====")
    print(output)
    print("==== DOCUMENT INFO ====")
    print(output.info.get("document_info"))
    print("==== DOCUMENT STRUCTURE ====")
    print(output.info.get("document_structure"))

if __name__ == "__main__":
    asyncio.run(main())