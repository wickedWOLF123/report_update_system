import asyncio
from enrichment_agent.state import InputState, OutputState, State
from enrichment_agent.graph import graph
from enrichment_agent.utils import extract_metadata_from_user_query, generate_search_query

async def main():
    print("🚀 Testing Document Processing Pipeline")
    print("=" * 60)
    
    # Test input
    input_state = InputState(
        topic="Update this Madoc for Nepal to 2025.",
        extraction_schema={
            "type": "object",
            "properties": {
                "target_year": {"type": "string"},
                "country": {"type": "string"},
                "document_type": {"type": "string"}
            },
            "required": ["target_year", "country"]
        },
        document_path="inputs/MA_Nepal_2020.pdf"
    )
    
    print(f"📄 Document Path: {input_state.document_path}")
    print(f"🎯 Topic: {input_state.topic}")
    print(f"📋 Extraction Schema: {input_state.extraction_schema}")
    print("\n" + "=" * 60)
    
    try:
        # Run the graph
        print("🔄 Running document analysis pipeline...")
        output: OutputState = await graph.ainvoke(input_state)
        
        print("\n" + "=" * 60)
        print("📊 FINAL OUTPUT STATE")
        print("=" * 60)
        
        # Print the complete output info
        print("🔍 Complete Output Info:")
        print("-" * 40)
        import json
        print(json.dumps(output.info, indent=2, default=str))
        
        print("\n" + "=" * 60)
        print("📋 DETAILED STATE BREAKDOWN")
        print("=" * 60)
        
        # Extract and print document info
        doc_info = output.info.get("document_info", {})
        print("📄 DOCUMENT INFO:")
        print("-" * 20)
        print(f"Path: {doc_info.get('path')}")
        print(f"Title: {doc_info.get('title')}")
        print(f"File Type: {doc_info.get('file_type')}")
        print(f"Page Count: {doc_info.get('page_count')}")
        print(f"Publication Date: {doc_info.get('publication_date')}")
        print(f"Metadata: {doc_info.get('metadata')}")
        
        # Extract and print document structure
        doc_structure = output.info.get("document_structure", {})
        print("\n📚 DOCUMENT STRUCTURE:")
        print("-" * 25)
        
        # TOC
        toc = doc_structure.get("table_of_contents", [])
        print(f"Table of Contents ({len(toc)} entries):")
        for i, entry in enumerate(toc[:5], 1):  # Show first 5 entries
            print(f"  {i}. {entry.get('title', 'N/A')} (Page {entry.get('page', 'N/A')})")
        if len(toc) > 5:
            print(f"  ... and {len(toc) - 5} more entries")
        
        # References
        references = doc_structure.get("references", [])
        print(f"\nReferences ({len(references)} entries):")
        for i, ref in enumerate(references[:3], 1):  # Show first 3 references
            print(f"  {i}. {ref.get('name', 'N/A')} ({ref.get('year', 'N/A')}) - {ref.get('link', 'N/A')}")
        if len(references) > 3:
            print(f"  ... and {len(references) - 3} more references")
        
        # Tables
        tables = doc_structure.get("tables", {})
        print(f"\nTables:")
        if tables:
            for page_num, page_tables in tables.items():
                print(f"  Page {page_num}: {len(page_tables)} table(s)")
        else:
            print("  No tables found")
        
        # Processing stage
        print(f"\n🔄 Processing Stage: {output.info.get('processing_stage')}")
        print(f"🎯 User Topic: {output.info.get('user_topic')}")
        
        print("\n" + "=" * 60)
        print("✅ PIPELINE TEST COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error running pipeline: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())