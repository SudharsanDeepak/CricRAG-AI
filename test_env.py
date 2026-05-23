import os
import sys
import argparse

def test_imports():
    print("Testing Library Imports...")
    try:
        import gradio
        print("[OK] Gradio imported successfully!")
    except ImportError as e:
        print(f"[FAIL] Gradio import failed: {e}")
        
    try:
        import chromadb
        print("[OK] ChromaDB imported successfully!")
    except ImportError as e:
        print(f"[FAIL] ChromaDB import failed: {e}")
        
    try:
        import sentence_transformers
        print("[OK] Sentence Transformers imported successfully!")
    except ImportError as e:
        print(f"[FAIL] Sentence Transformers import failed: {e}")
        
    try:
        import langchain
        print("[OK] LangChain imported successfully!")
    except ImportError as e:
        print(f"[FAIL] LangChain import failed: {e}")
        
    try:
        import pypdf
        print("[OK] PyPDF imported successfully!")
    except ImportError as e:
        print(f"[FAIL] PyPDF import failed: {e}")
        
    try:
        import pandas
        print("[OK] Pandas imported successfully!")
    except ImportError as e:
        print(f"[FAIL] Pandas import failed: {e}")

def test_ingestion():
    print("\nTesting Vector DB Ingestion and Retrieval...")
    try:
        from rag_engine import RAGEngine
        engine = RAGEngine()
        
        # Test clear and load sample directory
        engine.clear_database()
        print("[OK] Collection cleared.")
        
        if not os.path.exists("knowledge_base"):
            print("[FAIL] knowledge_base directory not found. Please create it first.")
            return
            
        print("Ingesting knowledge base...")
        files, chunks = engine.ingest_directory("knowledge_base")
        print(f"[OK] Ingestion successful: Loaded {files} files creating {chunks} chunks.")
        
        # Verify stats
        stats = engine.get_db_stats()
        print(f"[OK] Collection Stats: {stats}")
        assert stats["total_chunks"] > 0, "No chunks indexed!"
        
        # Test query
        print("\nTesting Semantic Query...")
        query_str = "highest individual score in IPL"
        results = engine.query(query_str, n_results=1)
        print(f"Query: '{query_str}'")
        if results:
            print(f"[OK] Search Result Success!")
            print(f"   Source: {results[0]['source']}")
            print(f"   Score: {results[0]['score']}")
            print(f"   Content Snippet: {results[0]['content'][:120]}...")
        else:
            print("[FAIL] Query returned no results.")
    except Exception as e:
        print(f"[FAIL] Database testing failed with error: {e}")
        import traceback
        traceback.print_exc()

def test_tools():
    print("\nTesting MCP Tool Execution...")
    try:
        import mcp_agent
        
        # Test player lookup
        print("Testing Player Stats Lookup (Dhoni)...")
        res_player = mcp_agent.player_stats_lookup("MS Dhoni")
        print(res_player)
        assert "5243" in res_player, "Dhoni runs lookup failed!"
        print("[OK] Player Stats tool works!")
        
        # Test rules lookup
        print("\nTesting Rules Lookup (Powerplay)...")
        res_rules = mcp_agent.ipl_rules_lookup("Powerplay")
        print(res_rules)
        assert "first 6 overs" in res_rules.lower(), "Powerplay rule lookup failed!"
        print("[OK] Rules lookup tool works!")
        
        # Test calculator
        print("\nTesting Stats Calculator...")
        res_calc = mcp_agent.stats_calculator("8004 / 252")
        print(res_calc)
        assert "31.76" in res_calc, "Calculator math failed!"
        print("[OK] Stats calculator tool works!")
        
        # Test Offline Agent Loop
        print("\nTesting Offline Reasoner Agent Loop...")
        from rag_engine import RAGEngine
        engine = RAGEngine()
        ans, steps = mcp_agent.run_offline_fallback_agent("Compare Kohli and Dhoni runs and find the difference", engine)
        print("Final Answer:")
        print(ans)
        print("\nSteps Processed:")
        for idx, step in enumerate(steps, 1):
            print(f"Step {idx}: Thought: {step['thought']}")
            print(f"        Call: {step['tool_call']}")
            print(f"        Output Snippet: {step['tool_output'][:100]}...")
            
        assert len(steps) >= 2, "Offline agent loop did not execute multiple steps!"
        print("[OK] Offline Reasoner Agent execution works!")
    except Exception as e:
        print(f"[FAIL] Tool testing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IPL CricRAG Verification Script")
    parser.add_argument("--imports", action="store_true", help="Test library imports only")
    parser.add_argument("--ingest", action="store_true", help="Test database ingestion and search")
    parser.add_argument("--tools", action="store_true", help="Test tool executions and agent reasoning loop")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    
    args = parser.parse_args()
    
    if len(sys.argv) == 1:
        args.all = True
        
    if args.imports or args.all:
        test_imports()
    if args.ingest or args.all:
        test_ingestion()
    if args.tools or args.all:
        test_tools()
