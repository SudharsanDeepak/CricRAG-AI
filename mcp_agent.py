import re
import math
from typing import List, Dict, Any, Tuple

# Prefer the newer google.genai package when available, fall back to google.generativeai for compatibility.
try:
    import google.genai as google_genai  # type: ignore
    USING_NEW_SDK = True
except Exception:
    google_genai = None
    USING_NEW_SDK = False

try:
    import google.generativeai as google_legacy_genai  # type: ignore
    USING_LEGACY_SDK = True
except Exception:
    google_legacy_genai = None
    USING_LEGACY_SDK = False

genai = google_genai if USING_NEW_SDK else google_legacy_genai
from rag_engine import RAGEngine


GEMINI_MODEL_CANDIDATES = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash-latest",
    "gemini-1.5-flash",
]


def _candidate_gemini_models() -> List[str]:
    # Return candidate list directly to avoid blocking network requests during discovery
    return GEMINI_MODEL_CANDIDATES


def generate_gemini_text(api_key: str, prompt) -> Tuple[str, str]:
    """Generate text with Gemini using the first supported model we can find."""
    if genai is None:
        raise RuntimeError("No Google GenAI library is installed. Install 'google-genai' or 'google-generativeai'.")

    last_error = None
    
    # 1. New google-genai SDK flow
    if USING_NEW_SDK and google_genai is not None:
        try:
            client = google_genai.Client(api_key=api_key)
            from google.genai import types
            
            # Format inputs to types.Content structure if list of dicts is provided (chat history)
            if isinstance(prompt, list):
                contents = []
                for msg in prompt:
                    role = msg.get("role", "user")
                    if role == "assistant":
                        role = "model"
                    parts = []
                    for p in msg.get("parts", []):
                        if isinstance(p, str):
                            parts.append(types.Part.from_text(text=p))
                        elif isinstance(p, dict) and "text" in p:
                            parts.append(types.Part.from_text(text=p["text"]))
                        else:
                            parts.append(types.Part.from_text(text=str(p)))
                    contents.append(types.Content(role=role, parts=parts))
            else:
                contents = str(prompt)
                
            for model_name in GEMINI_MODEL_CANDIDATES:
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=contents,
                    )
                    return response.text.strip(), model_name
                except Exception as exc:
                    last_error = exc
                    error_text = str(exc).lower()
                    if "404" not in error_text and "not found" not in error_text and "not supported" not in error_text:
                        raise
            
        except Exception as e:
            last_error = e

    # 2. Legacy google-generativeai SDK flow
    if USING_LEGACY_SDK and google_legacy_genai is not None:
        try:
            google_legacy_genai.configure(api_key=api_key)
            for model_name in GEMINI_MODEL_CANDIDATES:
                try:
                    model = google_legacy_genai.GenerativeModel(model_name)
                    response = model.generate_content(prompt)
                    return response.text.strip(), model_name
                except Exception as exc:
                    last_error = exc
                    error_text = str(exc).lower()
                    if "404" not in error_text and "not found" not in error_text and "not supported" not in error_text:
                        raise
        except Exception as e:
            last_error = e

    if last_error is not None:
        raise last_error

    raise RuntimeError("No supported Gemini models were available.")


# Define Player Database directly for fast, exact structural lookups (supporting player_stats_lookup tool)
PLAYER_DB = {
    "ms dhoni": {
        "name": "MS Dhoni",
        "matches": 264,
        "runs": 5243,
        "strike_rate": 137.54,
        "average": 39.12,
        "sixes": 252,
        "trophies": 5,
        "team": "CSK",
        "role": "Wicketkeeper-Batsman & Captain",
        "highlights": "Led CSK to 5 titles. Ultimate finisher with most 20th over runs in IPL."
    },
    "virat kohli": {
        "name": "Virat Kohli",
        "matches": 252,
        "runs": 8004,
        "strike_rate": 131.97,
        "average": 38.66,
        "sixes": 272,
        "centuries": 8,
        "fifties": 55,
        "team": "RCB",
        "role": "Top-Order Batsman",
        "highlights": "All-time leading run scorer in IPL history. Scored 973 runs in 2016 season."
    },
    "rohit sharma": {
        "name": "Rohit Sharma",
        "matches": 257,
        "runs": 6628,
        "strike_rate": 131.14,
        "average": 29.72,
        "sixes": 280,
        "centuries": 2,
        "team": "MI",
        "role": "Opening Batsman & Captain",
        "highlights": "Led MI to 5 titles. Won 6 trophies overall (1 with Deccan Chargers in 2009)."
    },
    "jasprit bumrah": {
        "name": "Jasprit Bumrah",
        "matches": 133,
        "wickets": 165,
        "economy": 7.30,
        "average": 22.51,
        "best": "5/10",
        "team": "MI",
        "role": "Fast Bowler",
        "highlights": "One of the best death bowlers in cricket history. Won 5 titles with MI."
    },
    "yuzvendra chahal": {
        "name": "Yuzvendra Chahal",
        "matches": 160,
        "wickets": 205,
        "economy": 7.84,
        "average": 22.44,
        "best": "5/40",
        "team": "RR",
        "role": "Leg-Spinner",
        "highlights": "All-time leading wicket-taker in IPL history. Won Purple Cap in 2022."
    },
    "chris gayle": {
        "name": "Chris Gayle",
        "matches": 142,
        "runs": 4965,
        "strike_rate": 148.96,
        "average": 39.72,
        "sixes": 357,
        "centuries": 6,
        "team": "RCB / KKR / PBKS",
        "role": "Opening Batsman",
        "highlights": "Scored highest individual score: 175* off 66 balls with 17 sixes against Pune in 2013."
    },
    "ab de villiers": {
        "name": "AB de Villiers",
        "matches": 184,
        "runs": 5162,
        "strike_rate": 151.68,
        "average": 39.70,
        "sixes": 251,
        "centuries": 3,
        "team": "RCB",
        "role": "Middle-Order Batsman (Mr. 360)",
        "highlights": "Won the most Player of the Match awards in IPL history (25 awards)."
    },
    "sunil narine": {
        "name": "Sunil Narine",
        "matches": 177,
        "wickets": 180,
        "economy": 6.73,
        "runs": 1532,
        "strike_rate": 162.88,
        "team": "KKR",
        "role": "Spin All-Rounder",
        "highlights": "Won 3 titles with KKR. Named MVP of tournament three times (2012, 2018, 2024)."
    },
    "andre russell": {
        "name": "Andre Russell",
        "matches": 127,
        "runs": 2484,
        "strike_rate": 174.00,
        "wickets": 115,
        "team": "KKR",
        "role": "Pace All-Rounder",
        "highlights": "Highest career strike rate in IPL. Named MVP twice (2015, 2019)."
    },
    "david warner": {
        "name": "David Warner",
        "matches": 184,
        "runs": 6565,
        "strike_rate": 139.77,
        "average": 40.52,
        "fifties": 62,
        "team": "DC / SRH",
        "role": "Opening Batsman",
        "highlights": "Led SRH to 2016 trophy. Only player to win Orange Cap three times."
    },
    "suresh raina": {
        "name": "Suresh Raina",
        "matches": 205,
        "runs": 5528,
        "strike_rate": 136.76,
        "average": 32.51,
        "team": "CSK",
        "role": "Middle-Order Batsman (Mr. IPL)",
        "highlights": "First player to reach 5,000 IPL runs. Played 158 consecutive matches for CSK."
    }
}

# --- TOOL IMPLEMENTATIONS ---

def ipl_semantic_search(query: str, engine: RAGEngine) -> str:
    """Tool: Searches the vector DB for context."""
    results = engine.query(query, n_results=3)
    if not results:
        return "No relevant vector DB documents found."
    
    response_parts = []
    for r in results:
        response_parts.append(f"[Source: {r['source']}] (Score: {r['score']})\nContent: {r['content']}")
    return "\n\n".join(response_parts)

def player_stats_lookup(player_name: str) -> str:
    """Tool: Looks up structured player statistics directly."""
    name_clean = player_name.lower().strip()
    
    # Try finding close match
    found_key = None
    for key in PLAYER_DB:
        if key in name_clean or name_clean in key:
            found_key = key
            break
            
    if not found_key:
        return f"Player '{player_name}' not found in structured profiles. Please try semantic search."
        
    p = PLAYER_DB[found_key]
    stats_str = f"Player Profile: {p['name']}\n"
    stats_str += f"- Team: {p['team']}\n"
    stats_str += f"- Role: {p['role']}\n"
    for k, v in p.items():
        if k not in ["name", "team", "role", "highlights"]:
            stats_str += f"- {k.replace('_', ' ').capitalize()}: {v}\n"
    stats_str += f"- Bio/Highlights: {p['highlights']}"
    return stats_str

def ipl_rules_lookup(topic: str) -> str:
    """Tool: Returns specific rules from the cricket handbook based on keyword matching."""
    topic_clean = topic.lower().strip()
    rules_data = {
        "powerplay": "Powerplay Rules:\n- First 6 overs of each innings.\n- Only 2 fielders allowed outside the 30-yard circle.\n- Remaining overs: Up to 5 fielders allowed outside.",
        "impact player": "Impact Player Rule:\n- Introduced in 2023.\n- Teams name 5 subs at the toss; can swap 1 player at any stage.\n- Replaced player cannot return.\n- If 4 overseas players are already in XI, sub must be Indian.",
        "super over": "Super Over Tie-Breaker:\n- Used when match scores are tied.\n- Each team plays 1 over (6 balls) or until 2 wickets fall.\n- Team batting second in main match bats first in Super Over.\n- Tied Super Over leads to subsequent Super Overs until a winner is decided.",
        "strategic timeout": "Strategic Timeout Rules:\n- 4 timeouts per match, each lasting 2 minutes 30 seconds.\n- Bowling team: can take between overs 6 and 9.\n- Batting team: can take between overs 11 and 16.",
        "net run rate": "Net Run Rate (NRR) Formula:\n- NRR = (Runs Scored / Overs Faced) - (Runs Conceded / Overs Bowled)\n- If bowled out, full 20 overs are used as overs faced."
    }
    
    for key, val in rules_data.items():
        if key in topic_clean or topic_clean in key:
            return val
            
    return f"Rule topic '{topic}' not found in direct database. Run semantic search to check documents."

def stats_calculator(expression: str) -> str:
    """Tool: Evaluates basic math expressions securely."""
    # Clean expression to allow only safe mathematical characters
    clean_expr = re.sub(r'[^0-9\+\-\*\/\(\)\.\s]', '', expression)
    if not clean_expr.strip():
        return "Invalid math expression."
    try:
        # Evaluate safely
        result = eval(clean_expr, {"__builtins__": None}, {})
        return f"Calculation: {expression} = {result}"
    except Exception as e:
        return f"Error executing calculation: {e}"

# --- AGENT SYSTEM ---

# Schema details to present to the model
TOOLS_METADATA = [
    {
        "name": "ipl_semantic_search",
        "description": "Searches the vector database for notes, rules, player facts, or team history.",
        "parameters": "query (string)"
    },
    {
        "name": "player_stats_lookup",
        "description": "Looks up exact career stats (runs, wickets, strike rate, average, trophies) for major players.",
        "parameters": "player_name (string)"
    },
    {
        "name": "ipl_rules_lookup",
        "description": "Retrieves official regulations for Powerplay, Super Over, Strategic Timeout, Impact Player, or Net Run Rate.",
        "parameters": "topic (string)"
    },
    {
        "name": "stats_calculator",
        "description": "Calculates math equations (e.g. Strike Rate = Runs/Balls * 100 or runs comparison).",
        "parameters": "expression (string)"
    }
]


TEAM_ALIASES = {
    "csk": "Chennai Super Kings",
    "chennai super kings": "Chennai Super Kings",
    "mi": "Mumbai Indians",
    "mumbai indians": "Mumbai Indians",
    "kkr": "Kolkata Knight Riders",
    "kolkata knight riders": "Kolkata Knight Riders",
    "rcb": "Royal Challengers Bangalore",
    "royal challengers bangalore": "Royal Challengers Bangalore",
    "rr": "Rajasthan Royals",
    "rajasthan royals": "Rajasthan Royals",
    "srh": "Sunrisers Hyderabad",
    "sunrisers hyderabad": "Sunrisers Hyderabad",
    "dc": "Delhi Capitals",
    "delhi capitals": "Delhi Capitals",
    "pbks": "Punjab Kings",
    "kings xi punjab": "Punjab Kings",
    "punjab kings": "Punjab Kings",
    "gt": "Gujarat Titans",
    "gujarat titans": "Gujarat Titans",
    "lsg": "Lucknow Super Giants",
    "lucknow super giants": "Lucknow Super Giants",
}


def _collect_unique_search_results(engine: RAGEngine, queries: List[str], n_results: int = 3) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    seen = set()

    for query in queries:
        try:
            matches = engine.query(query, n_results=n_results)
        except Exception:
            continue

        for match in matches:
            key = (match.get("source"), match.get("content"))
            if key in seen:
                continue
            seen.add(key)
            results.append(match)

    return results


def _build_offline_search_queries(query: str) -> List[str]:
    query_lower = query.lower()
    queries = [query]

    year_match = re.search(r"\b(20(?:0[8-9]|1\d|2[0-5]))\b", query_lower)
    if year_match:
        year = year_match.group(1)
        queries.extend([
            f"IPL {year}",
            f"IPL {year} winner orange cap purple cap final",
            f"season {year} champion summary",
        ])

    if any(term in query_lower for term in ["winner", "runner-up", "final", "orange cap", "purple cap", "mvp", "season"]):
        queries.extend([
            "IPL seasons winner runner-up orange cap purple cap MVP",
            "IPL season summaries champions finalists awards",
        ])

    team_match = None
    for alias, team_name in TEAM_ALIASES.items():
        if alias in query_lower:
            team_match = team_name
            break

    if team_match:
        queries.extend([
            team_match,
            f"{team_match} trophies key players home ground",
            f"{team_match} history records",
        ])

    if any(term in query_lower for term in ["record", "highest", "most", "fastest", "lowest", "hat-trick", "hat trick", "sixes", "wickets", "runs", "team total", "chase"]):
        queries.extend([
            "IPL records batting bowling team match records",
            "highest runs wickets sixes team totals fastest centuries",
        ])

    if any(term in query_lower for term in ["player", "batting", "bowling", "all-rounder", "captain", "strike rate", "average"]):
        queries.extend([
            "IPL player profiles runs wickets strike rate average",
            "major IPL players stats highlights",
        ])

    return queries


def _format_offline_search_answer(results: List[Dict[str, Any]]) -> str:
    if not results:
        return "No matching facts found in the local database. Try asking about an IPL season, team, player, or record."

    lines = []
    for result in results[:4]:
        content = result.get("content", "").strip()
        source = result.get("source", "unknown source")
        if content:
            lines.append(f"- *{content}* (Source: {source})")

    if not lines:
        return "No matching facts found in the local database. Try asking about an IPL season, team, player, or record."

    return "\n".join(lines)

def run_offline_fallback_agent(query: str, engine: RAGEngine) -> Tuple[str, List[Dict[str, str]]]:
    """Smart Rule-based routing engine that simulates MCP thoughts and runs actual tools."""
    steps = []
    query_lower = query.lower()
    
    # 1. Look for players in the query
    detected_players = []
    for key in PLAYER_DB:
        # Match "dhoni", "kohli", etc.
        name_parts = key.split()
        # check whole name or last name
        if key in query_lower or name_parts[-1] in query_lower:
            detected_players.append(PLAYER_DB[key]["name"])

    # 2. Look for rules in the query
    detected_rules = []
    for r in ["powerplay", "impact player", "super over", "strategic timeout", "net run rate", "purple cap", "orange cap"]:
        if r in query_lower:
            detected_rules.append(r)

    # STEP 1: GATHER INFO
    context_collected = []
    
    if detected_players:
        player_list_str = ", ".join(detected_players)
        steps.append({
            "thought": f"The query mentions player(s): {player_list_str}. I will retrieve structured stats for them using the `player_stats_lookup` tool.",
            "tool_call": f"player_stats_lookup({detected_players[0]})",
            "tool_output": ""
        })
        # Execute tool
        p_stats = player_stats_lookup(detected_players[0])
        steps[-1]["tool_output"] = p_stats
        context_collected.append(p_stats)
        
        # If second player exists, run a second step
        if len(detected_players) > 1:
            steps.append({
                "thought": f"I also need stats for the second player: {detected_players[1]}.",
                "tool_call": f"player_stats_lookup({detected_players[1]})",
                "tool_output": ""
            })
            p_stats2 = player_stats_lookup(detected_players[1])
            steps[-1]["tool_output"] = p_stats2
            context_collected.append(p_stats2)
            
    elif detected_rules:
        rule_name = detected_rules[0]
        steps.append({
            "thought": f"The query asks about the rule: '{rule_name}'. I will call `ipl_rules_lookup` to fetch the regulation text.",
            "tool_call": f"ipl_rules_lookup('{rule_name}')",
            "tool_output": ""
        })
        r_text = ipl_rules_lookup(rule_name)
        steps[-1]["tool_output"] = r_text
        context_collected.append(r_text)
    else:
        # Default to semantic search
        steps.append({
            "thought": "This query requires general database retrieval. I will use `ipl_semantic_search` to find relevant documents in ChromaDB.",
            "tool_call": f"ipl_semantic_search('{query}')",
            "tool_output": ""
        })
        search_res = ipl_semantic_search(query, engine)
        steps[-1]["tool_output"] = search_res
        context_collected.append(search_res)

    # STEP 2: DO MATH IF NEEDED
    # Check if user asks to compare, calculate, add, subtract
    math_indicators = ["difference", "compare", "add", "total", "combined", "multiply", "divide", "strike rate calculation", "average calculation"]
    if any(ind in query_lower for ind in math_indicators) and detected_players:
        # Let's perform a smart math mock calculation
        p1 = PLAYER_DB.get(detected_players[0].lower())
        p2 = PLAYER_DB.get(detected_players[1].lower()) if len(detected_players) > 1 else None
        
        expr = ""
        val1, val2 = 0, 0
        metric = ""
        
        if "runs" in query_lower:
            val1 = p1.get("runs", 0) if p1 else 0
            val2 = p2.get("runs", 0) if p2 else 0
            metric = "runs"
        elif "strike rate" in query_lower or "sr" in query_lower:
            val1 = p1.get("strike_rate", 0) if p1 else 0
            val2 = p2.get("strike_rate", 0) if p2 else 0
            metric = "strike rate"
        elif "sixes" in query_lower:
            val1 = p1.get("sixes", 0) if p1 else 0
            val2 = p2.get("sixes", 0) if p2 else 0
            metric = "sixes"
        elif "wickets" in query_lower:
            val1 = p1.get("wickets", 0) if p1 else 0
            val2 = p2.get("wickets", 0) if p2 else 0
            metric = "wickets"
            
        if val1 and val2:
            expr = f"{val1} - {val2}" if val1 > val2 else f"{val2} - {val1}"
            steps.append({
                "thought": f"The user wants to compare the {metric} of {p1['name']} ({val1}) and {p2['name']} ({val2}). I will use `stats_calculator` to compute the difference.",
                "tool_call": f"stats_calculator('{expr}')",
                "tool_output": ""
            })
            calc_res = stats_calculator(expr)
            steps[-1]["tool_output"] = calc_res
            context_collected.append(calc_res)
        elif val1 and "strike rate calculation" in query_lower:
            # calculate strike rate from runs and balls if they are somehow specified or mock
            pass

    # STEP 3: SYNTHESIZE ANSWER
    # Generate final answer grounded on context
    final_context = "\n\n".join(context_collected)
    
    # We formulate a nice grounded answer programmatically
    if detected_players:
        p1_name = detected_players[0]
        p1_data = PLAYER_DB[p1_name.lower()]
        
        if len(detected_players) > 1:
            p2_name = detected_players[1]
            p2_data = PLAYER_DB[p2_name.lower()]
            
            if "strike rate" in query_lower:
                higher_sr = p1_name if p1_data["strike_rate"] > p2_data["strike_rate"] else p2_name
                lower_sr = p2_name if higher_sr == p1_name else p1_name
                diff = abs(p1_data["strike_rate"] - p2_data["strike_rate"])
                answer = f"**AI Cricket Assistant Answer (Offline Demo Mode)**:\n\nBased on the retrieved profile stats:\n- **{p1_name}** has a strike rate of **{p1_data['strike_rate']}**.\n- **{p2_name}** has a strike rate of **{p2_data['strike_rate']}**.\n\n**{higher_sr}** has the higher strike rate by **{diff:.2f}** points compared to **{lower_sr}**.\n\n*(Grounded context: {p1_data['highlights']} | {p2_data['highlights']})*"
            elif "runs" in query_lower:
                higher_runs = p1_name if p1_data.get("runs", 0) > p2_data.get("runs", 0) else p2_name
                diff = abs(p1_data.get("runs", 0) - p2_data.get("runs", 0))
                answer = f"**AI Cricket Assistant Answer (Offline Demo Mode)**:\n\nLooking at the historical batting records:\n- **{p1_name}** has scored **{p1_data.get('runs', 0)}** runs.\n- **{p2_name}** has scored **{p2_data.get('runs', 0)}** runs.\n\n**{higher_runs}** leads by **{diff}** runs."
            else:
                answer = f"**AI Cricket Assistant Answer (Offline Demo Mode)**:\n\nHere is a comparison of {p1_name} and {p2_name}:\n\n1. **{p1_name}**:\n   - Role: {p1_data['role']}\n   - Key Fact: {p1_data['highlights']}\n\n2. **{p2_name}**:\n   - Role: {p2_data['role']}\n   - Key Fact: {p2_data['highlights']}"
        else:
            answer = f"**AI Cricket Assistant Answer (Offline Demo Mode)**:\n\nHere is the retrieved profile details for **{p1_name}**:\n- **Team**: {p1_data['team']}\n- **Role**: {p1_data['role']}\n- **Stats**: Matches: {p1_data.get('matches')}, Runs: {p1_data.get('runs', 'N/A')}, Wickets: {p1_data.get('wickets', 'N/A')}, Strike Rate: {p1_data.get('strike_rate', 'N/A')}\n- **Key Achievement**: {p1_data['highlights']}"
            
    elif detected_rules:
        r_name = detected_rules[0]
        answer = f"**AI Cricket Assistant Answer (Offline Demo Mode)**:\n\nBased on the IPL regulations database:\n\n{ipl_rules_lookup(r_name)}"
    else:
        # Broader cricket search answer formulation
        search_queries = _build_offline_search_queries(query)
        results = _collect_unique_search_results(engine, search_queries, n_results=3)
        answer = "**AI Cricket Assistant Answer (Offline Demo Mode)**:\n\nHere are the most relevant cricket facts I found:\n\n"
        answer += _format_offline_search_answer(results)

    return answer, steps

def run_llm_mcp_agent(query: str, engine: RAGEngine, provider: str, api_key: str, endpoint: str) -> Tuple[str, List[Dict[str, str]]]:
    """Runs a live LLM-powered agent loop executing tools dynamically."""
    steps = []
    
    # Define system instructions and tool definitions
    system_prompt = f"""You are CricRAG AI, a cricket assistant. You have access to local tools to help answer queries.
You must solve the user query step-by-step.
For each step, write your reasoning in a "Thought:" section, and choose one tool to call.
Available Tools:
{[{'name': t['name'], 'description': t['description'], 'params': t['parameters']} for t in TOOLS_METADATA]}

Format your response EXACTLY like this:
Thought: <your reasoning about what to do next>
Action: <tool_name>
Action Input: <input value to tool>

When you have collected all info needed to answer the query, output your final answer:
Final Answer: <your grounded response>

Do NOT output multiple actions in a single step.
"""

    if provider == "Gemini":
        try:
            chat_history = [{"role": "user", "parts": [system_prompt]}, {"role": "model", "parts": ["Understood. I will use the tools provided to answer cricket queries."]}]
            
            current_prompt = query
            max_steps = 4
            used_model_name = ""
            
            for step_idx in range(max_steps):
                # Generate LLM thought and action
                chat_history.append({"role": "user", "parts": [current_prompt]})
                
                # Call model
                llm_response, used_model_name = generate_gemini_text(api_key, chat_history)
                
                # Parse
                thought_match = re.search(r"Thought:\s*(.*?)(?:Action:|$)", llm_response, re.DOTALL)
                action_match = re.search(r"Action:\s*(\w+)", llm_response)
                input_match = re.search(r"Action Input:\s*(.*)", llm_response)
                
                thought = thought_match.group(1).strip() if thought_match else "Reasoning..."
                
                # If Final Answer is reached
                if "Final Answer:" in llm_response:
                    final_ans = llm_response.split("Final Answer:")[-1].strip()
                    return final_ans, steps
                    
                if action_match and input_match:
                    tool_name = action_match.group(1).strip()
                    tool_input = input_match.group(1).strip().strip("'\"")
                    
                    steps.append({
                        "thought": thought,
                        "tool_call": f"{tool_name}({tool_input})",
                        "tool_output": ""
                    })
                    
                    # Execute tool
                    tool_output = ""
                    if tool_name == "ipl_semantic_search":
                        tool_output = ipl_semantic_search(tool_input, engine)
                    elif tool_name == "player_stats_lookup":
                        tool_output = player_stats_lookup(tool_input)
                    elif tool_name == "ipl_rules_lookup":
                        tool_output = ipl_rules_lookup(tool_input)
                    elif tool_name == "stats_calculator":
                        tool_output = stats_calculator(tool_input)
                    else:
                        tool_output = f"Error: Tool '{tool_name}' not recognized."
                        
                    steps[-1]["tool_output"] = tool_output
                    
                    # Add to chat history
                    chat_history.append({"role": "model", "parts": [llm_response]})
                    current_prompt = f"Response: {tool_output}"
                else:
                    # Parse failure or final answer directly
                    if "Final Answer" not in llm_response:
                        return llm_response, steps
                    final_ans = llm_response.split("Final Answer:")[-1].strip()
                    return final_ans, steps
                    
            if used_model_name:
                return f"Reached max tool-calling steps without a final answer. (Gemini model: {used_model_name})", steps
            return "Reached max tool-calling steps without a final answer.", steps
        except Exception as e:
            # Fallback to offline agent on exception
            ans, fallback_steps = run_offline_fallback_agent(query, engine)
            return f"Gemini Error ({e}). Falling back to Offline Simulator.\n\n{ans}", fallback_steps
            
    elif provider == "Ollama":
        # Simulate local Ollama tool calling via HTTP client
        import requests
        try:
            url = f"{endpoint}/api/generate"
            prompt = f"{system_prompt}\nUser Query: {query}\n"
            
            # Simple simulation using Ollama endpoint
            payload = {
                "model": "llama3", # default model
                "prompt": prompt,
                "stream": False
            }
            
            # Since Ollama might be unavailable, we wrap it
            # To simulate a quick agent loop we can make consecutive calls
            # (In standard environments, we query Ollama and parse actions)
            # For robustness, we will perform one call. If it outputs actions, we run it and do one more.
            r = requests.post(url, json=payload, timeout=8)
            if r.status_code == 200:
                res_text = r.json().get("response", "").strip()
                
                # Check for actions
                action_match = re.search(r"Action:\s*(\w+)", res_text)
                input_match = re.search(r"Action Input:\s*(.*)", res_text)
                
                if action_match and input_match:
                    tool_name = action_match.group(1).strip()
                    tool_input = input_match.group(1).strip().strip("'\"")
                    
                    steps.append({
                        "thought": "Ollama requested tool execution.",
                        "tool_call": f"{tool_name}({tool_input})",
                        "tool_output": ""
                    })
                    
                    # Execute tool
                    tool_output = ""
                    if tool_name == "ipl_semantic_search":
                        tool_output = ipl_semantic_search(tool_input, engine)
                    elif tool_name == "player_stats_lookup":
                        tool_output = player_stats_lookup(tool_input)
                    elif tool_name == "ipl_rules_lookup":
                        tool_output = ipl_rules_lookup(tool_input)
                    elif tool_name == "stats_calculator":
                        tool_output = stats_calculator(tool_input)
                    steps[-1]["tool_output"] = tool_output
                    
                    # Generate final answer from Ollama using the tool result
                    second_prompt = f"{prompt}\nThought: Executing tool {tool_name}.\nTool Output: {tool_output}\nFinal Answer:"
                    payload["prompt"] = second_prompt
                    r2 = requests.post(url, json=payload, timeout=8)
                    if r2.status_code == 200:
                        ans = r2.json().get("response", "").strip()
                        return ans, steps
                    else:
                        return f"Ollama second pass failed: {tool_output}", steps
                else:
                    return res_text, steps
            else:
                raise Exception(f"HTTP Status {r.status_code}")
        except Exception as e:
            ans, fallback_steps = run_offline_fallback_agent(query, engine)
            return f"Ollama Connection Error ({e}). Falling back to Offline Simulator.\n\n{ans}", fallback_steps
            
    # Default fallback
    return run_offline_fallback_agent(query, engine)
