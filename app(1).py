import streamlit as st
import google.generativeai as genai
import typing_extensions as typing
import json
from typing import List, Dict

# --- CONFIGURATION ---
st.set_page_config(page_title="Math Research Copilot", layout="wide")

# Sidebar for API Key
api_key = st.sidebar.text_input("Enter Google Gemini API Key", type="password")

if not api_key:
    st.warning("Please enter your API Key in the sidebar to start.")
    st.markdown("""
    **Note:** This app uses the latest Gemini models. 
    Make sure your API key has access to:
    - `gemini-2.5-flash`
    """)
    st.stop()

# Configure Gemini
genai.configure(api_key=api_key)

# --- SCHEMA DEFINITION (Crucial for JSON stability) ---
class ReviewSchema(typing.TypedDict):
    score: int
    critique: str
    improved_version: str

# --- AGENT DEFINITIONS ---

def agent_generator(context: str, current_ideas: List[str]) -> str:
    """
    Uses Gemini 2.5 Flash for high-speed, creative brainstorming.
    """
    # System prompt to encourage novelty
    system_instruction = (
        "You are a creative mathematical researcher. "
        "Your goal is to propose NOVEL, non-obvious conjectures or research directions."
    )
    
    # Model Configuration
    model = genai.GenerativeModel(
        model_name='gemini-2.5-flash',
        system_instruction=system_instruction
    )
    
    prompt = f"""
    ORIGINAL CONTEXT:
    "{context}"
    
    PREVIOUSLY ACCEPTED IDEAS:
    {current_ideas}
    
    TASK:
    Propose ONE new, novel research follow-up or conjecture based on the context.
    - Be bold and creative.
    - Connect disparate mathematical concepts if possible.
    - Keep it concise (3-4 sentences).
    - Use standard LaTeX for math (e.g., $x^2$).
    """
    
    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.9  # High temperature for creativity
            )
        )
        return response.text
    except Exception as e:
        return f"Generator Error: {str(e)}"

def agent_critic(idea: str, context: str) -> Dict:
    """
    Uses Gemini 2.5 Preview for deep reasoning and strict validation.
    Enforces a JSON schema to prevent LaTeX parsing errors.
    """
    system_instruction = (
        "You are a strict mathematics reviewer. "
        "You evaluate ideas for rigor, novelty, and clarity. "
        "You output structured JSON data only."
    )
    
    # Try using Gemini 2.5
    model_name = 'gemini-2.5-flash' 
    
    try:
        model = genai.GenerativeModel(
            model_name=model_name, 
            system_instruction=system_instruction
        )
    except:
        # Fallback if the specific preview model isn't available
        model = genai.GenerativeModel(
            model_name='gemini-1.5-pro',
            system_instruction=system_instruction
        )

    prompt = f"""
    CONTEXT:
    "{context}"
    
    PROPOSED IDEA:
    "{idea}"
    
    TASK:
    Evaluate this idea.
    1. Score it from 1-10 based on novelty and rigor.
    2. Write a brief critique explaining the score.
    3. Rewrite the idea into a "improved_version" using formal academic tone and correct LaTeX.
    """
    
    try:
        # structured output enforcement
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1, # Low temperature for precision
                response_mime_type="application/json",
                response_schema=ReviewSchema
            )
        )
        return json.loads(response.text)
    except Exception as e:
        return {
            "score": 0, 
            "critique": f"Critic Error ({model_name}): {str(e)}", 
            "improved_version": idea
        }

# --- THE APP UI ---

st.title("🎓 Math Research Copilot")
st.caption("Powered by Gemini 2.5 Flash (Generator) & Gemini 2.5 (Critic)")

with st.expander("ℹ️ How it works", expanded=False):
    st.markdown("""
    1. **Generator (Gemini 2.5):** Brainstorms a creative follow-up idea.
    2. **Critic (Gemini 2.5):** strictly evaluates the idea.
    3. **Schema Enforcement:** Ensures the AI handles LaTeX backslashes correctly without crashing.
    4. **Loop:** Only high-quality ideas (Score >= 7) are added to the final draft.
    """)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input Abstract")
    original_text = st.text_area("Paste your math notes here:", height=300, 
        value="The Riemann Zeta function $\zeta(s)$ has trivial zeros at negative even integers. The Riemann Hypothesis asserts that all non-trivial zeros lie on the critical line $Re(s) = 1/2$.")
    
    iterations = st.slider("Brainstorming Iterations", 1, 10, 3)
    start_btn = st.button("🚀 Start Research Loop", type="primary")

with col2:
    st.subheader("Research Log")
    log_container = st.container()

# --- MAIN LOGIC ---

if start_btn and original_text:
    
    final_ideas = []
    current_context = original_text
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Container for the final report
    report_area = st.empty()

    for i in range(iterations):
        # Update Progress
        progress = (i + 1) / iterations
        progress_bar.progress(progress)
        
        # 1. GENERATE
        status_text.markdown(f"**Iter {i+1}/{iterations}:** 🧠 *Gemini 2.5 Flash is thinking...*")
        raw_idea = agent_generator(current_context, final_ideas)
        
        # 2. CRITIQUE
        status_text.markdown(f"**Iter {i+1}/{iterations}:** ⚖️ *Gemini 2.5 is reviewing...*")
        review = agent_critic(raw_idea, current_context)
        
        # 3. DISPLAY & DECIDE
        with log_container:
            with st.expander(f"Cycle {i+1}: Score {review.get('score', 0)}/10", expanded=True):
                st.markdown(f"**Draft:** {raw_idea}")
                st.markdown(f"**Critique:** *{review.get('critique')}*")
                
                if review.get('score', 0) >= 7:
                    st.success("✅ Accepted & Added to Context")
                    final_ideas.append(review['improved_version'])
                    # Append to context to influence next generation
                    current_context += f"\n\n[Extension {i+1}]: {review['improved_version']}"
                else:
                    st.error("❌ Rejected")
        
    status_text.success("Research Loop Complete!")
    
    # Show Final Report
    st.divider()
    st.header("📝 Final Research Plan")
    st.markdown(current_context)
    
    # Download Button
    st.download_button(
        label="Download Research Notes",
        data=current_context,
        file_name="research_notes.md",
        mime="text/markdown"
    )
