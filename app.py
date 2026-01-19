from __future__ import annotations

import os
from dataclasses import dataclass

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load local .env if present (optional). Streamlit Cloud will use st.secrets instead.
load_dotenv()


# -----------------------------
# Styling (Cherry + Sakura, light + calm)
# -----------------------------
def inject_css() -> None:
    st.markdown(
        """
        <style>
          /* Softer overall spacing */
          .block-container { padding-top: 2.0rem; padding-bottom: 2.5rem; }

          /* Sakura-like background bloom */
          [data-testid="stAppViewContainer"] {
            background:
              radial-gradient(1200px 600px at 15% 10%, rgba(255, 210, 225, 0.55), rgba(255, 255, 255, 0) 60%),
              radial-gradient(900px 450px at 85% 15%, rgba(255, 230, 238, 0.75), rgba(255, 255, 255, 0) 55%),
              linear-gradient(180deg, rgba(255, 248, 250, 1) 0%, rgba(255, 255, 255, 1) 65%);
          }

          /* Buttons: cherry red, rounded, friendly */
          div.stButton > button {
            border-radius: 14px;
            padding: 0.65rem 1rem;
            font-weight: 600;
            border: 1px solid rgba(210, 4, 45, 0.25);
            background: rgba(210, 4, 45, 0.92);
            color: white;
            transition: transform 0.05s ease-in-out, box-shadow 0.2s ease-in-out;
            box-shadow: 0 6px 18px rgba(210, 4, 45, 0.18);
          }
          div.stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 10px 24px rgba(210, 4, 45, 0.22);
          }

          /* Text areas and inputs: soft border, no harsh contrast */
          textarea, input, [data-baseweb="textarea"], [data-baseweb="input"] {
            border-radius: 14px !important;
          }

          /* Cards */
          .sakura-card {
            padding: 1rem 1.1rem;
            border-radius: 18px;
            border: 1px solid rgba(210, 4, 45, 0.10);
            background: rgba(255, 255, 255, 0.75);
            box-shadow: 0 10px 24px rgba(0,0,0,0.04);
          }

          /* Small cherry divider */
          .cherry-divider {
            height: 4px;
            width: 72px;
            border-radius: 999px;
            background: rgba(210, 4, 45, 0.75);
            margin: 0.25rem 0 0.75rem 0;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------
# Config
# -----------------------------
@dataclass(frozen=True)
class TonePreset:
    label: str
    instruction: str


TONES: dict[str, TonePreset] = {
    "Polite": TonePreset(
        label="Polite",
        instruction=(
            "Rewrite the sentence politely. Keep it short, respectful, and kind. "
            "Do not add extra information."
        ),
    ),
    "Friendly": TonePreset(
        label="Friendly",
        instruction=(
            "Rewrite the sentence in a warm, friendly tone. Keep it natural and gentle. "
            "Do not add extra information."
        ),
    ),
    "Professional": TonePreset(
        label="Professional",
        instruction=(
            "Rewrite the sentence in a professional tone (clear, calm, formal). "
            "Do not add extra information."
        ),
    ),
}


def get_api_key() -> str | None:
    """
    Priority:
    1) Streamlit secrets (Cloud + local if you create .streamlit/secrets.toml)
    2) Environment variable GEMINI_API_KEY
    3) Environment variable GOOGLE_API_KEY (supported by SDK)
    """
    key = None
    try:
        key = st.secrets.get("GEMINI_API_KEY", None)
        if not key:
            key = st.secrets.get("GOOGLE_API_KEY", None)
    except Exception:
        # st.secrets may not be configured; ignore safely
        pass

    if not key:
        key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    return key


@st.cache_resource
def get_client(api_key: str) -> genai.Client:
    # Official SDK supports passing api_key directly:
    # client = genai.Client(api_key="GEMINI_API_KEY")
    return genai.Client(api_key=api_key)


def build_prompt(user_text: str, tone: TonePreset) -> str:
    # Gentle, uplifting system intent (not therapy; just kind wording)
    return f"""
You are a helpful writing assistant. Your goal is to rewrite one sentence with a bright, supportive vibe.
Rules:
- Preserve the original meaning.
- Output ONLY the rewritten sentence. No quotes, no bullet points, no explanations.
- Keep it concise (1 sentence).
- Avoid harsh language or negativity; keep it calm and kind.

Task:
{tone.instruction}

Sentence:
{user_text.strip()}
""".strip()


def call_gemini_rewrite(
    client: genai.Client,
    model: str,
    prompt: str,
    temperature: float,
) -> str:
    response = client.models.generate_content(
        model=model,
        contents=[
            types.Content(
                role="user",
                parts=[types.Part(text=prompt)]
            )
        ],
        config=types.GenerateContentConfig(
            temperature=temperature,
            top_p=0.95,
            max_output_tokens=120,
        ),
    )

    return (response.text or "").strip()



def main() -> None:
    st.set_page_config(
        page_title="ToneCraft",
        page_icon="🌸",
        layout="centered",
        initial_sidebar_state="expanded",
    )

    inject_css()

    st.markdown(
        """
        <div class="sakura-card">
          <h2 style="margin: 0;">ToneCraft</h2>
          <div class="cherry-divider"></div>
          <p style="margin: 0.2rem 0 0.4rem 0;">
            Type one sentence, choose a tone, and get a gentle rewrite.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    # Sidebar controls
    with st.sidebar:
        st.markdown("### 🎛️ Settings")
        tone_key = st.selectbox("Tone", list(TONES.keys()), index=0)
        model = st.selectbox(
            "Gemini model",
            [
                "gemini-2.5-flash-lite",
                "gemini-2.5-flash"
            ],
            index=0,
            help="Flash is fast & cheap. Pro is stronger but slower/costlier.",
        )
        temperature = st.slider(
            "Creativity",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.05,
            help="Lower = more faithful rewrite. Higher = more varied phrasing.",
        )
        st.caption("Tip: Keep Creativity around 0.2–0.4 for clean rewrites.")

    user_text = st.text_area(
        "Your sentence",
        placeholder="Example: 'Send me the file by today.'",
        height=110,
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        generate = st.button("✨ Polish", use_container_width=True)
    with col2:
        clear = st.button("🧼 Clear", use_container_width=True)

    if clear:
        st.session_state.pop("output", None)
        st.rerun()

    if generate:
        if not user_text.strip():
            st.warning("Please type a sentence first 🌸")
            return

        api_key = get_api_key()
        if not api_key:
            st.error(
                "Missing API key. Add `GEMINI_API_KEY` in Streamlit Secrets (or as an environment variable)."
            )
            return

        tone = TONES[tone_key]
        prompt = build_prompt(user_text, tone)

        try:
            client = get_client(api_key)
            with st.spinner("Polishing softly…"):
                out = call_gemini_rewrite(client, model=model, prompt=prompt, temperature=temperature)

            if not out:
                st.error("I didn’t get a response. Please try again.")
                return

            st.session_state["output"] = out

        except Exception as e:
            st.error("Something went wrong while calling Gemini.")
            st.exception(e)

    if "output" in st.session_state:
        st.markdown("### ✅ Polished sentence")
        st.markdown(
            f"""
            <div class="sakura-card" style="font-size: 1.05rem;">
              {st.session_state["output"]}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")
        st.download_button(
            label="⬇️ Download as .txt",
            data=st.session_state["output"],
            file_name="polished_sentence.txt",
            mime="text/plain",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
