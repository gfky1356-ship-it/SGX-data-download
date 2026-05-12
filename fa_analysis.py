# ============================================================
# FA ANALYSIS SCRIPT — Claude API  (Colab & Local)
#
# Usage:
#   pip install anthropic requests -q
#   python fa_analysis.py AAPL
#   python fa_analysis.py D05.SI
#
# Set your API key:
#   export ANTHROPIC_API_KEY="sk-ant-..."
#   (or add ANTHROPIC_API_KEY to Colab Secrets — key icon in sidebar)
# ============================================================

import os
import sys
import requests
import anthropic

PROMPT_URL = (
    "https://raw.githubusercontent.com/gfky1356-ship-it/"
    "Stock-Financial-Analysis-Script/main/Prompt%20file%20for%20FA%20analysis"
)
MODEL = "claude-sonnet-4-5"


def fetch_prompt(url: str) -> str:
    """Fetch the FA analysis prompt file from GitHub."""
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        return resp.text
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to fetch prompt file: {e}")


def get_api_key() -> str:
    """Get API key from environment or Colab secrets."""
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        try:
            from google.colab import userdata
            key = userdata.get("ANTHROPIC_API_KEY")
        except Exception:
            pass
    if not key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set.\n"
            "  Local : export ANTHROPIC_API_KEY='sk-ant-...'\n"
            "  Colab : add it to Colab Secrets (key icon in sidebar)"
        )
    return key


def analyse(ticker: str) -> None:
    """Run FA analysis on a ticker, streaming the response."""
    print(f"\n{'='*60}")
    print(f"  FA ANALYSIS: {ticker.upper()}")
    print(f"  Model      : {MODEL}")
    print(f"{'='*60}\n")

    # 1. Fetch the prompt framework from GitHub
    print("Fetching prompt framework from GitHub...", end=" ", flush=True)
    system_prompt = fetch_prompt(PROMPT_URL)
    print("OK")

    # 2. Build the user message
    user_message = (
        f"Please perform a full FA analysis for the stock ticker: {ticker.upper()}\n\n"
        "Follow the framework exactly:\n"
        "1. Identify whether it is a US (SEC EDGAR) or Singapore (SGX) stock\n"
        "2. Retrieve and evaluate the latest available financial data\n"
        "3. Score each metric against the region-specific benchmarks\n"
        "4. Provide the qualitative Phase 2 analysis\n"
        "5. Give your final FIRE / WAIT / AVOID recommendation with clear reasoning"
    )

    # 3. Call Claude with streaming + prompt caching on the system prompt block
    client = anthropic.Anthropic(api_key=get_api_key())

    print(f"Analysing {ticker.upper()} with Claude...\n")
    print("-" * 60)

    try:
        with client.messages.stream(
            model=MODEL,
            max_tokens=4096,
            system=[
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},  # cache the prompt file
                }
            ],
            messages=[
                {"role": "user", "content": user_message}
            ],
        ) as stream:
            for text in stream.text_stream:
                print(text, end="", flush=True)

            # Print token usage after the response completes
            final = stream.get_final_message()
            usage = final.usage
            print(f"\n\n{'-'*60}")
            print(f"  Tokens — input: {usage.input_tokens}  |  output: {usage.output_tokens}", end="")
            if getattr(usage, "cache_creation_input_tokens", None):
                print(f"  |  cache_write: {usage.cache_creation_input_tokens}", end="")
            if getattr(usage, "cache_read_input_tokens", None):
                print(f"  |  cache_read: {usage.cache_read_input_tokens}", end="")
            print(f"\n{'='*60}\n")

    except anthropic.AuthenticationError:
        print("\nInvalid API key. Check ANTHROPIC_API_KEY.")
        sys.exit(1)
    except anthropic.RateLimitError:
        print("\nRate limit hit. Wait a moment and retry.")
        sys.exit(1)
    except anthropic.APIStatusError as e:
        print(f"\nAPI error {e.status_code}: {e.message}")
        sys.exit(1)
    except anthropic.APIConnectionError:
        print("\nConnection error. Check your internet connection.")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        ticker_input = sys.argv[1]
    else:
        ticker_input = input("Enter stock ticker (e.g. AAPL or D05.SI): ").strip()

    if not ticker_input:
        print("No ticker provided.")
        sys.exit(1)

    analyse(ticker_input)
