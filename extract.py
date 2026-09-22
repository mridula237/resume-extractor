import argparse
import json
import time
import os

import anthropic
from openai import OpenAI
from pypdf import PdfReader

from schema import Resume

CLAUDE_MODEL = "claude-sonnet-4-6"
GPT_MODEL = "gpt-5.4-nano"

claude = anthropic.Anthropic()
gpt = OpenAI()

SYSTEM_PROMPT = """You are a resume parser. Extract ALL information from the resume text and return it as structured data.

IMPORTANT:
- employment: look carefully for work experience, jobs, internships, roles. Check every section. Never return an empty list if there is any work history.
- education: extract graduation year if mentioned anywhere near the degree
- emails and phones: lists, empty list if none
- skills and certifications: lists
- total_years_experience: estimate from employment dates if not stated
- employment summary: 1-2 sentences on what they did
- For missing optional fields use null, for missing lists use []"""


def read_pdf(path: str) -> str:
    reader = PdfReader(path)
    return "\n".join(page.extract_text() for page in reader.pages)


def extract_claude(text: str, cache: bool = False) -> dict:
    system = [{"type": "text", "text": SYSTEM_PROMPT}]
    if cache:
        system[0]["cache_control"] = {"type": "ephemeral"}

    tools = [{
        "name": "extract_resume",
        "description": "Extract structured data from a resume",
        "input_schema": Resume.model_json_schema()
    }]

    start = time.perf_counter()
    r = claude.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=2048,
        system=system,
        tools=tools,
        tool_choice={"type": "tool", "name": "extract_resume"},
        messages=[{"role": "user", "content": f"Parse this resume:\n\n{text}"}],
    )
    latency = time.perf_counter() - start

    tool_block = next(b for b in r.content if b.type == "tool_use")
    data = tool_block.input

    if isinstance(data.get("skills"), dict):
        flat = []
        for v in data["skills"].values():
            flat.extend(v) if isinstance(v, list) else flat.append(v)
        data["skills"] = flat

    return {
        "data": Resume(**data).model_dump(),
        "model": CLAUDE_MODEL,
        "in": r.usage.input_tokens,
        "out": r.usage.output_tokens,
        "latency": latency,
        "cache_read": getattr(r.usage, "cache_read_input_tokens", 0),
    }


def extract_gpt(text: str, cache: bool = False) -> dict:
    start = time.perf_counter()
    r = gpt.beta.chat.completions.parse(
        model=GPT_MODEL,
        max_completion_tokens=2048,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Parse this resume:\n\n{text}"},
        ],
        response_format=Resume,
    )
    latency = time.perf_counter() - start

    return {
        "data": json.loads(r.choices[0].message.content),
        "model": GPT_MODEL,
        "in": r.usage.prompt_tokens,
        "out": r.usage.completion_tokens,
        "latency": latency,
        "cache_read": 0,
    }


def extract(path: str, model: str = "claude", cache: bool = False) -> dict:
    text = read_pdf(path)
    if model == "claude":
        return extract_claude(text, cache)
    elif model == "gpt":
        return extract_gpt(text, cache)
    else:
        raise ValueError(f"Unknown model: {model}. Use 'claude' or 'gpt'.")


def main():
    parser = argparse.ArgumentParser(description="Extract structured data from a resume PDF")
    parser.add_argument("path", help="Path to resume PDF")
    parser.add_argument("--model", default="claude", choices=["claude", "gpt"])
    parser.add_argument("--cache", action="store_true", help="Use prompt caching for system prompt")
    args = parser.parse_args()

    result = extract(args.path, args.model, args.cache)
    print(json.dumps(result["data"], indent=2))
    print(f"\n# model={result['model']} in={result['in']} out={result['out']} "
          f"latency={result['latency']:.2f}s cache_read={result['cache_read']}", flush=True)


if __name__ == "__main__":
    main()