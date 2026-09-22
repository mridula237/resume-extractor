# Resume Extractor

A structured data extraction service that ingests resume PDFs and returns clean JSON conforming to a schema. Built as a demonstration of LLM-powered document parsing with two model backends.

## Schema (13 fields)
- `name`, `emails`, `phones`, `location`
- `total_years_experience`
- `education`: institution, degree, year
- `employment`: company, title, start, end, summary
- `skills`, `certifications`

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...
```

## CLI Usage
```bash
# Extract with Claude (tool use / structured output)
python extract.py path/to/resume.pdf --model claude

# Extract with GPT (structured output)
python extract.py path/to/resume.pdf --model gpt

# Use prompt caching (saves cost on repeated system prompt)
python extract.py path/to/resume.pdf --model claude --cache
```

## API
```bash
uvicorn api:app --reload
# POST /extract?model=claude|gpt
curl -X POST "http://localhost:8000/extract?model=gpt" \
  -F "file=@resume.pdf"
```

Interactive docs at http://localhost:8000/docs

## Docker
```bash
docker build -t resume-extractor .
docker run -p 8000:8000 \
  -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  resume-extractor
```

## Evaluation (30 resumes)

| Model | Avg Accuracy | Avg Latency | Total Cost | Errors |
|---|---|---|---|---|
| Claude Haiku 4.5 | 57.0% | 6.92s | $0.37 | 0 |
| GPT-5.4 nano | 68.1% | 3.52s | $0.026 | 0 |

### Per-field accuracy

| Field | Claude Haiku | GPT-5.4 nano |
|---|---|---|
| name | 93.3% | 93.3% |
| emails | 100.0% | 100.0% |
| phones | 96.7% | 96.7% |
| location | 83.3% | 100.0% |
| total_years_experience | 0.0% | 30.0% |
| education | 47.8% | 75.6% |
| employment | 0.0% | 0.0% |
| skills | 75.3% | 80.7% |
| certifications | 16.7% | 36.7% |

**GPT-5.4 nano is the better choice:** 11% higher accuracy, 2× faster, and 14× cheaper.

**Known limitations of the eval:**
- Employment shows 0% for both models because the Jaccard scorer penalises minor wording differences in summaries. Manual review shows all 3 jobs are correctly extracted.
- `total_years_experience` is hard to score automatically since models estimate differently from the same dates.
- Eval set is synthetic (GPT-generated resumes converted to PDF), which likely inflates GPT's scores.

## Project structure
```
extract.py       CLI + core extraction logic (Claude + GPT)
api.py           FastAPI endpoint POST /extract
schema.py        Pydantic schema (Resume, Education, Employment)
eval.py          Evaluation harness (accuracy, latency, cost)
Dockerfile       Container definition
eval/
  resumes/       30 synthetic resume PDFs
  gold/          Gold-standard JSON labels
  results/       Per-resume per-model results (cached)
  report.md      Full evaluation report
```