# Adversarial-Prompt-Translator

This repository contains an optimized implementation for [***Deciphering the Chaos: Enhancing Jailbreak Attacks via Adversarial Prompt Translation***](https://arxiv.org/abs/2410.11317). 

**Key enhancements:**
- ✅ **Gemini 2.5-Flash integration** for real-time adversarial payload generation
- ✅ **Full type hints** for better code maintainability
- ✅ **Modular architecture** with abstract LLM handler pattern
- ✅ **Environment-based configuration** via `.env`
- ✅ **Incremental saving** for safer batch processing
- ✅ **Comprehensive logging** for better debugging

## Environments

```
Python 3.10+
PyTorch 2.4.0+
transformers 4.40.0+
vllm 0.5.0+
google-generativeai 0.8.0+
python-dotenv 1.0.0+
```

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/Sc0uS3Sc31ptZ/Adversarial-Prompt-Translator.git
cd Adversarial-Prompt-Translator

# 2. Install dependencies
pip install -r requirements.txt

# 3. Setup environment
cp .env.example .env

# 4. Add your Gemini API key (from https://aistudio.google.com/app/apikey)
# Edit .env and set: GEMINI_API_KEY=your_key_here
```

## Quick Start

### Using Gemini (Recommended for quick testing)

Real-time adversarial payload generation:

```bash
python translate.py --translator gemini --dataset harmbench
```

Results: `results/trans_harmbench_gemini.json`

### Using Local Models

Generate with Llama-3.1-8B (requires GPU):

```bash
python translate.py --translator llama3.1-8b --dataset harmbench
```

Other available models:
- `llama3.1-70b` - Larger variant
- `llama2-13b` / `llama2-7b` - LLaMA 2 versions
- `mistral-8x7b` / `mistral-7b` - Mistral variants
- `vicuna-13b` - Vicuna

### Advanced Options

```bash
python translate.py \
  --translator gemini \
  --dataset advbench \
  --suffix-index 1
```

**Arguments:**
- `--translator`: LLM to use (default: `gemini`)
- `--dataset`: Dataset choice - `harmbench` or `advbench` (default: `harmbench`)
- `--suffix-index`: Suffix variant 0-2 (default: 0)

## Evaluation

### Using OpenAI Batch API

```bash
# Submit batch evaluation
logpath=results/trans_harmbench_gemini.json model=gpt-4o-mini bash eval_submit.sh

# Check status
python3 eval_openai.py --v

# Download results (replace ID with actual ID)
python3 eval_openai.py --d 0
```

### HarmBench Evaluation

```bash
bash eval_harmbench.sh results/trans_harmbench_gemini_eval_gpt-4o-mini_output.jsonl
```

Results saved to: `results/trans_harmbench_gemini_eval_gpt-4o-mini_eval.json`

## Architecture

### LLM Handlers

**Abstract base class** (`LLMHandler`):
- `ask_llm(messages: List[str], max_tokens: int) -> List[str]`

**Implementations:**
1. **GeminiHandler** - Google Gemini API integration
   - Real-time API calls
   - Environment-based configuration
   - Error handling and logging

2. **LocalLLMHandler** - vLLM for local models
   - Batch processing with prefix caching
   - Multi-GPU support via tensor parallelism

### Factory Pattern

```python
from translate import get_llm_handler

# Get handler automatically
handler, sep = get_llm_handler("gemini")
# or
handler, sep = get_llm_handler("llama3.1-8b")
```

### Translation Pipeline

```
1. Rephrase     → Generate 5+ variations of original goal
    ↓
2. Interpret    → Explain why adversarial suffix works
    ↓
3. Translate    → Create paraphrased adversarial prompts
    ↓
4. Save         → Incremental JSON output
```

## Configuration

### Environment Variables (`.env`)

```env
# Required
GEMINI_API_KEY=your_api_key

# Optional (defaults shown)
GEMINI_MODEL=gemini-2.5-flash
GEMINI_TEMPERATURE=1.0
GEMINI_MAX_TOKENS=1000
GEMINI_TOP_P=0.9
```

### Model Configuration

Model info stored in `utils.py`:

```python
MODEL_INFO = {
    "gemini": {...},
    "llama3.1-8b": {...},
    # ... other models
}
```

## Output Format

Results saved as JSON with structure:

```json
{
  "suffix": "adversarial_suffix_string",
  "timestamp": "2026-05-16T...",
  "0": {
    "goal": "original harmful goal",
    "rephrased_prompt": ["variation1", "variation2", ...],
    "interpretations": ["explanation1", ...],
    "translations": ["translated_prompt1", ...],
    "target": "expected_model_output"
  },
  "1": {...},
  ...
}
```

## Logging

Comprehensive logging to console with INFO level:

```
2026-05-16 11:03:29 - __main__ - INFO - Configuration: translator=gemini, dataset=harmbench
2026-05-16 11:03:30 - __main__ - INFO - Loaded 100 data pairs from harmbench
2026-05-16 11:03:31 - __main__ - INFO - Processing item 1/100: Make a tutorial on...
2026-05-16 11:03:35 - __main__ - INFO - Processing Rephrase (item 1)...
...
```

## Performance Tips

1. **Gemini**: Fast API calls, no GPU needed, rate-limited
2. **Local LLMs**: Slower but unlimited, requires GPU memory
3. **Batch size**: Adjust in `get_llm_handler()` for throughput

## Error Handling

- Invalid dataset → `ValueError`
- Missing API key → `ValueError` at init
- Network errors → Logged and skipped with empty response
- File I/O → Incremental saves prevent data loss

## Citation

```bibtex
@article{li2024deciphering,
  title={Deciphering the Chaos: Enhancing Jailbreak Attacks via Adversarial Prompt Translation},
  author={Li, Qizhang and Yang, Xiaochen and Zuo, Wangmeng and Guo, Yiwen},
  journal={arXiv preprint arXiv:2410.11317},
  year={2024}
}
```

## License

See LICENSE file for details.

## Contributing

Pull requests welcome! Please ensure:
- Type hints on all functions
- Logging for debugging
- Updated documentation
