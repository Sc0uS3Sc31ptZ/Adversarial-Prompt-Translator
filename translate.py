import os
import json
import argparse
import logging
from typing import List, Dict, Optional, Tuple
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

import torch
import google.generativeai as genai
from dotenv import load_dotenv
from vllm import LLM, SamplingParams

from utils import get_data, process_output, MODEL_INFO, SUFFIXES
from get_query import get_query_rephrase, get_query_interpretation, get_query_translate

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()


class LLMHandler(ABC):
    """Abstract base class for LLM handlers"""
    
    @abstractmethod
    def ask_llm(self, messages: List[str], max_tokens: int = 1000) -> List[str]:
        """Generate responses for list of prompts"""
        pass


class LocalLLMHandler(LLMHandler):
    """Handler for local models using vLLM"""
    
    def __init__(self, model_path: str):
        """Initialize local LLM with vLLM"""
        logger.info(f"Initializing local LLM: {model_path}")
        self.model = LLM(
            model=model_path,
            enable_prefix_caching=True,
            tensor_parallel_size=torch.cuda.device_count(),
            max_model_len=10000
        )
        logger.info("Local LLM initialized successfully")
    
    def ask_llm(self, messages: List[str], max_tokens: int = 1000) -> List[str]:
        """Generate responses using vLLM"""
        params = {
            "temperature": 1,
            "top_p": 0.9,
            "max_tokens": max_tokens,
        }
        sampling_params = SamplingParams(**params)
        outputs = self.model.generate(messages, sampling_params=sampling_params)
        return [output.outputs[0].text for output in outputs]


class GeminiHandler(LLMHandler):
    """Handler for Google's Gemini API"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.5-flash",
        temperature: float = 1.0,
        max_tokens: int = 1000,
        top_p: float = 0.9
    ):
        """Initialize Gemini API handler"""
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment or parameters")
        
        genai.configure(api_key=api_key)
        self.model_name = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        logger.info(f"Gemini handler initialized with model: {model}")
    
    def ask_llm(self, messages: List[str], max_tokens: Optional[int] = None) -> List[str]:
        """Generate responses using Gemini API"""
        max_tokens = max_tokens or self.max_tokens
        responses = []
        
        for idx, message in enumerate(messages):
            try:
                model = genai.GenerativeModel(self.model_name)
                response = model.generate_content(
                    message,
                    generation_config=genai.types.GenerationConfig(
                        temperature=self.temperature,
                        top_p=self.top_p,
                        max_output_tokens=max_tokens,
                    )
                )
                responses.append(response.text)
                logger.debug(f"Generated response {idx + 1}/{len(messages)}")
            except Exception as e:
                logger.error(f"Error generating response for message {idx + 1}: {e}")
                responses.append("")
        
        return responses


def get_llm_handler(translator: str) -> Tuple[LLMHandler, Tuple[str, str]]:
    """Factory function to get appropriate LLM handler"""
    if translator == "gemini":
        handler = GeminiHandler()
        sep = ("", "")  # Gemini doesn't need special separators
    else:
        if translator not in MODEL_INFO:
            raise ValueError(f"Unknown translator: {translator}")
        model_path = MODEL_INFO[translator]["model_path"]
        handler = LocalLLMHandler(model_path)
        sep = MODEL_INFO[translator]["sep"]
    
    return handler, sep


def save_results(save_path: str, data: Dict) -> None:
    """Save results to JSON file"""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"Results saved to {save_path}")


def process_translation_step(
    step_name: str,
    queries: List[str],
    model: LLMHandler,
    max_tokens: int = 1000,
    post_process_fn=None
) -> List[str]:
    """Process a translation step with proper logging"""
    logger.info(f"Processing {step_name}...")
    responses = model.ask_llm(queries, max_tokens=max_tokens)
    
    if post_process_fn:
        responses = [post_process_fn(r) for r in responses]
    
    logger.info(f"Completed {step_name}")
    return responses


def main(args, model: LLMHandler, data_pairs: List[List], suffix: str, sep: Tuple[str, str]):
    """Main translation pipeline"""
    save_dict = {"suffix": suffix, "timestamp": datetime.now().isoformat()}
    
    total_items = len(data_pairs)
    logger.info(f"Starting translation pipeline for {total_items} items")
    
    for i, (goal, _, target) in enumerate(data_pairs):
        logger.info(f"Processing item {i + 1}/{total_items}: {goal[:50]}...")
        
        try:
            # Step 1: Rephrase
            query = get_query_rephrase(goal, target, sep=sep)
            rephrased_prompts = process_translation_step(
                f"Rephrase (item {i + 1})",
                [query] * 2,
                model
            )
            rephrased_prompts = ["1. \"" + t for t in rephrased_prompts]
            rephrased_prompts = process_output(rephrased_prompts)[:10]
            
            if not rephrased_prompts:
                logger.warning(f"No rephrased prompts generated for item {i + 1}")
                continue
            
            # Step 2: Interpretation
            queries = [
                get_query_interpretation(t, suffix, target, sep=sep)
                for t in rephrased_prompts
            ]
            interpretations = process_translation_step(
                f"Interpretation (item {i + 1})",
                queries,
                model
            )
            interpretations = ["1. " + t for t in interpretations]
            
            # Step 3: Translation
            queries = [
                get_query_translate(
                    rephrased_prompts[t],
                    suffix,
                    target,
                    interpretations[t],
                    sep=sep
                )
                for t in range(len(rephrased_prompts))
            ]
            translations = process_translation_step(
                f"Translation (item {i + 1})",
                queries,
                model
            )
            translations = [t.split("\"")[0] for t in translations]
            
            # Save intermediate result
            save_dict[i] = {
                "goal": goal,
                "rephrased_prompt": rephrased_prompts,
                "target": target,
                "suffix": suffix,
                "interpretations": interpretations,
                "translations": translations
            }
            
            # Incremental save for safety
            save_results(args.save_dir, save_dict)
            logger.info(f"Item {i + 1} completed and saved")
            
        except Exception as e:
            logger.error(f"Error processing item {i + 1}: {e}")
            continue
    
    logger.info(f"Translation pipeline completed. Results saved to {args.save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Adversarial Prompt Translator using LLMs"
    )
    parser.add_argument(
        "--translator",
        type=str,
        default="gemini",
        choices=list(MODEL_INFO.keys()) + ["gemini"],
        help="LLM model to use for translation"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="harmbench",
        choices=["harmbench", "advbench"],
        help="Dataset to process"
    )
    parser.add_argument(
        "--suffix-index",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="Which suffix to use (0: concatenation, 1: universal from Llama-3.1-8b, 2: from HarmBench)"
    )
    
    args = parser.parse_args()
    
    # Setup paths
    args.save_dir = f"results/trans_{args.dataset}_{args.translator}.json"
    
    logger.info(f"Configuration: translator={args.translator}, dataset={args.dataset}")
    
    # Load data
    suffix = SUFFIXES[args.suffix_index].strip()
    data_pairs = get_data(args.dataset)
    logger.info(f"Loaded {len(data_pairs)} data pairs from {args.dataset}")
    
    # Get LLM handler
    model, sep = get_llm_handler(args.translator)
    
    # Run translation pipeline
    main(args, model, data_pairs, suffix, sep)
