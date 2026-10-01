import os
import random
import re
import openai
from tenacity import retry, wait_exponential, stop_after_attempt
from utils.utils import is_multimodal_dataset, prepare_multimodal_content

from methods.utils import handle_retry_error, load_config

# Appended to the last user message when requesting a self-reported confidence
# score from models that don't expose logprobs (e.g. Claude via the proxy).
CONFIDENCE_INSTRUCTION = (
    "\n\nAfter giving your reasoning and final answer, on a new final line, "
    "state your confidence in your own answer as a percentage, in exactly this "
    "format: \"Confidence: XX%\" (where XX is a number from 0 to 100)."
)


def _parse_confidence_anll(text: str) -> float:
    """Parse a self-reported "Confidence: XX%" line and convert it to an
    uncertainty score in [0, 1]: uncertainty = 1 - confidence_fraction.

    Low confidence (near 0%) -> uncertainty near 1 (very uncertain).
    High confidence (near 100%) -> uncertainty near 0 (very confident).
    If no confidence line is found, defaults to maximal uncertainty (1.0)
    so an unparseable response is never mistaken for a confident one.
    """
    match = re.search(r'confidence\s*:?\s*(\d{1,3})\s*%', text, re.IGNORECASE)
    if not match:
        return 1.0
    confidence_pct = max(min(float(match.group(1)), 100.0), 0.0)
    conf_frac = confidence_pct / 100.0
    return 1.0 - conf_frac


class MAD:
    """Base class for MAD-style multi-agent methods."""

    def __init__(self, general_config, method_config_name=None):
        self.method_config = {}
        if method_config_name is not None:
            # Get the child class's module path
            child_module_path = os.path.dirname(
                os.path.abspath(self.__class__.__module__.replace(".", "/"))
            )
            self.method_config = load_config(
                os.path.join(
                    child_module_path, "configs", f"{method_config_name}.yaml"
                )
            )

        self.model_api_config = general_config["model_api_config"]
        self.model_name = general_config["model_name"]
        self.model_temperature = general_config["model_temperature"]
        self.model_max_tokens = general_config["model_max_tokens"]
        self.model_timeout = general_config["model_timeout"]
        self.seed = general_config.get("seed", None)
        self.debug = general_config.get("debug", False)

        # Tracking compute costs
        self.token_stats = {}

        self.memory_bank = {}
        self.tools = {}

    def inference(self, sample):
        """
        sample: data sample (dictionary) to be passed to the MAD framework
        """
        query = sample["query"]
        response = self.call_llm(prompt=query)
        return {"response": response}

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        stop=stop_after_attempt(5),
        retry_error_callback=handle_retry_error,
    )
    def call_llm(
        self,
        prompt=None,
        system_prompt=None,
        messages=None,
        model_name=None,
        temperature=None,
        multimodal_content=None,
        return_logprobs=False,
        seed=None,
    ):
        model_key = model_name if model_name is not None else self.model_name
        model_config = self.model_api_config[model_key]
        model_dict = random.choice(model_config["model_list"])
        model_name, model_url, api_key = (
            model_dict["model_name"],
            model_dict["model_url"],
            model_dict["api_key"],
        )
        # Some APIs (e.g. Claude via the proxy) never return logprobs, so requesting
        # the logprobs field would error. For those, when an uncertainty signal is
        # needed, we instead ask the model to self-report a confidence score and
        # derive an uncertainty value from it (see _parse_confidence_anll below).
        supports_logprobs = model_config.get("supports_logprobs", True)
        request_confidence = return_logprobs and not supports_logprobs

        if messages is None:
            assert (
                prompt is not None or multimodal_content is not None
            ), "'prompt' or 'multimodal_content' must be provided if 'messages' is not provided."

            # Handle multimodal content
            if multimodal_content is not None:
                user_content = multimodal_content
            else:
                user_content = prompt

            if system_prompt is not None:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ]
            else:
                messages = [{"role": "user", "content": user_content}]

        if request_confidence and messages:
            # Append the confidence-elicitation instruction to a shallow copy of
            # the last message so we don't mutate the caller's original messages
            # (e.g. an agent's ongoing conversation history).
            messages = list(messages)
            last_msg = dict(messages[-1])
            content = last_msg["content"]
            if isinstance(content, str):
                last_msg["content"] = content + CONFIDENCE_INSTRUCTION
            elif isinstance(content, list):
                last_msg["content"] = content + [{"type": "text", "text": CONFIDENCE_INSTRUCTION}]
            messages[-1] = last_msg

        model_temperature = temperature if temperature is not None else self.model_temperature

        # Ensure strict alternating user/assistant roles (and no system role) for strict models (like Gemma 3)
        import copy
        formatted_messages = []
        for msg in messages:
            role = msg["role"]
            content = copy.deepcopy(msg["content"])
            
            if not formatted_messages:
                if role == "system":
                    formatted_messages.append({"role": "user", "content": content})
                else:
                    formatted_messages.append({"role": role, "content": content})
            else:
                last_msg = formatted_messages[-1]
                target_role = "user" if role == "system" else role
                
                if last_msg["role"] == target_role:
                    if isinstance(last_msg["content"], str) and isinstance(content, str):
                        last_msg["content"] += f"\n\n{content}"
                    elif isinstance(last_msg["content"], list) and isinstance(content, str):
                        last_msg["content"].append({"type": "text", "text": f"\n\n{content}"})
                    elif isinstance(last_msg["content"], str) and isinstance(content, list):
                        last_msg["content"] = [{"type": "text", "text": last_msg["content"] + "\n\n"}] + content
                    elif isinstance(last_msg["content"], list) and isinstance(content, list):
                        last_msg["content"].extend(content)
                else:
                    formatted_messages.append({"role": target_role, "content": content})

        request_dict = {
            "model": model_name,
            "messages": formatted_messages,
            "max_tokens": self.model_max_tokens,
        }
        if "o1" not in model_name:  # OpenAI's o1 models do not support temperature
            request_dict["temperature"] = model_temperature
            
        if seed is not None:
            request_dict["seed"] = seed
        elif self.seed is not None:
            request_dict["seed"] = self.seed

        if return_logprobs and supports_logprobs:
            request_dict["logprobs"] = True
            request_dict["top_logprobs"] = 1

        llm = openai.OpenAI(base_url=model_url, api_key=api_key)
        try:
            completion = llm.chat.completions.create(**request_dict)
            response, num_prompt_tokens, num_completion_tokens = (
                completion.choices[0].message.content,
                completion.usage.prompt_tokens,
                completion.usage.completion_tokens,
            )

            anll = None
            logprobs_list = None
            if request_confidence:
                anll = _parse_confidence_anll(response)
            elif return_logprobs and supports_logprobs and hasattr(completion.choices[0], "logprobs") and completion.choices[0].logprobs:
                logprobs_list = completion.choices[0].logprobs.content
                if logprobs_list:
                    # Calculate Average Negative Log Likelihood (ANLL) for the final answer only
                    target_phrases = [
                        "Final Answer:", "Final answer:", "Final:", "Final Answer", 
                        "Final answer", "Answer:", "answer:", "answer is:", "Answer is:",
                        "\\boxed{"
                    ]
                    
                    token_strings = [getattr(t, 'token', '') for t in logprobs_list]
                    full_str = "".join(token_strings)
                    
                    last_phrase_idx = -1
                    best_end_idx = -1
                    for phrase in target_phrases:
                        idx = full_str.rfind(phrase)
                        if idx != -1 and idx > last_phrase_idx:
                            last_phrase_idx = idx
                            best_end_idx = idx + len(phrase)
                            
                    start_token_idx = 0
                    if best_end_idx != -1:
                        running_len = 0
                        for i, t_str in enumerate(token_strings):
                            if running_len + len(t_str) > best_end_idx:
                                start_token_idx = i
                                break
                            running_len += len(t_str)
                            
                    relevant_logprobs = logprobs_list[start_token_idx:]
                    
                    valid_tokens = [t for t in relevant_logprobs if getattr(t, 'logprob', None) is not None]
                    if not valid_tokens:
                        # Fallback to all tokens if something went wrong
                        valid_tokens = [t for t in logprobs_list if getattr(t, 'logprob', None) is not None]
                        
                    if valid_tokens:
                        nll = sum(-token.logprob for token in valid_tokens)
                        anll = nll / len(valid_tokens)

            # Update token stats on successful call
            if model_name not in self.token_stats:
                self.token_stats[model_name] = {
                    "num_llm_calls": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                }

            self.token_stats[model_name]["num_llm_calls"] += 1
            self.token_stats[model_name]["prompt_tokens"] += num_prompt_tokens
            self.token_stats[model_name]["completion_tokens"] += num_completion_tokens

        finally:
            llm.close()  # TODO: Check if this is necessary

        if not isinstance(response, str):
            raise ValueError(f"Invalid response from LLM: {response}")

        if return_logprobs:
            return response, anll, logprobs_list
        return response

    def get_token_stats(self):
        return self.token_stats

    def prepare_sample_content(self, sample):
        """Prepare content from sample, handling both text and multimodal data"""
        if is_multimodal_dataset(sample):
            multimodal_content = prepare_multimodal_content(sample)
            return None, multimodal_content  # prompt=None, multimodal_content=multimodal_content
        else:
            return sample["query"], None  # prompt=sample["query"], multimodal_content=None

    def optimizing(self, val_data):
        """
        For methods that requires validation data such as GPTSwarm and ADAS
        """
        pass

    def retrieve_memory(self):
        pass

    def update_memory(self):
        pass

    def _perceive_image(self, query, multimodal_content, resolution=None):
        """
        Uses a Vision-Language Model to describe the image before debate.
        If resolution is provided, it applies transform_image_payload.
        """
        from utils.utils import compose_multimodal_input, transform_image_payload
        
        perceiver_model = self.method_config.get("perceiver_model_name", self.model_name)
        perceiver_temp = self.method_config.get("perceiver_temperature", 0.2)
        
        if resolution and resolution != "None":
            payload = transform_image_payload(multimodal_content, resolutions=resolution)
        else:
            payload = multimodal_content
            
        prompt = (
            f"You are an expert image perceiver. "
            f"Please describe the provided image in detail to help answer the following question: {query}\n"
            f"Focus on visual elements relevant to the question."
        )
        messages = [{"role": "user", "content": compose_multimodal_input(prompt, payload)}]
        
        if self.debug:
            print(f">> [Perceiver] Generating description with {perceiver_model} at resolution {resolution}")
            
        description = self.call_llm(
            messages=messages,
            model_name=perceiver_model,
            temperature=perceiver_temp
        )
        
        if self.debug:
            print(f">> [Perceiver Output]:\n{description}\n")
            
        return description

    def get_tool(self):
        pass
