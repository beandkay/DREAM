import random
import re

from typing import List, Dict, Any, Optional, Tuple

from methods.mad_base import MAD
from methods.gating_utils import gated_resolution_decision, parse_resolution_str
from utils.utils import compose_multimodal_input, transform_image_payload

SUPPORTED_STRATEGIES = {"io", "cot", "sbp", "ddcot", "ccot", "pot"}

STRATEGY_DESCRIPTIONS = {
    "io": "Direct answer style. Provide a concise reasoning (if necessary) and end with a line starting with 'Final:' followed by the answer.",
    "cot": "Use explicit chain-of-thought reasoning. Think step by step before concluding. End with 'Final:' line.",
    "pot": "Program-of-Thought: write an executable Python program that solves the problem and stores the final numerical/text result in a variable named ans before concluding with 'Final: <answer>'.",
    "ccot": "Compositional chain-of-thought. Decompose the task into intermediate representations (e.g., scene graphs or structured notes) before answering. End with 'Final:' line.",
    "sbp": "Use step-back prompting: first abstract the problem into higher-level concepts, then solve. Finish with 'Final:' line.",
    "ddcot": "Use a two-phase deliberate reasoning: (Phase1) brainstorm multiple solution paths. (Phase2) converge and justify the best. End with 'Final:' line.",
}

JUDGE_SYSTEM_PROMPT = (
    "You are an impartial judge. Given multiple agents' final answers (with reasoning), select the best final answer. "
    "Return JSON with keys: winner_index (int), rationale (string), final_answer (string)."
)

AGENT_BASE_SYSTEM = "You are a helpful reasoning agent. Follow the specified reasoning style strictly."

MAJORITY_VOTE_PROMPT = (
    "You are evaluating peer answers. Given all final answers below, pick the index of the best. "
    "Return JSON {\"vote\": <int>, \"reason\": <short rationale>}"
)

TEXT_STRATEGY_INSTRUCTIONS = {
    "cot": "Solve the problem step by step before giving the final answer.",
    "sbp": "First step back to identify the higher-level concepts or principles involved, then reason forward to solve the problem.",
    "pot": (
        "Write a short executable Python program that solves the problem. "
        "Store the final result in a variable named ans and report its value."
    ),
}

PROMPT_DDCOT_SUBQUESTIONS = """Given the image, question and options, please think step-by-step about the preliminary knowledge to answer the question, deconstruct the problem as completely as possible down to necessary sub-questions. Then with the aim of helping humans answer the original question, try to answer the sub-questions. The expected answering form is as follows:
Sub-questions:
1. <sub-question 1>
2. <sub-question 2>
...

Sub-answers:
1. <sub-answer 1>
2. <sub-answer 2>
...
"""

PROMPT_DDCOT_ANSWER_WITH_SUBQUESTIONS = """The problem can be deconstructed down to sub-questions. 
{subquestion_answers}
"""

PROMPT_CCOT_MAKE_SCENE_GRAPH = """For the provided image and its associated question, generate a scene graph in JSON format that includes the following:
1. Objects that are relevant to answering the question.
2. Object attributes that are relevant to answering the question.
3. Obect relationships that are relevant to answering the question.

Just generate the scene graph in JSON format. Do not say extra words.
"""

PROMPT_CCOT_ANSWER_WITH_SCENE_GRAPH = """The scene graph of the image in JSON format:
{scene_graph}

Use the image and scene graph as context and answer the following question.
"""



PROMPT_EVAL_JSON_ANSWERS = """Here are some candidate answers using different methods. 
1. [
Directly answer the question.
{answer1}
]

2. [
First, get the scene graph of the image in JSON format:
{scene_graph}

Then, use the image and scene graph as context to answer the question.
{answer2}
]

3. [
First, the problem can be deconstructed down to sub-questions. 
{subquestion_answers}

Then, according to the sub-questions and sub-answers to answer the question.
{answer3}
]
"""

PROMPT_EVAL_JSON_INSTRUCTION = '''Please choose the best solution and output your answer in JSON format, with the format as follows: {"Reason": "", "Index": ""}. "Index" in the format should only be the index number of the right solution. Please strictly output in JSON format, do not output irrelevant content.'''

PROMPT_CCOT_SOLUTION = """
First, get the scene graph of the image in JSON format:
{scene_graph}
Then, use the image and scene graph as context to answer the question.
{answer}
"""

PROMPT_DDCOT_SOLUTION = """
First, deconstruct the question down to sub-questions. 
{subquestion_answers}
Then, accord to the sub-questions and sub-answers to answer the question.
{answer}
"""

CHOOSE_ONE_OPTION_PROMPT = """Only one option is correct. Please choose the right option and explain why you choose it. You must answer in the following format. For example, if the right answer is A, you should answer: 
The answer is A. 
Because ...
"""


class DMAD_Main(MAD):
    """DMAD (Diverse Multi-Agent Debate) simplified implementation.

    Core ideas:
      1. Assign each agent a distinct reasoning strategy (prompting template).
      2. Initial generation per agent using its strategy.
      3. Multi-round refinement: each agent sees others' latest answers and may refine (breaking mental set).
      4. Aggregation: judge or majority voting to pick final answer.

    Output dict MUST contain key 'response'. Trace contains per-agent histories and aggregation metadata.
    """

    _message_announced = False

    def __init__(self, general_config, method_config_name=None):
        method_config_name = "config_main" if method_config_name is None else method_config_name
        super().__init__(general_config, method_config_name)
        cfg = getattr(self, "method_config", {}) or {}
        cfg_strategies = cfg.get("strategies")
        self.custom_strategies = [str(s).lower() for s in cfg_strategies] if cfg_strategies else None
        self.text_strategies = [str(s).lower() for s in cfg.get("text_strategies", ["cot", "sbp", "pot"])]
        self.multimodal_strategies = [str(s).lower() for s in cfg.get("multimodal_strategies", ["io", "ccot", "ddcot"])]
        self.max_round = int(cfg.get("max_round", 3))
        self.agent_temperature = float(cfg.get("agent_temperature", 0.5))
        self.judge_temperature = float(cfg.get("judge_temperature", 0.5))
        self.judge_model_name = cfg.get("judge_model_name")
        self.return_all = bool(cfg.get("return_all", True))

        self._validate_strategies(self.text_strategies)
        self._validate_strategies(self.multimodal_strategies)
        if self.custom_strategies:
            self._validate_strategies(self.custom_strategies)

        self.text_agent_profiles = self._build_agent_profiles(
            self.text_strategies, cfg.get("text_agents"), label="text_agents"
        )
        self.multimodal_agent_profiles = self._build_agent_profiles(
            self.multimodal_strategies, cfg.get("multimodal_agents"), label="multimodal_agents"
        )
        self.custom_agent_profiles = (
            self._build_agent_profiles(self.custom_strategies, cfg.get("agents"), label="agents")
            if self.custom_strategies
            else None
        )
    # -------------- Helpers --------------
    def _validate_strategies(self, strategies: List[str]):
        invalid = [s for s in strategies if s not in SUPPORTED_STRATEGIES]
        if invalid:
            raise ValueError(f"Unsupported strategy tags: {invalid}. Supported values: {sorted(SUPPORTED_STRATEGIES)}")

    def _build_agent_profiles(
        self,
        fallback_strategies: Optional[List[str]],
        configured_agents: Optional[List[Dict[str, Any]]],
        label: str,
    ) -> List[Dict[str, Any]]:
        profiles: List[Dict[str, Any]] = []
        fallback_strategies = fallback_strategies or []

        if configured_agents:
            for idx, entry in enumerate(configured_agents):
                entry = entry or {}
                strategy = entry.get("strategy")
                if not strategy:
                    if idx < len(fallback_strategies):
                        strategy = fallback_strategies[idx]
                    else:
                        raise ValueError(f"{label}[{idx}] missing 'strategy' field.")
                strategy = str(strategy).lower()
                self._validate_strategies([strategy])
                profiles.append(
                    {
                        "strategy": strategy,
                        "name": entry.get("name") or f"{strategy.upper()} Agent {idx + 1}",
                        "model_name": entry.get("model_name"),
                        "temperature": entry.get("temperature"),
                    }
                )
        else:
            for idx, strategy in enumerate(fallback_strategies):
                profiles.append(
                    {
                        "strategy": strategy,
                        "name": f"{strategy.upper()} Agent {idx + 1}",
                        "model_name": None,
                        "temperature": None,
                    }
                )

        return profiles

    def _maybe_announce_profiles(self, profiles: List[Dict[str, Any]]):
        hetero_agents = any(
            profile.get("model_name") or profile.get("temperature") is not None for profile in profiles
        )
        judge_override = bool(self.judge_model_name)
        if (hetero_agents or judge_override) and not DMAD_Main._message_announced:
            segments = [
                f"{profile['name']} ({profile['strategy']}) -> "
                f"{profile.get('model_name') or self.model_name}"
                + (f" (temp={profile.get('temperature')})" if profile.get("temperature") is not None else "")
                for profile in profiles
            ]
            if judge_override:
                segments.append(f"Judge -> {self.judge_model_name}")
            mapping = "; ".join(segments)
            print(f"[DMAD] Heterogeneous agents enabled. Model mapping: {mapping}")
            DMAD_Main._message_announced = True

    def _select_agent_profiles(self, is_multimodal: bool) -> List[Dict[str, Any]]:
        if self.custom_agent_profiles:
            return list(self.custom_agent_profiles)
        return list(self.multimodal_agent_profiles if is_multimodal else self.text_agent_profiles)


    def _strategy_header(self, strategy: str) -> str:
        desc = STRATEGY_DESCRIPTIONS.get(strategy, STRATEGY_DESCRIPTIONS["io"])
        return f"[Strategy: {strategy.upper()}]\n{desc}"

    def _init_agent_messages(self, agent_state: Dict[str, Any]):
        if "messages" not in agent_state:
            agent_state["messages"] = [{"role": "system", "content": AGENT_BASE_SYSTEM}]
            agent_state["image_attached"] = False

    def _format_user_content(
        self,
        text: str,
        multimodal_content: Optional[Any],
        agent_state: Dict[str, Any],
    ):
        if multimodal_content is not None and not agent_state.get("image_attached", False):
            agent_state["image_attached"] = True
            return compose_multimodal_input(text, multimodal_content)
        return text

    def _send_with_history(
        self,
        agent_state: Dict[str, Any],
        user_text: str,
        multimodal_content: Optional[Any],
        model_name: Optional[str] = None,
        temperature_override: Optional[float] = None,
        return_logprobs: bool = False,
    ) -> Any:
        self._init_agent_messages(agent_state)
        messages = agent_state["messages"]
        user_content = self._format_user_content(user_text, multimodal_content, agent_state)
        full_messages = messages + [{"role": "user", "content": user_content}]
        call_result = self.call_llm(
            messages=full_messages,
            temperature=self.agent_temperature if temperature_override is None else temperature_override,
            model_name=model_name,
            return_logprobs=return_logprobs,
            seed=(self.seed * 10 + agent_state.get("id", 0) + 1 if getattr(self, "seed", None) is not None else agent_state.get("id", 0) + 1),
        )
        
        if return_logprobs:
            response, anll, logprobs = call_result
        else:
            response = call_result
            
        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": response})
        
        return call_result if return_logprobs else response

    def _apply_refinement_hint(
        self,
        base_text: str,
        peer_summaries: Optional[str],
        previous_answer: Optional[str],
    ) -> str:
        refinement_hint = ""
        if peer_summaries:
            refinement_hint += f"\nPeer Answers:\n{peer_summaries}"
        if previous_answer:
            refinement_hint += (
                f"\nYour Previous Answer:\n{previous_answer}\n"
                "Refine ONLY if you can materially improve correctness or clarity."
            )
        return f"{base_text}{refinement_hint}"

    def _build_agent_prompt(
        self,
        strategy: str,
        query: str,
        peer_summaries: Optional[str] = None,
        previous_answer: Optional[str] = None,
    ) -> str:
        instruction = TEXT_STRATEGY_INSTRUCTIONS.get(strategy, "")
        header = self._strategy_header(strategy)
        base = (
            f"{header}\nUser Query:\n{query}\n"
            f"{instruction}\n"
            "After reasoning, output a final line in the format 'Final: <answer>'."
        )
        return self._apply_refinement_hint(base, peer_summaries, previous_answer)

    def _call_agent(
        self,
        strategy: str,
        query: str,
        peer_summaries: Optional[str],
        previous_answer: Optional[str],
        multimodal_content: Optional[Any],
        agent_state: Dict[str, Any],
        agent_profile: Optional[Dict[str, Any]] = None,
        return_logprobs: bool = False,
    ) -> Tuple[Any, Dict[str, Any]]:
        agent_profile = agent_profile or {}
        model_override = agent_profile.get("model_name")
        temp_override = agent_profile.get("temperature")

        if strategy in {"io", "ccot", "ddcot"}:
            return self._call_multimodal_strategy(
                strategy,
                query,
                peer_summaries,
                previous_answer,
                multimodal_content,
                agent_state,
                model_override,
                temp_override,
                return_logprobs,
            )
        prompt = self._build_agent_prompt(strategy, query, peer_summaries, previous_answer)
        result = self._send_with_history(
            agent_state,
            prompt,
            None,
            model_name=model_override,
            temperature_override=temp_override,
            return_logprobs=return_logprobs,
        )
        if return_logprobs:
            response, anll, _ = result
        else:
            response = result
            
        agent_state["answer"] = response.strip()
        return result, agent_state

    def _call_multimodal_strategy(
        self,
        strategy: str,
        query: str,
        peer_summaries: Optional[str],
        previous_answer: Optional[str],
        multimodal_content: Optional[Any],
        agent_state: Dict[str, Any],
        model_override: Optional[str] = None,
        temp_override: Optional[float] = None,
        return_logprobs: bool = False,
    ) -> Tuple[Any, Dict[str, Any]]:
        if multimodal_content is None:
            raise ValueError("Multimodal strategy requires multimodal content.")
        self._init_agent_messages(agent_state)
        if strategy == "io":
            base = (
                f"{self._strategy_header('io')}\nQuestion:\n{query}\n"
                "Directly answer the question using the provided image and information. "
                "End with 'Final: <answer>'."
            )
            prompt = self._apply_refinement_hint(base, peer_summaries, previous_answer)
            result = self._send_with_history(
                agent_state,
                prompt,
                multimodal_content,
                model_name=model_override,
                temperature_override=temp_override,
                return_logprobs=return_logprobs,
            )
            response = result[0] if return_logprobs else result
            agent_state["answer"] = response.strip()
            return result, agent_state

        if strategy == "ccot":
            sg_prompt = f"Question:\n{query}\n\n{PROMPT_CCOT_MAKE_SCENE_GRAPH}"
            sg_prompt = self._apply_refinement_hint(sg_prompt, peer_summaries, previous_answer)
            scene_graph = self._send_with_history(
                agent_state,
                sg_prompt,
                multimodal_content,
                model_name=model_override,
                temperature_override=temp_override,
            ).strip()
            agent_state["scene_graph"] = scene_graph

            answer_prompt = PROMPT_CCOT_ANSWER_WITH_SCENE_GRAPH.format(scene_graph=scene_graph)
            answer_prompt += f"Question:\n{query}\n\n{CHOOSE_ONE_OPTION_PROMPT}"
            answer_prompt = self._apply_refinement_hint(answer_prompt, peer_summaries, previous_answer)
            result = self._send_with_history(
                agent_state,
                answer_prompt,
                None,
                model_name=model_override,
                temperature_override=temp_override,
                return_logprobs=return_logprobs,
            )
            answer = result[0] if return_logprobs else result
            answer = answer.strip()
            agent_state["answer"] = answer
            merged = PROMPT_CCOT_SOLUTION.format(scene_graph=scene_graph, answer=answer).strip()
            if return_logprobs:
                return (merged, result[1], result[2]), agent_state
            return merged, agent_state

        if strategy == "ddcot":
            sub_prompt = f"Question:\n{query}\n\n{PROMPT_DDCOT_SUBQUESTIONS}"
            sub_prompt = self._apply_refinement_hint(sub_prompt, peer_summaries, previous_answer)
            subanswers = self._send_with_history(
                agent_state,
                sub_prompt,
                multimodal_content,
                model_name=model_override,
                temperature_override=temp_override,
            ).strip()
            agent_state["subquestion_answers"] = subanswers

            answer_prompt = PROMPT_DDCOT_ANSWER_WITH_SUBQUESTIONS.format(subquestion_answers=subanswers)
            answer_prompt += f"\n{CHOOSE_ONE_OPTION_PROMPT}"
            answer_prompt = f"Question:\n{query}\n\n{answer_prompt}"
            answer_prompt = self._apply_refinement_hint(answer_prompt, peer_summaries, previous_answer)
            result = self._send_with_history(
                agent_state,
                answer_prompt,
                None,
                model_name=model_override,
                temperature_override=temp_override,
                return_logprobs=return_logprobs,
            )
            answer = result[0] if return_logprobs else result
            answer = answer.strip()
            agent_state["answer"] = answer
            merged = PROMPT_DDCOT_SOLUTION.format(subquestion_answers=subanswers, answer=answer).strip()
            if return_logprobs:
                return (merged, result[1], result[2]), agent_state
            return merged, agent_state

        # Fall back to text handling if other strategy is specified.
        prompt = self._build_agent_prompt(strategy, query, peer_summaries, previous_answer)
        result = self._send_with_history(
            agent_state,
            prompt,
            None,
            model_name=model_override,
            temperature_override=temp_override,
            return_logprobs=return_logprobs,
        )
        return result, agent_state

    def _multimodal_judge(
        self,
        query: str,
        strategies: List[str],
        agent_states: List[Dict[str, Any]],
        latest_responses: List[str],
        multimodal_content: Optional[Any],
    ) -> Dict[str, Any]:
        state_map = {strat: state for strat, state in zip(strategies, agent_states)}
        response_map = {strat: resp for strat, resp in zip(strategies, latest_responses)}

        io_state = state_map.get("io", {})
        ccot_state = state_map.get("ccot", {})
        ddcot_state = state_map.get("ddcot", {})

        answer1 = response_map.get("io", "")
        answer2 = response_map.get("ccot", "")
        answer3 = response_map.get("ddcot", "")

        scene_graph = ccot_state.get("scene_graph", "")
        subq_answers = ddcot_state.get("subquestion_answers", "")

        prompt_body = PROMPT_EVAL_JSON_ANSWERS.format(
            scene_graph=scene_graph,
            subquestion_answers=subq_answers,
            answer1=answer1,
            answer2=answer2,
            answer3=answer3,
        )
        prompt = f"Question:\n{query}\n\n{prompt_body}\n{PROMPT_EVAL_JSON_INSTRUCTION}"
        if multimodal_content is not None:
            payload = compose_multimodal_input(prompt, multimodal_content)
            raw = self.call_llm(
                multimodal_content=payload,
                system_prompt=AGENT_BASE_SYSTEM,
                temperature=self.judge_temperature,
                model_name=self.judge_model_name,
            )
        else:
            raw = self.call_llm(
                prompt=prompt,
                system_prompt=AGENT_BASE_SYSTEM,
                temperature=self.judge_temperature,
                model_name=self.judge_model_name,
            )
        choice = self._parse_choice(raw)
        candidates = {1: answer1, 2: answer2, 3: answer3}
        selected_answer = candidates[choice]
        return {
            "mode": "judge_select",
            "raw": raw,
            "winner_index": choice - 1 if 1 <= choice <= 3 else 0,
            "final_answer": selected_answer,
            "scene_graph": scene_graph,
            "subquestion_answers": subq_answers,
        }

    def _text_judge(
        self,
        query: str,
        answers: List[str],
    ) -> Dict[str, Any]:
        candidates = [ans.strip() for ans in answers]
        formatted = ["Here are some candidate answers using different methods."]
        for idx, ans in enumerate(candidates, start=1):
            formatted.append(f"{idx}. [\n{ans}\n]\n")
        prompt_body = "\n".join(formatted)
        prompt = f"Question:\n{query}\n\n{prompt_body}\n{PROMPT_EVAL_JSON_INSTRUCTION}"
        raw = self.call_llm(
            prompt=prompt,
            system_prompt=AGENT_BASE_SYSTEM,
            temperature=self.judge_temperature,
            model_name=self.judge_model_name,
        )
        match = re.search(r'"Index"\s*:\s*"?(\d+)"?', raw)
        if match:
            idx = int(match.group(1)) - 1
        else:
            digits = re.findall(r"\b(\d+)\b", raw)
            idx = int(digits[-1]) - 1 if digits else 0
        if idx < 0 or idx >= len(candidates):
            idx = 0
        return {
            "mode": "judge_select",
            "raw": raw,
            "winner_index": idx,
            "final_answer": candidates[idx],
        }

    def _parse_choice(self, raw: str) -> int:
        match = re.search(r'Index\"\s*:\s*\"([123])\"', raw)
        if match:
            return int(match.group(1))
        digits = re.findall(r"\b([123])\b", raw)
        if digits:
            return int(digits[-1])
        return random.randint(1, 3)

    # -------------- Public Inference --------------
    def inference(self, sample: Dict[str, Any]):
        query = sample.get("query") or sample.get("question") or sample.get("input") or ""
        prompt, multimodal_content = self.prepare_sample_content(sample)
        is_multimodal = multimodal_content is not None
        query_for_agent = prompt if prompt is not None else query

        active_profiles = self._select_agent_profiles(is_multimodal)
        if not active_profiles:
            raise ValueError("No agent profiles configured for DMAD.")
        self._maybe_announce_profiles(active_profiles)
        active_strategies = [profile["strategy"] for profile in active_profiles]
        num_agents = len(active_profiles)
        agent_histories: List[List[str]] = [[] for _ in range(num_agents)]
        agent_states: List[Dict[str, Any]] = [{"id": i} for i in range(num_agents)]
        latest_answers: List[str] = [""] * num_agents

        cfg = getattr(self, "method_config", {}) or {}
        routing_strategy = cfg.get("routing_strategy")
        if routing_strategy in ("gated_mix_res", "consensus_scaling", "mix_res"):
            resolutions = cfg.get("resolutions", ["2x", "1x", "0.5x"])
        else:
            resolutions = [None]

        gating_threshold = float(cfg.get("gated_anll_threshold", 0.2))
        gating_metadata = None

        parsed_res = []
        for r in resolutions:
            if isinstance(r, str):
                parsed_res.extend(parse_resolution_str(r))
            else:
                parsed_res.append(r)
        
        agent_resolutions = []
        if routing_strategy == "consensus_scaling":
            agent_resolutions = [str(parsed_res[0]) for _ in range(num_agents)]
        else:
            for i in range(num_agents):
                agent_resolutions.append(str(parsed_res[i]) if i < len(parsed_res) else (str(parsed_res[-1]) if parsed_res else "None"))

        is_gated = routing_strategy == "gated_mix_res" and is_multimodal and multimodal_content is not None
        gated_probe_anlls = None

        if is_gated and len(parsed_res) > 1:
            gated_probe_anlls = []
            agent_answers = []
            for i, profile in enumerate(active_profiles):
                strat = profile["strategy"]
                res_for_probe = parse_resolution_str(agent_resolutions[i]) if agent_resolutions[i] != "None" else None
                active_multimodal = transform_image_payload(
                    multimodal_content, resolutions=res_for_probe
                )
                result, _ = self._call_agent(
                    "cot",
                    query_for_agent,
                    peer_summaries=None,
                    previous_answer=None,
                    multimodal_content=active_multimodal,
                    agent_state={"id": i},
                    agent_profile={**profile, "temperature": 0.0},  # Force greedy decoding
                    return_logprobs=True
                )
                resp, anll, _ = result
                gated_probe_anlls.append(anll)
                agent_answers.append(resp)

            adaptive = cfg.get("adaptive_threshold", False)
            decision = gated_resolution_decision(
                gated_probe_anlls, agent_resolutions, agent_answers, num_agents,
                gating_threshold, adaptive_threshold=adaptive, debug=self.debug,
            )
            routing_strategy = decision["routing_strategy"]
            if routing_strategy == "mix_res":
                agent_resolutions = decision["initial_res_list"]
            else:
                agent_resolutions = decision["new_resolutions"]
            gating_metadata = decision["metadata"]
            # After gating, use consensus_scaling for the debate (no further rotation)
            routing_strategy = "consensus_scaling"

        # --- Generalized Perceiver Step ---
        if is_multimodal and self.method_config.get("use_perceiver", False):
            consensed_res = None
            if routing_strategy == "mix_res" and decision.get("initial_res_list"):
                consensed_res = decision["initial_res_list"][0]
            elif agent_resolutions:
                def get_res_val(r):
                    try:
                        return float(str(r).strip('x'))
                    except Exception:
                        return 0.0
                consensed_res = max(agent_resolutions, key=get_res_val)

            description = self._perceive_image(query, multimodal_content, consensed_res)
            query = f"{query}\n\nVisual Description of the image: {description}"
            query_for_agent = f"{query_for_agent}\n\nVisual Description of the image: {description}"
            
            is_multimodal = False
            multimodal_content = None
            
            # Re-select active profiles since it's now text-only
            active_profiles = self._select_agent_profiles(is_multimodal)
            active_strategies = [profile["strategy"] for profile in active_profiles]

        agent_answer_history = [[] for _ in range(num_agents)]

        # Initial generation (Round 1)
        round1_anlls = []
        for i, profile in enumerate(active_profiles):
            strat = profile["strategy"]
            res_val = parse_resolution_str(agent_resolutions[i]) if agent_resolutions[i] != "None" else None
            if isinstance(res_val, list) and len(res_val) == 1 and res_val[0] == "None":
                res_val = None
                
            active_multimodal = transform_image_payload(
                multimodal_content, resolutions=res_val
            ) if is_multimodal else None
            
            result, agent_states[i] = self._call_agent(
                strat,
                query_for_agent,
                peer_summaries=None,
                previous_answer=None,
                multimodal_content=active_multimodal,
                agent_state=agent_states[i],
                agent_profile=profile,
                return_logprobs=True
            )
            resp, anll, _ = result
            agent_states[i]["latest_anll"] = anll
            round1_anlls.append(anll)
            agent_histories[i].append(resp)
            latest_answers[i] = resp
            agent_answer_history[i].append({
                "round": 1,
                "resolution": agent_resolutions[i],
                "answer": resp,
                "anll": anll,
            })

        rounds_completed = 1

        # Refinement rounds
        for round_idx in range(1, self.max_round):
            if is_multimodal and routing_strategy == "consensus_scaling" and not is_gated:
                for j in range(num_agents):
                    current_res = agent_resolutions[j]
                    if current_res in parsed_res:
                        current_level = parsed_res.index(current_res)
                    else:
                        current_level = -1
                    agent_resolutions[j] = str(parsed_res[(current_level + 1) % len(parsed_res)])

            for i, profile in enumerate(active_profiles):
                strat = profile["strategy"]
                
                # Sort peer summaries by resolution value (highest to lowest resolution)
                peer_indices = [idx for idx in range(num_agents) if idx != i]
                def get_res_val(res_str):
                    res_str = str(res_str)
                    try:
                        return float(res_str.strip('x'))
                    except Exception:
                        return 0.0

                peer_indices.sort(key=lambda idx: get_res_val(agent_resolutions[idx]), reverse=True)

                peer_summary = "\n\n".join(
                    [
                        f"Agent {idx+1} ({active_strategies[idx]}):\n{latest_answers[idx]}"
                        for idx in peer_indices
                    ]
                )
                
                res_val = parse_resolution_str(agent_resolutions[i]) if agent_resolutions[i] != "None" else None
                if isinstance(res_val, list) and len(res_val) == 1 and res_val[0] == "None":
                    res_val = None
                    
                active_multimodal = transform_image_payload(
                    multimodal_content, resolutions=res_val
                ) if is_multimodal else None

                result, agent_states[i] = self._call_agent(
                    strat,
                    query_for_agent,
                    peer_summaries=peer_summary,
                    previous_answer=latest_answers[i],
                    multimodal_content=active_multimodal,
                    agent_state=agent_states[i],
                    agent_profile=profile,
                    return_logprobs=True
                )
                resp, anll, _ = result
                agent_states[i]["latest_anll"] = anll
                agent_histories[i].append(resp)
                latest_answers[i] = resp
                agent_answer_history[i].append({
                    "round": round_idx + 1,
                    "resolution": agent_resolutions[i],
                    "answer": resp,
                    "anll": anll,
                })

            rounds_completed += 1

        # History-aware answer selection with probe-calibrated rescue (aligned with llm_debate).
        # delta = alpha * sigma(0)_ANLL when the gated probe ran for this sample; otherwise
        # falls back to the ratio+floor heuristic, same as llm_debate's aggregate_with_history.
        # Set enable_answer_rescue: false in the method config to disable this (ablation).
        enable_rescue = cfg.get("enable_answer_rescue", True)
        alpha = cfg.get("anll_rescue_alpha", 0.5)
        anll_threshold = cfg.get("anll_fallback_threshold", 1.5)
        anll_fallback_min_anll = cfg.get("anll_fallback_min_anll", 0.15)

        delta = None
        if enable_rescue and gated_probe_anlls:
            valid_probe = [a for a in gated_probe_anlls if a is not None]
            if len(valid_probe) > 1:
                mean_probe = sum(valid_probe) / len(valid_probe)
                variance = sum((a - mean_probe) ** 2 for a in valid_probe) / len(valid_probe)
                sigma0 = variance ** 0.5
                delta = alpha * sigma0

        selected_outputs = []
        for i, history in enumerate(agent_answer_history):
            final_entry = history[-1]
            final_anll = final_entry.get('anll')

            best_prior_entry = None
            min_prior_anll = float('inf')
            for entry in history[:-1]:
                anll = entry.get('anll')
                if anll is not None and anll < min_prior_anll:
                    min_prior_anll = anll
                    best_prior_entry = entry

            rescued = False
            if enable_rescue and final_anll is not None and best_prior_entry is not None:
                if delta is not None:
                    rescued = (final_anll - min_prior_anll) > delta
                else:
                    rescued = (final_anll > min_prior_anll * anll_threshold
                               and final_anll > anll_fallback_min_anll)

            chosen_entry = best_prior_entry if rescued else final_entry
            selected_outputs.append(chosen_entry['answer'].strip())

        final_outputs = selected_outputs

        if is_multimodal:
            judge_multimodal = transform_image_payload(
                multimodal_content, resolutions=parsed_res[-1] if parsed_res else None
            )
            agg = self._multimodal_judge(
                query_for_agent,
                active_strategies,
                agent_states,
                final_outputs,
                judge_multimodal,
            )
        else:
            agg = self._text_judge(query_for_agent, final_outputs)

        trace = {
            "strategies": active_strategies,
            "rounds_completed": rounds_completed,
            "agent_histories": agent_histories if self.return_all else None,
            "agent_answer_history": agent_answer_history,
            "final_answers": final_outputs,
            "aggregation": agg,
        }
        if gating_metadata:
            trace["gating_metadata"] = gating_metadata
        return {"response": agg["final_answer"], "trace": trace}
