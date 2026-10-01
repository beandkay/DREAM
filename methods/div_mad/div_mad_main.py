import re
import json
from copy import deepcopy

from ..mad_base import MAD
from .div_mad_utils import Agent
from .div_mad_prompt import *
from utils.utils import (
    is_multimodal_dataset,
    prepare_multimodal_content,
    compose_multimodal_input,
)
from ..gating_utils import gated_resolution_decision, parse_resolution_str

NAME_LIST = [
    "Affirmative side",
    "Negative side",
    "Moderator",
]

class DivMADMain(MAD):
    _message_announced = False

    def __init__(self, general_config, method_config_name=None):
        method_config_name = "config_main" if method_config_name is None else method_config_name
        super().__init__(general_config, method_config_name)
        self.config = {}
        
        self.num_players = self.method_config["num_players"]
        self.max_round = self.method_config["max_round"]

        self.player_meta_prompt = ""
        self.moderator_prompt = ""
        self.affirmative_prompt = ""
        self.judge_prompt_last2 = ""
        self.agent_profiles = self._build_agent_profiles()
        self.judge_model_name = self.method_config.get("judge_model_name")
        self.judge_temperature = float(self.method_config.get("judge_temperature", 0.5))
        self._maybe_announce_agent_models()

    def _build_agent_profiles(self):
        """Load optional per-agent settings (model_name, temperature, system prompt)."""
        configured_agents = self.method_config.get("agents") or []
        profiles = []

        if configured_agents:
            if len(configured_agents) != self.num_players:
                raise ValueError(
                    f"num_players ({self.num_players}) does not match number of entries in 'agents' ({len(configured_agents)})"
                )
            source = configured_agents
        else:
            source = [{} for _ in range(self.num_players)]

        for idx, entry in enumerate(source):
            entry = entry or {}
            default_name = NAME_LIST[idx] if idx < len(NAME_LIST) else f"Agent {idx + 1}"
            profiles.append(
                {
                    "name": entry.get("name") or default_name,
                    "model_name": entry.get("model_name"),
                    "temperature": entry.get("temperature"),
                }
            )

        return profiles

    def _maybe_announce_agent_models(self):
        """Print a one-time notice when agents use custom models."""
        profile_rows = []
        is_heterogeneous = False
        for profile in self.agent_profiles:
            model_name = profile.get("model_name")
            profile_rows.append(f"{profile['name']} -> {model_name or self.model_name}")
            if model_name:
                is_heterogeneous = True

        judge_override = bool(self.judge_model_name)
        if (is_heterogeneous or judge_override) and not DivMADMain._message_announced:
            mapping = "; ".join(profile_rows)
            judge_str = f"; Judge -> {self.judge_model_name}" if judge_override else ""
            print(f"[Div-MAD] Heterogeneous agents enabled. Model mapping: {mapping}{judge_str}")
            DivMADMain._message_announced = True

    def _call_agent(self, agent, **kwargs):
        """Wrapper around call_llm that honors per-agent model overrides."""
        agent_idx = getattr(agent, "id", 0)
        temp = kwargs.pop("temperature", agent.temperature)
        return self.call_llm(
            model_name=agent.model_name, 
            temperature=temp, 
            seed=(self.seed * 10 + agent_idx + 1 if getattr(self, "seed", None) is not None else agent_idx + 1),
            **kwargs
        )

    def is_valid_answer(self, debate_answer):
        """Check if debate_answer is a valid answer (not a continuation message or side name)"""
        if not debate_answer or not isinstance(debate_answer, str):
            return False
        
        answer_lower = debate_answer.lower().strip()
        
        # Check for continuation messages (case-insensitive)
        continuation_phrases = [
            "the debate continues",
            "debate continues",
            "continue to the next round",
            "continues to the next round",
            "proceed to the next round"
        ]
        
        for phrase in continuation_phrases:
            if phrase in answer_lower:
                return False
        
        # Check if answer is a side name (not a valid answer)
        # "Affirmative" or "Negative" should not be the final answer
        if answer_lower in ["affirmative", "negative", "affirmative or negative"]:
            return False
        
        # If it's not empty and not a continuation message and not a side name, consider it valid
        return len(answer_lower) > 0

    def parse_json_string(self, json_string, default_value=None):
        """Parse JSON string to Python dict with error handling"""
        try:
            # Clean the string first (remove markdown code blocks)
            cleaned = re.sub(r"```json|```", "", json_string).strip()
            return json.loads(cleaned)
        except (json.JSONDecodeError, ValueError) as e:
            if getattr(self, "debug", False):
                print(f"Warning: Failed to parse JSON: {e}")
                print(f"Raw string: {json_string}")
            
            # Try to fix common JSON issues
            try:
                # Fix LaTeX escaping issues
                # The problem: JSON doesn't allow \(, \), \times, etc. as escape sequences
                # Solution: escape all backslashes that are not already part of valid JSON escapes
                fixed = cleaned
                
                # Strategy: iterate through the string and escape invalid escape sequences
                # Valid JSON escapes: \\, \", \/, \b, \f, \n, \r, \t, \uXXXX
                # Invalid ones like \(, \), \times, etc. need to be escaped
                
                # Find all backslashes and process them
                # We'll iterate through and fix invalid escapes
                # Special handling needed for \t and \n which might be part of LaTeX commands like \times, \neq
                result = []
                i = 0
                while i < len(fixed):
                    if fixed[i] == '\\':
                        # Check if it's a valid escape sequence
                        if i + 1 < len(fixed):
                            next_char = fixed[i + 1]
                            # Special case: \t might be part of \times, \text, \to, etc.
                            if next_char == 't':
                                # Check if it's \times (followed by "imes")
                                if i + 6 < len(fixed) and fixed[i+2:i+6] == 'imes':
                                    result.append('\\\\times')
                                    i += 6
                                # Check if it's \text (followed by "ext")
                                elif i + 6 < len(fixed) and fixed[i+2:i+6] == 'ext':
                                    result.append('\\\\text')
                                    i += 6
                                # Check if it's \to (followed by "o")
                                elif i + 3 < len(fixed) and fixed[i+2] == 'o':
                                    result.append('\\\\to')
                                    i += 3
                                else:
                                    # This is a real \t (tab), keep it
                                    result.append('\\t')
                                    i += 2
                            # Special case: \n might be part of \neq, \not, etc.
                            elif next_char == 'n':
                                # Check if it's \neq (followed by "eq")
                                if i + 4 < len(fixed) and fixed[i+2:i+4] == 'eq':
                                    result.append('\\\\neq')
                                    i += 4
                                # Check if it's \not (followed by "ot")
                                elif i + 4 < len(fixed) and fixed[i+2:i+4] == 'ot':
                                    result.append('\\\\not')
                                    i += 4
                                else:
                                    # This is a real \n (newline), keep it
                                    result.append('\\n')
                                    i += 2
                            # Special case: \b might be part of \boxed
                            elif next_char == 'b':
                                # Check if it's \boxed (followed by "oxed")
                                if i + 7 < len(fixed) and fixed[i+2:i+7] == 'oxed':
                                    result.append('\\\\boxed')
                                    i += 7
                                else:
                                    # This is a real \b (backspace), keep it
                                    result.append('\\b')
                                    i += 2
                            # Special case: \f might be part of \frac
                            elif next_char == 'f':
                                # Check if it's \frac (followed by "rac")
                                if i + 5 < len(fixed) and fixed[i+2:i+5] == 'rac':
                                    result.append('\\\\frac')
                                    i += 5
                                else:
                                    # This is a real \f (formfeed), keep it
                                    result.append('\\f')
                                    i += 2
                            # Special case: \c might be part of \cdot, \circ, etc.
                            elif next_char == 'c':
                                # Check if it's \cdot (followed by "dot")
                                if i + 6 < len(fixed) and fixed[i+2:i+6] == 'dot':
                                    result.append('\\\\cdot')
                                    i += 6
                                # Check if it's \circ (followed by "irc")
                                elif i + 6 < len(fixed) and fixed[i+2:i+6] == 'irc':
                                    result.append('\\\\circ')
                                    i += 6
                                else:
                                    # Not a valid JSON escape, escape it
                                    result.append('\\\\' + next_char)
                                    i += 2
                            # Other valid JSON escapes (single character)
                            elif next_char in ['"', '/', 'r']:
                                result.append(fixed[i:i+2])
                                i += 2
                            # Unicode escape sequence \uXXXX (4 hex digits)
                            elif next_char == 'u':
                                # Check if it's a valid \uXXXX sequence
                                if i + 5 < len(fixed) and all(c in '0123456789abcdefABCDEF' for c in fixed[i+2:i+6]):
                                    result.append(fixed[i:i+6])
                                    i += 6
                                else:
                                    # Invalid \u sequence, escape the backslash
                                    result.append('\\\\' + next_char)
                                    i += 2
                            # Double backslash
                            elif next_char == '\\':
                                result.append('\\\\')
                                i += 2
                            # Invalid escape (LaTeX like \(, \), \sqrt, etc.), escape it
                            else:
                                result.append('\\\\' + next_char)
                                i += 2
                        else:
                            # Lone backslash at end, escape it
                            result.append('\\\\')
                            i += 1
                    else:
                        result.append(fixed[i])
                        i += 1
                
                fixed = ''.join(result)
                try:
                    return json.loads(fixed)
                except json.JSONDecodeError as e2:
                    # If still fails, try to fix control characters and other issues
                    # Replace actual newlines in string values with \n
                    # This is a more aggressive fix for malformed JSON
                    try:
                        # Escape control characters in string values
                        # Escape control characters (except those already escaped)
                        # Replace \n (actual newline) with \\n in string values
                        # But be careful not to break valid JSON
                        fixed2 = re.sub(r'(?<!\\)\n', r'\\n', fixed)
                        fixed2 = re.sub(r'(?<!\\)\r', r'\\r', fixed2)
                        fixed2 = re.sub(r'(?<!\\)\t', r'\\t', fixed2)
                        return json.loads(fixed2)
                    except (json.JSONDecodeError, ValueError) as e3:
                        if getattr(self, "debug", False):
                            print(f"Warning: Even after fixing, JSON parsing failed: {e2}")
                            print(f"Second attempt also failed: {e3}")
                        # If still fails, return default
                        return default_value or {"debate_answer": "", "Reason": "JSON parsing failed"}
            except (json.JSONDecodeError, ValueError) as e2:
                if getattr(self, "debug", False):
                    print(f"Warning: Even after fixing, JSON parsing failed: {e2}")
                # If still fails, return default
                return default_value or {"debate_answer": "", "Reason": "JSON parsing failed"}

    def init_prompt(self, debate_topic):
        """initialize the prompt"""
        self.player_meta_prompt = PLAYER_META_PROMPT.replace("##debate_topic##", debate_topic)
        self.moderator_prompt = MODERATOR_META_PROMPT.replace("##debate_topic##", debate_topic)
        self.affirmative_prompt = AFFIRMATIVE_PROMPT.replace("##debate_topic##", debate_topic)
        self.judge_prompt_last2 = JUDGE_PROMPT_LAST2.replace("##debate_topic##", debate_topic)

    def create_agents(self):
        """create players and moderator"""
        if len(self.agent_profiles) < 3:
            raise ValueError("MAD requires at least three agent profiles: affirmative, negative, and moderator.")

        self.players = []
        for i, profile in enumerate(self.agent_profiles):
            agent = Agent(
                profile["name"],
                model_name=profile.get("model_name"),
                temperature=profile.get("temperature"),
            )
            agent.id = i
            self.players.append(agent)
            
        self.affirmative = self.players[0]
        self.negative = self.players[1]
        self.moderator = self.players[2]

    def init_agents(self):
        """initialize player_meta_prompt, and start the first round of debate"""
        self.affirmative.set_meta_prompt(self.player_meta_prompt)
        self.negative.set_meta_prompt(self.player_meta_prompt)
        self.moderator.set_meta_prompt(self.moderator_prompt)

        # An affirmative agent starts the debate
        self.affirmative.add_event(self.affirmative_prompt)
        # self.aff_ans = self.affirmative.ask()
        self.aff_ans = self._call_agent(agent=self.affirmative, messages=self.affirmative.memory_lst)
        self.affirmative.add_memory(self.aff_ans)
        self.base_answer = self.aff_ans  

        # A negative agent responds to the affirmative agent
        self.negative.add_event(NEGATIVE_PROMPT.replace('##aff_ans##', self.aff_ans))
        self.neg_ans = self._call_agent(agent=self.negative, messages=self.negative.memory_lst)
        self.negative.add_memory(self.neg_ans)

        # A moderator evaluates the answers from both sides
        self.moderator.add_event(
            MODERATOR_PROMPT.replace('##aff_ans##', self.aff_ans)
            .replace('##neg_ans##', self.neg_ans)
            .replace('##round##', 'first')
        )
        self.mod_ans = self._call_agent(agent=self.moderator, messages=self.moderator.memory_lst)
        self.moderator.add_memory(self.mod_ans)
        self.mod_ans = self.parse_json_string(self.mod_ans)

    def init_agents_multimodal(self, multimodal_content=None, per_agent_resolutions=None):
        """Initialize agents with multimodal support"""
        self.affirmative.set_meta_prompt(self.player_meta_prompt)
        self.negative.set_meta_prompt(self.player_meta_prompt)
        self.moderator.set_meta_prompt(self.moderator_prompt)

        res_aff = per_agent_resolutions[0] if per_agent_resolutions else None
        res_neg = per_agent_resolutions[1] if per_agent_resolutions else None
        res_mod = per_agent_resolutions[2] if per_agent_resolutions else None

        # An affirmative agent starts the debate
        self.affirmative.add_event(self.affirmative_prompt)
        if multimodal_content is not None:
            aff_messages = self.prepare_multimodal_messages(self.affirmative.memory_lst, multimodal_content, res_aff)
            self.aff_ans = self._call_agent(agent=self.affirmative, messages=aff_messages)
        else:
            self.aff_ans = self._call_agent(agent=self.affirmative, messages=self.affirmative.memory_lst)
        self.affirmative.add_memory(self.aff_ans)
        self.base_answer = self.aff_ans  

        # A negative agent responds to the affirmative agent
        self.negative.add_event(NEGATIVE_PROMPT.replace('##aff_ans##', self.aff_ans))
        if multimodal_content is not None:
            # Negative agent should see the same multimodal content (image + question)
            neg_messages = self.prepare_multimodal_messages(self.negative.memory_lst, multimodal_content, res_neg)
            self.neg_ans = self._call_agent(agent=self.negative, messages=neg_messages)
        else:
            self.neg_ans = self._call_agent(agent=self.negative, messages=self.negative.memory_lst)
        self.negative.add_memory(self.neg_ans)

        # A moderator evaluates the answers from both sides
        self.moderator.add_event(
            MODERATOR_PROMPT.replace('##aff_ans##', self.aff_ans)
            .replace('##neg_ans##', self.neg_ans)
            .replace('##round##', 'first')
        )
        moderator_messages = self.prepare_multimodal_messages(self.moderator.memory_lst, multimodal_content, res_mod)
        self.mod_ans = self._call_agent(agent=self.moderator, messages=moderator_messages)
        self.mod_ans = re.sub(r"```json|```", "", self.mod_ans).strip()
        self.moderator.add_memory(self.mod_ans)
        self.mod_ans = self.parse_json_string(self.mod_ans)

    def round_dct(self, num: int):
        dct = {
            1: 'first', 2: 'second', 3: 'third', 4: 'fourth', 5: 'fifth',
            6: 'sixth', 7: 'seventh', 8: 'eighth', 9: 'ninth', 10: 'tenth'
        }
        return dct.get(num, f"{num}th")

    def prepare_multimodal_messages(self, agent_memory, multimodal_content=None, resolution=None):
        """Attach image content to the latest user message for the given memory."""
        if multimodal_content is None:
            return agent_memory

        agent_messages = deepcopy(agent_memory)
        
        content_to_use = multimodal_content
        if resolution is not None and resolution != "None" and isinstance(content_to_use, list) and len(content_to_use) > 0:
            content_to_use = deepcopy(multimodal_content)
            if isinstance(content_to_use[0], dict) and "image" in content_to_use[0]:
                content_to_use[0]["resolution"] = resolution

        for idx in range(len(agent_messages) - 1, -1, -1):
            msg = agent_messages[idx]
            if msg["role"] != "user":
                continue

            msg_text = msg["content"] if isinstance(msg["content"], str) else ""
            msg["content"] = compose_multimodal_input(msg_text, content_to_use)
            break

        return agent_messages

    def print_answer(self, debate_topic):
        print("\n\n===== Debate Done! =====")
        print("\n----- Debate Topic -----")
        print(debate_topic)
        print("\n----- Base Answer -----")
        print(self.base_answer)
        print("\n----- Debate Answer -----")
        print(self.debate_answer)
        print("\n----- Debate Reason -----")
        print(self.config.get("Reason", "No reason provided."))

    def inference(self, sample):
        """inference function for MAD"""
        # Check if multimodal and prepare content
        debate_topic = sample["query"]
        is_multimodal = is_multimodal_dataset(sample)
        if is_multimodal:
            multimodal_content = prepare_multimodal_content(sample)
            if not isinstance(multimodal_content, list):
                raise ValueError("Multimodal sample missing valid image content.")
        else:
            multimodal_content = None
        
        self.init_prompt(debate_topic)
        self.create_agents()
        self.debate_answer = ""
        
        # Gated resolution logic for DivMAD
        gating_metadata = None
        per_agent_resolutions = None
        is_gated = False
        
        if is_multimodal:
            routing = self.method_config.get("routing_strategy")
            if routing == "gated_mix_res":
                # Standard probe resolution ladder for gauging confidence (not from agent configs)
                probe_ladder = self.method_config.get("probe_resolutions", ["0.5x", "1x", "2x"])
                if not isinstance(probe_ladder, list):
                    probe_ladder = parse_resolution_str(str(probe_ladder))
            
                num_agents = self.num_players
                initial_res = []
                for i in range(num_agents):
                    initial_res.append(probe_ladder[i % len(probe_ladder)])
                    
                probe_anlls = []
                probe_answers = []
                probe_res = []

                # Standard CoT prompt without debater/moderator persona for probing
                cot_probe_prompt = f"{debate_topic}\n\nSolve the problem step by step before giving your final answer."
                
                # Run the probe for all N agents
                for i in range(num_agents):
                    agent = self.players[i]
                    res = initial_res[i]
                    
                    # Remove persona for probe round
                    agent.set_meta_prompt("")
                    agent.memory_lst = []
                    
                    agent.add_event(cot_probe_prompt)
                    
                    agent_mm_content = deepcopy(multimodal_content)
                    if isinstance(agent_mm_content, list) and len(agent_mm_content) > 0:
                        if isinstance(agent_mm_content[0], dict) and "image" in agent_mm_content[0]:
                            agent_mm_content[0]["resolution"] = res
                            
                    probe_messages = self.prepare_multimodal_messages(agent.memory_lst, agent_mm_content)
                    
                    call_result = self._call_agent(
                        agent=agent, 
                        messages=probe_messages, 
                        temperature=0.0,
                        return_logprobs=True
                    )
                    ans, anll, _ = call_result

                    probe_answers.append(ans)
                    probe_res.append(res)
                    probe_anlls.append(anll if anll is not None else 999.0)

                    # Clear memory for actual debate
                    agent.memory_lst = []

                threshold = self.method_config.get("gated_anll_threshold", 0.2)
                adaptive = self.method_config.get("adaptive_threshold", False)
                decision = gated_resolution_decision(
                    round_anlls=probe_anlls,
                    agent_resolutions=probe_res,
                    agent_answers=probe_answers,
                    num_agents=num_agents,
                    threshold=threshold,
                    adaptive_threshold=adaptive,
                    debug=self.debug,
                )
                
                per_agent_resolutions = decision["new_resolutions"]
                gating_metadata = decision["metadata"]
                is_gated = True
                
            elif routing == "mix_res":
                mix_ladder = self.method_config.get("resolutions", ["2x", "1x", "0.5x"])
                if not isinstance(mix_ladder, list):
                    mix_ladder = parse_resolution_str(str(mix_ladder))
                per_agent_resolutions = [
                    mix_ladder[i % len(mix_ladder)] for i in range(self.num_players)
                ]
            elif routing == "consensus_scaling":
                mix_ladder = self.method_config.get("resolutions", ["2x", "1x", "0.5x"])
                if not isinstance(mix_ladder, list):
                    mix_ladder = parse_resolution_str(str(mix_ladder))
                per_agent_resolutions = [mix_ladder[0] for _ in range(self.num_players)]

        # --- Generalized Perceiver Step ---
        if is_multimodal and self.method_config.get("use_perceiver", False):
            consensed_res = None
            if routing == "mix_res" and is_gated and decision and decision.get("initial_res_list"):
                consensed_res = decision["initial_res_list"][0]
            elif per_agent_resolutions:
                def get_res_val(r):
                    try:
                        return float(str(r).strip('x'))
                    except Exception:
                        return 0.0
                consensed_res = max(per_agent_resolutions, key=get_res_val)

            description = self._perceive_image(debate_topic, multimodal_content, consensed_res)
            debate_topic = f"{debate_topic}\n\nVisual Description of the image: {description}"
            
            # Re-initialize prompts with updated text-only topic
            self.init_prompt(debate_topic)
            
            is_multimodal = False
            multimodal_content = None

        if is_multimodal:
            self.init_agents_multimodal(multimodal_content, per_agent_resolutions)
        else:
            self.init_agents()
            
        res_aff = per_agent_resolutions[0] if per_agent_resolutions else None
        res_neg = per_agent_resolutions[1] if per_agent_resolutions else None
        res_mod = per_agent_resolutions[2] if per_agent_resolutions else None

        for round in range(self.max_round - 1):
            if is_multimodal and self.method_config.get("routing_strategy") == "consensus_scaling" and not is_gated:
                mix_ladder = self.method_config.get("resolutions", ["2x", "1x", "0.5x"])
                if not isinstance(mix_ladder, list):
                    mix_ladder = parse_resolution_str(str(mix_ladder))
                
                for j in range(self.num_players):
                    current_res = per_agent_resolutions[j]
                    if current_res in mix_ladder:
                        current_level = mix_ladder.index(current_res)
                    else:
                        current_level = -1
                    per_agent_resolutions[j] = mix_ladder[(current_level + 1) % len(mix_ladder)]
                    
                res_aff = per_agent_resolutions[0]
                res_neg = per_agent_resolutions[1]
                res_mod = per_agent_resolutions[2]

            # Check if moderator provided a valid answer
            if self.is_valid_answer(self.mod_ans.get("debate_answer", "")):
                break
            else:
                # set the prompt for the affirmative side and update memory
                self.affirmative.add_event(DEBATE_PROMPT.replace('##oppo_ans##', self.neg_ans))
                if multimodal_content is not None:
                    aff_messages = self.prepare_multimodal_messages(self.affirmative.memory_lst, multimodal_content, res_aff)
                    self.aff_ans = self._call_agent(agent=self.affirmative, messages=aff_messages)
                else:
                    self.aff_ans = self._call_agent(agent=self.affirmative, messages=self.affirmative.memory_lst)
                self.affirmative.add_memory(self.aff_ans)

                # set the prompt for the negative side and update memory
                self.negative.add_event(DEBATE_PROMPT.replace('##oppo_ans##', self.aff_ans))
                if multimodal_content is not None:
                    neg_messages = self.prepare_multimodal_messages(self.negative.memory_lst, multimodal_content, res_neg)
                    self.neg_ans = self._call_agent(agent=self.negative, messages=neg_messages)
                else:
                    self.neg_ans = self._call_agent(agent=self.negative, messages=self.negative.memory_lst)
                self.negative.add_memory(self.neg_ans)

                # set the prompt for the moderator and update memory
                self.moderator.add_event(
                    MODERATOR_PROMPT.replace('##aff_ans##', self.aff_ans)
                    .replace('##neg_ans##', self.neg_ans)
                    .replace('##round##', self.round_dct(round+2))
                )
                moderator_messages = self.prepare_multimodal_messages(self.moderator.memory_lst, multimodal_content, res_mod)
                self.mod_ans = self._call_agent(agent=self.moderator, messages=moderator_messages)
                self.moderator.add_memory(self.mod_ans)
                self.mod_ans = self.parse_json_string(self.mod_ans)

        # Check if moderator provided a valid answer
        debate_answer = self.mod_ans.get("debate_answer", "")
        

        if self.is_valid_answer(debate_answer):
            self.debate_answer = debate_answer
            self.config.update(self.mod_ans)
            self.config['success'] = True
        else:
            # let the judge decide the debate
            judge_player = Agent(name='Judge', model_name=self.judge_model_name, temperature=self.judge_temperature)
            
            # Use final round answers (memory_lst[-1]) instead of hardcoded round 1 (memory_lst[2])
            aff_ans = self.affirmative.memory_lst[-1]['content']
            neg_ans = self.negative.memory_lst[-1]['content']

            # set the prompt for the judge and update memory
            judge_player.set_meta_prompt(self.moderator_prompt)
            judge_player.add_event(JUDGE_PROMPT_LAST1.replace('##aff_ans##', aff_ans).replace('##neg_ans##', neg_ans))
            judge_messages = self.prepare_multimodal_messages(judge_player.memory_lst, multimodal_content, res_mod)
            ans = self._call_agent(agent=judge_player, messages=judge_messages)
            judge_player.add_memory(ans)

            # let the judge decide the debate and give the final answer
            judge_player.add_event(self.judge_prompt_last2)
            judge_messages = self.prepare_multimodal_messages(judge_player.memory_lst, multimodal_content, res_mod)
            ans = self._call_agent(agent=judge_player, messages=judge_messages)
            judge_player.add_memory(ans)

            ans = self.parse_json_string(ans)
            if ans and self.is_valid_answer(ans.get("debate_answer", "")):
                self.debate_answer = ans["debate_answer"]
                self.config['success'] = True

            self.config.update(ans)
            self.players.append(judge_player)

        # Collect the complete debate process with debug metadata
        debate_process = {
            "affirmative_history": self.affirmative.memory_lst,
            "negative_history": self.negative.memory_lst,
            "moderator_history": self.moderator.memory_lst,
            "base_answer": self.base_answer,
            "final_answer": self.debate_answer,
            "debate_config": self.config,
            "resolutions": per_agent_resolutions if is_multimodal else None,
            "debate_answers": {
                "affirmative": self.aff_ans,
                "negative": self.neg_ans,
                "moderator": self.debate_answer
            }
        }
        
        if gating_metadata:
            debate_process["gating_metadata"] = gating_metadata
            self.config.update(gating_metadata)
        
        return {
            "response": self.debate_answer,
            "debate_process": debate_process
        }
