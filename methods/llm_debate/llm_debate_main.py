# The official implementation of LLM Debate https://github.com/composable-models/llm_multiagent_debate offen encounters errors.
# This is a modified version of the original implementation.

import os
import re
from copy import deepcopy
from ..mad_base import MAD
from utils.utils import (
    is_multimodal_dataset,
    prepare_multimodal_content,
    compose_multimodal_input,
    transform_image_payload,
    prepare_qwen_augmented_content
)
from ..gating_utils import gated_resolution_decision, parse_resolution_str

class LLM_Debate_Main(MAD):
    _message_announced = False

    def __init__(self, general_config, method_config_name=None):
        
        # If no config is passed, default to config_main
        if method_config_name is None:
            method_config_name = "config_main"
            
        super().__init__(general_config, method_config_name)

        self.agents_num = self.method_config["agents_num"]
        self.rounds_num = self.method_config["rounds_num"]
        self.judge_model_name = self.method_config.get("judge_model_name")
        self.judge_temperature = self.method_config.get("judge_temperature")
        self.dataset_name = general_config.get("test_dataset_name", "")
        self.agent_profiles = self._build_agent_profiles()
        self._maybe_announce_agent_models()

    def _build_agent_profiles(self):
        """Load optional per-agent settings (model_name, temperature, system prompt)."""
        configured_agents = self.method_config.get("agents") or []
        profiles = []

        if configured_agents:
            if len(configured_agents) != self.agents_num:
                raise ValueError(
                    f"agents_num ({self.agents_num}) does not match number of entries in 'agents' ({len(configured_agents)})"
                )
            source = configured_agents
        else:
            source = [{} for _ in range(self.agents_num)]

        for idx, entry in enumerate(source):
            entry = entry or {}
            profiles.append(
                {
                    "name": entry.get("name") or f"Agent {idx + 1}",
                    "model_name": entry.get("model_name"),
                    "temperature": entry.get("temperature"),
                    "resolutions": entry.get("resolutions") or entry.get("resolution"),
                    "flips": entry.get("flips") or entry.get("flip"),
                    "shifts": entry.get("shifts") or entry.get("shift"),
                    "crops": entry.get("crops") or entry.get("crop"),
                    "blurs": entry.get("blurs") or entry.get("blur"),
                    "grayscale": entry.get("grayscale"),
                    "mix_all": entry.get("mix_all", False)
                }
            )

        return profiles

    def _maybe_announce_agent_models(self):
        """Print a one-time notice when agents use custom models."""
        profile_rows = []
        for profile in self.agent_profiles:
            model_name = profile.get("model_name") or self.model_name
            profile_rows.append(f"{profile['name']} -> {model_name}")

        is_heterogeneous = any(profile.get("model_name") for profile in self.agent_profiles)
        if is_heterogeneous and not LLM_Debate_Main._message_announced:
            mapping = "; ".join(profile_rows)
            print(f"[LLM-Debate] Heterogeneous agents enabled. Model mapping: {mapping}")
            LLM_Debate_Main._message_announced = True
    
    def inference(self, sample):
        # Check if multimodal and prepare content
        is_multimodal = is_multimodal_dataset(sample)
        if is_multimodal:
            multimodal_content = prepare_multimodal_content(sample)
            query = sample["query"]
        else:
            multimodal_content = None
            query = sample["query"]

        base_prompt = f"""{query} Make sure to state your answer at the end of the response."""
        agent_contexts = []

        for _ in self.agent_profiles:
            agent_contexts.append([{"role": "user", "content": base_prompt}])

        # --- Resolution routing setup ---
        routing_strategy = self.method_config.get("routing_strategy", "conventional")
        if routing_strategy in ("gated_mix_res", "consensus_scaling"):
            resolutions = self.method_config.get("resolutions", self.method_config.get("resolution_ladder", ["2x", "1x", "0.5x"]))
        else:
            resolutions = [None]

        # Populate agent_resolutions from profiles
        agent_resolutions = [str(profile.get("resolutions", "None")) for profile in self.agent_profiles]

        # For consensus_scaling, initialize all agents at the first resolution in the ladder
        if routing_strategy == "consensus_scaling" and is_multimodal:
            for j in range(self.agents_num):
                agent_resolutions[j] = resolutions[0]

        # Pre-build per-agent multimodal content
        per_agent_multimodal = []
        if is_multimodal and multimodal_content is not None:
            for idx, profile in enumerate(self.agent_profiles):
                if profile.get("mix_all"):
                    agent_mc = prepare_qwen_augmented_content(sample)
                elif routing_strategy == "consensus_scaling":
                    agent_mc = transform_image_payload(
                        multimodal_content,
                        resolutions=agent_resolutions[idx],
                    )
                else:
                    agent_mc = transform_image_payload(
                        multimodal_content,
                        resolutions=profile.get("resolutions"),
                        flips=profile.get("flips"),
                        shifts=profile.get("shifts"),
                        crops=profile.get("crops"),
                        blurs=profile.get("blurs"),
                        grayscale=profile.get("grayscale"),
                    )
                per_agent_multimodal.append(agent_mc)

        check_uncertainty = True

        # Per-agent answer history 
        agent_answer_history = [[] for _ in range(self.agents_num)]

        # --- Gated resolution probe ---
        is_gated = routing_strategy == "gated_mix_res" and is_multimodal and multimodal_content is not None
        gated_metadata = {}
        probe_anlls = None

        if is_gated:
            threshold = self.method_config.get("gated_anll_threshold", 0.2)

            # Assign each agent a different resolution from the ladder for the probe
            for idx, profile in enumerate(self.agent_profiles):
                profile["resolutions"] = resolutions[idx % len(resolutions)]
            agent_resolutions = [str(profile.get("resolutions", "1x")) for profile in self.agent_profiles]

            # Rebuild per-agent multimodal with probe resolutions
            per_agent_multimodal = []
            for profile in self.agent_profiles:
                per_agent_multimodal.append(
                    transform_image_payload(multimodal_content, resolutions=profile.get("resolutions"))
                )

            # Run probe round
            probe_anlls = []
            probe_answers = []
            for i, (agent_context, profile) in enumerate(zip(agent_contexts, self.agent_profiles)):
                model_name_override = profile.get("model_name")
                temperature_override = profile.get("temperature")

                messages = self.prepare_multimodal_messages(agent_context, per_agent_multimodal[i])
                # import pdb; pdb.set_trace()
                call_result = self.call_llm(
                    messages=messages,
                    model_name=model_name_override,
                    temperature=0.0,  # Force greedy decoding for deterministic probing
                    return_logprobs=True,
                    seed=(self.seed * 10 + i + 1 if getattr(self, "seed", None) is not None else i + 1)
                )
                response, anll, _ = call_result
                probe_anlls.append(anll if anll is not None else 999.0)
                probe_answers.append(response)

            # Call shared gating decision
            adaptive = self.method_config.get("adaptive_threshold", False)
            decision = gated_resolution_decision(
                round_anlls=probe_anlls,
                agent_resolutions=agent_resolutions,
                agent_answers=probe_answers,
                num_agents=self.agents_num,
                threshold=threshold,
                adaptive_threshold=adaptive,
                debug=self.debug,
            )

            routing_strategy = decision["routing_strategy"]
            gated_metadata.update(decision["metadata"])

            if self.debug:
                print(">> Discarding probe round. Starting debate with optimal resolutions.")

            # Apply decided resolutions
            for j in range(self.agents_num):
                if routing_strategy == "mix_res":
                    new_res = decision["initial_res_list"]
                else:
                    new_res = parse_resolution_str(decision["new_resolutions"][j])

                self.agent_profiles[j]["resolutions"] = new_res
                agent_resolutions[j] = decision["new_resolutions"][j]
                per_agent_multimodal[j] = transform_image_payload(
                    multimodal_content, resolutions=new_res
                )

            # After gating, use consensus_scaling for the debate (no further rotation)
            routing_strategy = "consensus_scaling"

        # --- Generalized Perceiver Step ---
        if is_multimodal and self.method_config.get("use_perceiver", False):
            # Select consensed resolution (highest resolution from decisions, or first)
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

            # Generate description
            description = self._perceive_image(query, multimodal_content, consensed_res)
            
            # Rewrite query and base prompt
            query = f"{query}\n\nVisual Description of the image: {description}"
            base_prompt = f"""{query} Using the visual description, please reason step-by-step to answer the question. Make sure to state your answer at the end of the response."""
            
            # Update agent contexts
            for ctx in agent_contexts:
                ctx[0]["content"] = base_prompt
                
            # Disable multimodal for the remainder of the debate
            is_multimodal = False
            multimodal_content = None

        # --- Main debate loop ---
        for round in range(self.rounds_num):
            round_anlls = []
            for i, (agent_context, profile) in enumerate(zip(agent_contexts, self.agent_profiles)):
                if round != 0:
                    agent_contexts_other = agent_contexts[:i] + agent_contexts[i+1:]
                    agent_resolutions_other = agent_resolutions[:i] + agent_resolutions[i+1:]
                    message = self.construct_message(agent_contexts_other, query, 2*round - 1, is_multimodal, multimodal_content, agent_resolutions_other)
                    agent_context.append(message)
                
                model_name_override = profile.get("model_name")
                temperature_override = profile.get("temperature")

                if is_multimodal and multimodal_content is not None:
                    active_multimodal = per_agent_multimodal[i]
                    messages = self.prepare_multimodal_messages(agent_context, active_multimodal)
                    call_result = self.call_llm(
                        messages=messages,
                        model_name=model_name_override,
                        temperature=temperature_override,
                        return_logprobs=check_uncertainty,
                        seed=(self.seed * 10 + i + 1 if self.seed is not None else i + 1)
                    )
                else:
                    # Text-only case
                    call_result = self.call_llm(
                        messages=agent_context,
                        model_name=model_name_override,
                        temperature=temperature_override,
                        return_logprobs=check_uncertainty,
                        seed=(self.seed * 10 + i + 1 if self.seed is not None else i + 1)
                    )
                
                if check_uncertainty:
                    response, anll, _ = call_result
                    if anll is not None:
                        round_anlls.append(anll)
                        if self.debug:
                            print(f"\n>> [Round {round + 1}] Agent {i + 1} Uncertainty Score (ANLL): {anll:.4f}\n")
                else:
                    response = call_result
                
                agent_context.append({"role": "assistant", "content": response})
                
            # Save history
            for j in range(self.agents_num):
                entry = {
                    "round": round + 1,
                    "resolution": agent_resolutions[j] if agent_resolutions else "1x",
                    "answer": agent_contexts[j][-1]['content'],
                }
                if j < len(round_anlls):
                    entry["anll"] = round_anlls[j]
                agent_answer_history[j].append(entry)

            # --- Consensus scaling: round-robin rotation (non-gated only) ---
            if routing_strategy == "consensus_scaling" and not is_gated and round < self.rounds_num - 1:
                if is_multimodal:
                    for j in range(self.agents_num):
                        current_res = agent_resolutions[j]
                        if current_res in resolutions:
                            current_level = resolutions.index(current_res)
                        else:
                            current_level = -1
                        agent_resolutions[j] = resolutions[(current_level + 1) % len(resolutions)]
                        # Rebuild image payload so the actual resolution changes
                        per_agent_multimodal[j] = transform_image_payload(
                            multimodal_content, resolutions=agent_resolutions[j]
                        )

        final_answer = self.aggregate_with_history(query, agent_answer_history, multimodal_content if is_multimodal else None, probe_anlls=probe_anlls)
        output = {
            "response": final_answer,
            "debate_process": agent_contexts
        }
        
        output["agent_answer_history"] = agent_answer_history
        output.update(gated_metadata)
            
        return output
    
    def construct_message(self, agents, question, idx, is_multimodal=False, multimodal_content=None, agent_resolutions=None):

        # Use introspection in the case in which there are no other agents.
        if len(agents) == 0:
            return {"role": "user", "content": "Can you verify that your answer is correct. Please reiterate your answer, making sure to state your answer at the end of the response."}

        prefix_string = "These are the recent/updated opinions from other agents: "

        if agent_resolutions is not None and len(agents) == len(agent_resolutions):
            def get_res_val(res_str):
                import ast
                res_str = str(res_str)
                if res_str.startswith("['") and res_str.endswith("']"):
                    try:
                        res_list = ast.literal_eval(res_str)
                        return float(str(res_list[0]).strip('x')) if res_list else 0.0
                    except Exception:
                        pass
                try:
                    return float(res_str.strip('x'))
                except Exception:
                    return 0.0

            agent_items = list(zip(agents, agent_resolutions))
            agent_items.sort(key=lambda item: get_res_val(item[1]), reverse=True)
            agents = [item[0] for item in agent_items]

        for agent in agents:
            agent_response = agent[idx]["content"]
            response = "\n\n One agent response: ```{}```".format(agent_response)

            prefix_string = prefix_string + response

        prefix_string = prefix_string + "\n\n Use these opinions carefully as additional references. Update your answer only if necessary. Make sure to state your answer at the end of the response. \n The original problem is {}.".format(question)
        
        return {"role": "user", "content": prefix_string}

    def aggregate_with_history(self, query, agent_answer_history, multimodal_content=None, probe_anlls=None):
        """History-aware aggregation with probe-calibrated rescue.

        For each agent, prefer the final-round (debate-refined) answer a(R)_i.
        If the gated resolution probe ran for this sample, the rescue threshold
        is calibrated to that probe round's cross-agent ANLL spread:
        delta = alpha * sigma(0)_ANLL, with alpha from ``anll_rescue_alpha``
        (default 0.5). An agent is rescued - its final-round answer replaced
        with a(r*_i)_i, the response from its lowest-ANLL prior round - when
        final_anll - min_prior_anll > delta. Whenever the rescue triggers,
        r*_i is necessarily an earlier round than R, since only prior rounds
        are searched.

        Without a probe (e.g. text-only runs, where sigma(0)_ANLL has no
        source), falls back to the ratio+floor heuristic controlled by
        ``anll_fallback_threshold`` / ``anll_fallback_min_anll``.

        Anti-recency bias pre-aggregation: rescued answers are surfaced as
        structured headers at the beginning of the judge's input, and the
        judge is directed to weigh them equally with final-round answers, so
        an earlier, lower-uncertainty answer isn't discounted merely for
        appearing before the final, potentially eroded round.

        Set ``enable_answer_rescue: false`` in the method config to disable
        this entire mechanism (ablation) - every agent's final-round answer
        is used as-is, with no rescue and no anti-recency headers.
        """
        enable_rescue = self.method_config.get("enable_answer_rescue", True)
        alpha = self.method_config.get("anll_rescue_alpha", 0.5)
        anll_threshold = self.method_config.get("anll_fallback_threshold", 1.5)
        anll_fallback_min_anll = self.method_config.get("anll_fallback_min_anll", 0.15)

        # delta = alpha * sigma(0)_ANLL, sigma(0)_ANLL = std dev of the probe
        # round's per-agent ANLL. None when no probe ran for this sample.
        delta = None
        if enable_rescue and probe_anlls:
            # 999.0 is the sentinel the probe uses for a missing ANLL (see inference()); exclude it too.
            valid_probe = [a for a in probe_anlls if a is not None and a < 900]
            if len(valid_probe) > 1:
                mean_probe = sum(valid_probe) / len(valid_probe)
                variance = sum((a - mean_probe) ** 2 for a in valid_probe) / len(valid_probe)
                sigma0 = variance ** 0.5
                delta = alpha * sigma0

        # --- Select the best answer per agent using the rescue rule ---
        selected_answers = []
        for i, history in enumerate(agent_answer_history):
            final_entry = history[-1]
            final_anll = final_entry.get('anll')

            # Find the lowest-ANLL entry among rounds strictly before the
            # final one, so a rescue can only point to an earlier round.
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

            if rescued:
                chosen_entry = best_prior_entry
                selection_reason = "rescued (probe-calibrated)" if delta is not None else "rescued (fallback)"
            else:
                chosen_entry = final_entry
                selection_reason = "final-round (debate-refined)"

            chosen_anll = chosen_entry.get('anll')
            selected_answers.append({
                "agent_id": i + 1,
                "answer": chosen_entry['answer'],
                "anll": chosen_anll if chosen_anll is not None else float('inf'),
                "resolution": chosen_entry.get('resolution', 'unknown'),
                "reason": selection_reason,
                "rescued": rescued,
                "rescue_round": chosen_entry.get('round') if rescued else None,
            })

        # --- Anti-recency bias pre-aggregation ---
        # Rescued answers are injected as structured headers before the
        # chronological "Agent N said ..." list, so the judge sees them
        # up front rather than buried behind the (possibly corrupted)
        # final round.
        rescued_headers = [
            f"[Agent {sa['agent_id']} Rescued Answer (Round {sa['rescue_round']}, uncertainty: low): "
            f"\"{sa['answer']}\"]"
            for sa in selected_answers if sa["rescued"]
        ]

        aggregate_instruction = ""
        if rescued_headers:
            aggregate_instruction += "\n".join(rescued_headers) + "\n\n"

        aggregate_instruction += f"Task:\n{query}\n\n"
        aggregate_instruction += (
            "Below are the selected answers from different agents.\n\n"
        )

        for sa in selected_answers:
            aggregate_instruction += f"Agent {sa['agent_id']} said \"{sa['answer']}\"\n\n"

        aggregate_instruction += (
            "Given all the above solutions, select the most correct one and state your final answer."
        )
        if rescued_headers:
            aggregate_instruction += (
                " Some agents' answers above were rescued from an earlier, lower-uncertainty "
                "round because their final-round confidence degraded significantly; weigh "
                "rescued answers equally with final-round answers when determining the "
                "majority consensus."
            )
        judge_temp = self.judge_temperature if self.judge_temperature is not None else 0.5
        if multimodal_content is not None:
            combined = compose_multimodal_input(aggregate_instruction, multimodal_content)
            response = self.call_llm(
                messages=[{"role": "user", "content": combined}],
                model_name=self.judge_model_name,
                temperature=judge_temp,
            )
        else:
            response = self.call_llm(
                prompt=aggregate_instruction,
                model_name=self.judge_model_name,
                temperature=judge_temp,
            )
        return response

    def prepare_multimodal_messages(self, agent_context, multimodal_content=None):
        if multimodal_content is None:
            return agent_context

        messages = deepcopy(agent_context)

        for idx in range(len(messages) - 1, -1, -1):
            msg = messages[idx]
            if msg["role"] != "user":
                continue

            msg_text = msg["content"] if isinstance(msg["content"], str) else ""
            msg["content"] = compose_multimodal_input(msg_text, multimodal_content)
            break

        return messages
