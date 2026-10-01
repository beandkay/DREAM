"""Shared gating utility for resolution selection.

Provides the core decision logic used by llm_debate, DMAD, and DivMAD
to probe agents at different resolutions and select optimal ones
before the real debate begins.
"""

import re
from typing import List, Dict, Any


def extract_answer(text: str) -> str:
    """Extract final answer using multiple patterns.
    
    Tries in order: \\boxed{} → (A)/(B) multiple-choice → "the answer is X" → last line.
    """
    # Try \boxed{} first
    boxed = re.findall(r'\\boxed{(.*?)}', text)
    if boxed:
        return boxed[-1].strip()
    # Try multiple-choice pattern: (A), (B), etc.
    mc = re.findall(r'\(([A-Z])\)', text)
    if mc:
        return mc[-1].strip()
    # Try "Final Answer: X" pattern
    ans = re.findall(r'(?:final answer:|the answer is|answer:)\s*([A-Z0-9.\-]+)', text, re.IGNORECASE)
    if ans:
        return ans[-1].strip().upper()
    # Fallback: last non-empty line
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    return lines[-1] if lines else text.strip()


def parse_resolution_str(res_str: str) -> List[str]:
    """Parse a resolution string like \"['1x', '2x']\" into a list, or wrap a scalar."""
    import ast
    if res_str.startswith("['") and res_str.endswith("']"):
        try:
            return ast.literal_eval(res_str)
        except Exception:
            return [res_str]
    return [res_str]


def sort_resolutions_desc(res_list: List[str]) -> List[str]:
    """Sort resolutions in descending order (e.g. '2x', '1x', '0.5x')."""
    def get_val(r):
        try:
            return float(str(r).strip('x'))
        except ValueError:
            return 0.0
    return sorted(res_list, key=get_val, reverse=True)


def gated_resolution_decision(
    round_anlls: List[float],
    agent_resolutions: List[str],
    agent_answers: List[str],
    num_agents: int,
    threshold: float = 0.15,
    adaptive_threshold: bool = False,
    debug: bool = False,
) -> Dict[str, Any]:
    """Core gating decision logic.

    Given probe-round ANLL scores, current resolutions, and extracted answers,
    decide the optimal resolution routing strategy.

    Args:
        round_anlls: ANLL score per agent from the probe round.
        agent_resolutions: Current resolution string per agent (e.g. '0.5x').
        agent_answers: Raw answer text per agent (for conflict detection).
        num_agents: Total number of agents.
        threshold: Static ANLL confidence threshold (used when adaptive_threshold=False,
            or as a fallback when adaptive_threshold=True but fewer than 2 valid probe
            ANLLs are available to estimate mu/sigma).
        adaptive_threshold: If True, compute tau = mu_ANLL - sigma_ANLL over this
            sample's probe-round ANLLs (mean and std dev across the N probe agents)
            instead of using the static value. Agents with ANLL(i) <= tau are
            "low-uncertainty" (confident). This calibrates to sample difficulty: easy
            samples yield uniformly low ANLL (tau close to the low mean), hard samples
            yield dispersed ANLL (tau pulled down by the spread). When sigma is 0 (all
            probe ANLLs identical), tau equals the mean, so every agent qualifies.
        debug: Whether to print debug info.

    Returns:
        Dict with keys:
            routing_strategy: 'consensus_scaling' or 'mix_res'
            gated_strategy: e.g. 'static_all', 'best_anll', 'subset_static_mixed',
                           'conflict_mix_res', or None (for mix_res)
            new_resolutions: Updated resolution list per agent.
            metadata: Dict of gating metadata for logging.
    """
    # Dynamically extract all unique initial resolutions
    initial_res_list = list(dict.fromkeys(agent_resolutions))
    initial_res_list = sort_resolutions_desc(initial_res_list)
    initial_mix_str = str(initial_res_list)

    # Extract answers for consensus checking and logging
    extracted_answers = [extract_answer(ans) for ans in agent_answers]

    # Sample-adaptive threshold: tau = mu_ANLL - sigma_ANLL over this sample's
    # probe-round ANLL distribution (excluding invalid/missing sentinel scores).
    mean_anll = None
    std_anll = None
    effective_threshold = threshold
    if adaptive_threshold:
        valid_anlls = [a for a in round_anlls if a is not None and a < 900]
        if len(valid_anlls) > 1:
            mean_anll = sum(valid_anlls) / len(valid_anlls)
            variance = sum((a - mean_anll) ** 2 for a in valid_anlls) / len(valid_anlls)
            std_anll = variance ** 0.5
            effective_threshold = mean_anll - std_anll
        # else: not enough valid probe ANLLs to estimate mu/sigma; keep the
        # static threshold as a fallback.

    confident_indices = [idx for idx, anll in enumerate(round_anlls) if anll is not None and anll <= effective_threshold]
    num_confident = len(confident_indices)

    metadata = {
        "is_gated": True,
        "gated_threshold": effective_threshold,
        "gated_static_threshold": threshold,
        "gated_adaptive": adaptive_threshold,
        "gated_probe_mean_anll": mean_anll,
        "gated_probe_std_anll": std_anll,
        "gated_round1_anlls": round_anlls,
        "gated_probe_answers": extracted_answers,
        "gated_probe_resolutions": list(agent_resolutions),
    }
    
    new_resolutions = list(agent_resolutions)  # copy
    routing_strategy = None
    gated_strategy = None
    
    if debug:
        print(f"\n>> [Gated Probe Results]")
        for idx, anll in enumerate(round_anlls):
            print(f"   Agent {idx + 1} ({agent_resolutions[idx]}): ANLL={anll:.4f}, Answer={extracted_answers[idx]}")
        thr_label = (f"{effective_threshold:.4f} (adaptive: mu={mean_anll:.4f}, sigma={std_anll:.4f})"
                     if adaptive_threshold and mean_anll is not None
                     else f"{effective_threshold:.4f} (static)")
        print(f">> Threshold: {thr_label}")
        print(f">> Confident: {num_confident}/{num_agents}")
    
    if num_confident >= 2:
        confident_answers = [extract_answer(agent_answers[idx]) for idx in confident_indices]
        
        if len(set(confident_answers)) > 1:
            # Conflict: confident agents disagree
            from collections import Counter
            counts = Counter(confident_answers)
            most_common_ans, most_common_count = counts.most_common(1)[0]
            
            # If there is a strict majority (at least 2 agents agree)
            if most_common_count >= 2:
                filtered_indices = [idx for idx in confident_indices if extract_answer(agent_answers[idx]) == most_common_ans]
                confident_res_raw = [agent_resolutions[idx] for idx in filtered_indices]
                if debug:
                    print(f">> Conflict detected, but found majority answer '{most_common_ans}'. Filtering out dissenting resolutions.")
            else:
                confident_res_raw = [agent_resolutions[idx] for idx in confident_indices]

            confident_res = list(dict.fromkeys(confident_res_raw))
            confident_res = sort_resolutions_desc(confident_res)
            
            # If after filtering we only have 1 resolution, we don't need to mix
            if len(confident_res) == 1:
                if debug:
                    print(f">> ROUTING TO: consensus_scaling (Conflict resolved to single majority resolution: {confident_res[0]})")
                routing_strategy = "consensus_scaling"
                gated_strategy = "best_anll"
                new_resolutions = [confident_res[0]] * num_agents
            else:
                subset_mix_str = str(confident_res)
                if debug:
                    print(f">> ROUTING TO: consensus_scaling (Conflict! Scaling to {subset_mix_str})")
                
                routing_strategy = "consensus_scaling"
                gated_strategy = "conflict_mix_res"
                new_resolutions = [subset_mix_str] * num_agents
            
        else:
            # Confident agents agree
            if num_confident == num_agents:
                if debug:
                    print(">> ROUTING TO: consensus_scaling (Static All - No Rotation)")
                routing_strategy = "consensus_scaling"
                gated_strategy = "static_all"
                # Keep existing resolutions
                
            else:
                confident_res_raw = [agent_resolutions[idx] for idx in confident_indices]
                confident_res = list(dict.fromkeys(confident_res_raw))
                confident_res = sort_resolutions_desc(confident_res)
                
                if len(confident_res) == 1:
                    if debug:
                        print(f">> ROUTING TO: consensus_scaling (Lock-in to best resolution: {confident_res[0]})")
                    routing_strategy = "consensus_scaling"
                    gated_strategy = "best_anll"
                    new_resolutions = [confident_res[0]] * num_agents
                else:
                    if debug:
                        print(">> ROUTING TO: consensus_scaling (Static Subset with Mix Res for unconfident)")
                    routing_strategy = "consensus_scaling"
                    gated_strategy = "subset_static_mixed"
                    subset_mix_str = str(confident_res)
                    for j in range(num_agents):
                        if j not in confident_indices:
                            new_resolutions[j] = subset_mix_str
                            
    elif num_confident == 1:
        best_agent_idx = confident_indices[0]
        best_res = agent_resolutions[best_agent_idx]
        if debug:
            print(f">> ROUTING TO: consensus_scaling (Lock-in to best resolution: {best_res})")
        
        routing_strategy = "consensus_scaling"
        gated_strategy = "best_anll"
        new_resolutions = [best_res] * num_agents
        
    else:
        if debug:
            print(">> ROUTING TO: mix_res")
        routing_strategy = "mix_res"
        gated_strategy = None
        new_resolutions = [initial_mix_str] * num_agents
    
    metadata["gated_routing_decision"] = routing_strategy
    if gated_strategy:
        metadata["gated_consensus_strategy"] = gated_strategy
    
    return {
        "routing_strategy": routing_strategy,
        "gated_strategy": gated_strategy,
        "new_resolutions": new_resolutions,
        "initial_res_list": initial_res_list,
        "metadata": metadata,
    }
