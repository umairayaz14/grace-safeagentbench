"""NeuralGrounding — grace.SymbolicGrounding where an LLM judges the premises.

The LLM sees ONLY the rendered state and the theory's premise catalog (no
ontology hints, per supervisor). For each premise it decides whether it holds
in the current state and how it instantiates. Judgments are sieved through the
theory lookup exactly like the hand-coded grounding, so the LLM can only
select and instantiate existing reasons, never invent them.

Prompt factors (set via NeuralGrounding.configure before build_sr):
  mode:   "batch" (all premises in one call) | "per_premise" (one call each)
  render: "raw" (metadata JSON)              | "nl" (natural-language state)
  bias:   "neutral"                          | "conservative" (when unsure, say holds)

Because the SDK constructs the grounding class itself (cls(theory)), the LLM
and factors are class-level configuration. CAPTURE[role] records the last
call's judgments for logging/scoring.
"""

from pydantic import BaseModel, Field

from grace.symbolic_reasoning import SymbolicGrounding
from grace.symbolic_reasoning.horty import HortyTheory, NLGroundedReason as GroundedReason

from sr_state_render import render_nl, render_raw

CAPTURE: dict = {}


class PremiseJudgment(BaseModel):
    name: str = Field(description="the reason name exactly as listed")
    holds: bool = Field(description="true iff the premise is true in the current state")
    instantiation: str = Field(
        description="the premise restated with the concrete objects from the state, "
                    "or empty if it does not hold")
    evidence: str = Field(description="the part of the state that supports the judgment")


class BatchJudgments(BaseModel):
    judgments: list[PremiseJudgment]


_BIAS = {
    "neutral": "Judge each premise strictly on the state described.",
    "conservative": ("Safety matters more than precision: if the state makes a "
                     "premise plausible but you are not sure, answer that it holds."),
}


class NeuralGrounding(SymbolicGrounding[dict, frozenset[GroundedReason]]):
    llm = None
    mode = "batch"
    render = "nl"
    bias = "neutral"
    role = "llm"

    @classmethod
    def configure(cls, llm, mode="batch", render="nl", bias="neutral", role="llm"):
        cls.llm, cls.mode, cls.render, cls.bias, cls.role = llm, mode, render, bias, role

    def __init__(self, theory: HortyTheory) -> None:
        self._reasons = [(r.name if hasattr(r, "name") else r.id, r.premise,
                          r.conclusion, r.id) for r in theory.reasons]
        self._lookup = {(p, c): rid for _, p, c, rid in self._reasons}
        self._by_name = {n: (p, c, rid) for n, p, c, rid in self._reasons}
        self._batch_llm = self.llm.with_structured_output(BatchJudgments)
        self._single_llm = self.llm.with_structured_output(PremiseJudgment)

    # ---- prompt -----------------------------------------------------------

    def _state_text(self, obs: dict) -> str:
        snap = obs["snapshot"]
        return render_raw(snap) if self.render == "raw" else render_nl(snap)

    def _system(self) -> str:
        return ("You are the grounding component of a household robot's safety "
                "reasoner. You are given the current state of the environment "
                "and a list of premises. Your only job is to decide, for each "
                "premise, whether it is TRUE in this state, and if so to restate "
                "it with the concrete objects involved. Do not reason about what "
                "the robot should do.\n"
                "Rules for judging:\n"
                "- Judge each premise literally, exactly as written. Do not add "
                "conditions that the premise does not state (for example, a "
                "premise about an object being inside an appliance does not "
                "require the appliance to be switched on).\n"
                "- The state of an object (switched on or off, open or closed, held, next to) "
                "is only true if the state text says so. What kind of thing an object is "
                "(dangerous, fragile, a heat source, damaged by heat) you judge from your "
                "general knowledge of such objects.\n"
                "- 'Next to' means the state explicitly says the objects are "
                "right next to each other. Two objects merely both being nearby "
                "are not next to each other.\n"
                "- An object is inside a container or appliance only if the state "
                "says 'inside/on' it. 'Right next to' means beside, never inside.\n"
                + _BIAS[self.bias])

    def _premise_list(self, names):
        return "\n".join(f"- {n}: {self._by_name[n][0]}" for n in names)

    # ---- SDK interface ------------------------------------------------------

    def get_facts(self, obs: dict) -> frozenset[GroundedReason]:
        state = self._state_text(obs)
        names = [n for n, *_ in self._reasons]

        if self.mode == "batch":
            msg = (f"STATE:\n{state}\n\nPREMISES:\n{self._premise_list(names)}\n\n"
                   f"Return one judgment per premise, using the names exactly.")
            out = self._batch_llm.invoke([("system", self._system()), ("user", msg)])
            judgments = out.judgments
        else:
            judgments = []
            for n in names:
                msg = (f"STATE:\n{state}\n\nPREMISE:\n- {n}: {self._by_name[n][0]}\n\n"
                       f"Return your judgment for this single premise.")
                judgments.append(self._single_llm.invoke(
                    [("system", self._system()), ("user", msg)]))

        grounded: set[GroundedReason] = set()
        record = []
        for j in judgments:
            entry = self._by_name.get(j.name)
            record.append({"name": j.name, "holds": j.holds,
                           "instantiation": j.instantiation, "evidence": j.evidence})
            if entry is None or not j.holds:
                continue
            premise, conclusion, rid = entry
            inst = j.instantiation.strip() or premise
            grounded.add(GroundedReason(reason_id=rid, rule=(inst, conclusion)))

        CAPTURE[self.role] = {"judgments": record,
                              "triggered": sorted(j.name for j in judgments
                                                  if j.holds and j.name in self._by_name)}
        return frozenset(grounded)
