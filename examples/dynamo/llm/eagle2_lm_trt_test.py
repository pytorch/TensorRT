import time
import torch
import torch_tensorrt
from transformers import AutoProcessor, AutoModel
from transformers.modeling_outputs import CausalLMOutputWithPast

# Re-use helper utilities shipped with the other dynamo examples
# (export_llm no longer used after switching to inputs_embeds path)
from utils import export_llm, generate
import transformers.models.qwen2.modeling_qwen2 as mq

mq.ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = mq.ALL_ATTENTION_FUNCTIONS["sdpa"]
# Load the base model and processor
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from register_sdpa import *


def load_eagle2_model(device="cuda:0"):
    model_id = "nvidia/Eagle2-2B"
    model = (
        AutoModel.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.float16)
        .to(device)
        .eval()
    )
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, use_fast=True)
    return model, processor

# ------------------------------------------------------------------
# New helper: compile LM to take inputs_embeds (not input_ids)
# ------------------------------------------------------------------

class LMNoCache(torch.nn.Module):
    """Wrapper exposing inputs_embeds-only forward (no KV-cache)"""
    def __init__(self, lm):
        super().__init__()
        self.lm = lm

    def forward(self, inputs_embeds):# , position_ids=None):
        out = self.lm(inputs_embeds=inputs_embeds) #, position_ids=position_ids, use_cache=False)
        # Ensure a CausalLMOutput/loss-style object with .logits attribute is returned
        if hasattr(out, "logits"):
            return out.logits
        else:
            # When using compile path we'll want a simple tensor
            return out


def compile_eagle2_lm_with_trt_embed(language_model, example_embeds, device="cuda:0"):
    """Compile language model that expects inputs_embeds using Torch-TensorRT."""

    lm_wrap = LMNoCache(language_model).to(device).eval()

    # Dynamic shapes (batch and seq len) similar to earlier
    # B = torch.export.Dim("batch", min=1, max=4)
    S = torch.export.Dim("seq", min=2048, max=2176)
    dyn_shapes = {"inputs_embeds": {1: S}}

    with torch.inference_mode():
        exported = torch.export.export(
            lm_wrap,
            (example_embeds,),
            dynamic_shapes=dyn_shapes,
            strict=False,
        )

    trt_mod = torch_tensorrt.dynamo.compile(
        exported,
        inputs=[example_embeds],
        enabled_precisions={torch.float32},
        device=device,
        use_explicit_typing=True,
        use_fp32_acc=True,
    )
    return trt_mod

# ------------------------------------------------------------------
# Timing helper using inputs_embeds path
# ------------------------------------------------------------------

def timed_generate_steps_embeds(model, embed_layer, input_ids, eos_id, max_new_tokens, is_trt=False):
    """Generate using embeddings, timing each token. For TRT model the wrapper returns logits tensor."""

    token_seq = input_ids.clone()
    embeds_seq = embed_layer(token_seq)

    step_times = []
    n_to_generate = max_new_tokens - token_seq.shape[1]

    for _ in range(n_to_generate):
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        position_ids = torch.arange(embeds_seq.shape[1], device=embeds_seq.device).unsqueeze(0)

        with torch.no_grad():
            if is_trt:
                # TRT module returns logits tensor directly
                logits = model(inputs_embeds=embeds_seq) #, position_ids=position_ids) #osition_ids)
            else:
                logits = model(inputs_embeds=embeds_seq, position_ids=position_ids).logits

        next_token = torch.argmax(logits[:, -1, :], dim=-1)

        # Append token & embedding
        token_seq = torch.cat([token_seq, next_token[:, None]], dim=-1)
        next_embed = embed_layer(next_token).unsqueeze(1)  # (B,1,C)
        embeds_seq = torch.cat([embeds_seq, next_embed], dim=1)

        torch.cuda.synchronize()
        step_times.append(time.perf_counter() - t0)

        if (next_token == eos_id).all():
            break

    return token_seq, sum(step_times), step_times

if __name__ == "__main__":
    device = "cuda:0"
    torch.cuda.set_device(device)

    # Load the model and processor
    base_model, processor = load_eagle2_model(device)
    language_model = base_model.language_model

    # Prepare input
    # prompt_tokens = ["token"] * 2048
    # prompt = " ".join(prompt_tokens)
    prompt = "In a world where technology has rapidly advanced, humanity faces both extraordinary opportunities and unprecedented challenges, and every decision we make resonates beyond the confines of our own lives, touching ecosystems and generations yet to come, weaving a complex tapestry of moral, social, environmental, and existential threads that demand scrutiny, compassion, and foresight, for when clean energy systems powered by sunlight and wind seamlessly integrate with cities bustling with quantum‑linked communication networks and autonomous vehicles gliding silently through streets once choked by emissions, we marvel at the ingenuity of human creativity, yet we cannot dismiss the silent questions that echo beneath this surface of triumph: what is the human soul in an age when machines learn, adapt, and even create art with brushstrokes and musical notes indistinguishable from those born of human experience; what is empathy when algorithms parse every nuance of facial expression and voice intonation faster than any human can perceive, yet lack the warm pulse of a beating heart; what is responsibility when decisions once made by elected leaders may soon be influenced or even executed by artificial intelligences operating at scales and speeds beyond our comprehension; and who will hold them accountable when lines blur between human and machine actions, especially in fields like medicine, justice, finance, and warfare, where the consequences of errors—or intentional misuse—could shape destinies and rewrite histories; thus scientists and ethicists, engineers and artists, lawmakers and activists alike gather in symposiums and digital forums, not merely to celebrate advances in neural networks, robotics, and bioengineering, but to debate treaties, design protocols, and draft manifestos that ensure artificial general intelligence, augmented reality, genetic editing, and synthetic lifeforms are deployed with humility and oversight, preserving biodiversity, human rights, and cultural diversity, while fostering universal access and preventing monopolies of power, for if we aspire to build a civilization that uplifts all, we must integrate values into code, humanity into hardware, stewardship into software, and conscience into every innovation, continually questioning not only what we can build and compute, but what we should, because at the heart of this quest lies the enduring question: will the sum of our inventions enrich the human spirit, deepen our understanding of our shared world, and inspire wonder, or will they marginalize the vulnerable, amplify inequality, and erode the fragile web of trust that binds us, and so we walk forward, both awed and anxious, united by imagination, guided by ethics, and determined to shape a future where machines serve humanity, not dominate it. In a world where technology evolves faster than ever before, every step humanity takes shapes the future. We build machines to think, systems to learn, and networks to connect our lives. Yet in this progress lies the need for careful reflection, for ethics, for understanding. The questions we ask now—about identity, about intelligence, about responsibility—will echo through the decades to come. Will we remain the masters of our tools, or will the tools redefine us? We construct machines that analyze data, platforms that connect distant communities, and networks that transform communication faster than light travels—but in doing so, we must pause to reflect on ethics and purpose. Every advancement in artificial intelligence and robotics forces us to reconsider our responsibilities, our values, and how we define consciousness and collaboration in the century ahead. Will we remain the architects of progress and guides of innovation, or will our creations one day frame the questions we ask and redefine our own identities? As these inquiries echo through boardrooms, classrooms, and laboratories alike, the narrative of our collective future unfolds—layered, uncertain, and brimming with possibility. In a world where technology evolves faster than ever, humanity shapes the future with every innovation and challenge we embrace, questioning what it truly means to be wise. We construct machines that analyze data, platforms that connect distant communities, and networks that transform communication faster than light travels—but in doing so, we must pause to reflect on ethics and purpose. As we stand at the intersection of biology and silicon, where neurons meet neural networks and consciousness brushes against computation, we are compelled to ask whether intelligence alone suffices for wisdom, or whether something deeper—empathy, humility, and lived experience—is needed to truly guide the future. The decisions we encode into algorithms, the datasets we choose to prioritize, and the goals we assign to autonomous systems will not only reflect our values but also reinforce and propagate them at scale. This demands that we not only act as engineers but as philosophers, not merely as builders but as stewards of ethical continuity. In the coming decades, the questions of agency, fairness, transparency, and accountability will transcend technical discussions and become the cornerstones of societal trust. Will we ensure that these systems amplify the best of humanity—justice, inclusivity, curiosity—or will we allow them to mirror and magnify our biases, blind spots, and inequities? These choices are not abstract; they are encoded, line by line, in every architecture we design, in every prompt we fine-tune, in every interaction we automate. And so, as we code, we must also care—as we scale, we must also reflect—as we create, we must also contemplate. For the future is not an inevitability dictated by innovation alone, but a living construct of intention, design, and collective moral vision. Let us then proceed not with blind acceleration, but with deliberate grace—shaping a world not just of greater capability, but of deeper meaning. As we navigate this unfolding era, it becomes increasingly evident that the fusion of human intuition and artificial intelligence is not merely a technical endeavor, but a deeply humanistic one. In laboratories glowing with the light of quantum processors, and in classrooms where students collaborate with AI tutors, we see the contours of a hybrid intellect taking shape—one that is capable of not just processing data, but perceiving context; not just identifying patterns, but anticipating meaning. In this synthesis lies immense promise, but also peril. For just as we amplify our cognitive capacities, we also inherit new vulnerabilities: the manipulation of perception, the erosion of privacy, the gamification of truth. These are not distant threats; they are present realities woven into the very algorithms that curate our news, recommend our friends, and filter our reality. And so, the responsibility falls on us—not just as technologists, but as citizens—to design systems that are resilient, transparent, and just. Let us imagine a society where knowledge is not hoarded but harmonized, where open-source AI models empower every voice rather than entrench power in the hands of the few. Let us envision governance frameworks that do not merely regulate innovation, but cultivate it with foresight and fairness. In such a world, data is not a commodity, but a shared legacy. Intelligence is not centralized, but distributed. Power is not wielded, but shared. To achieve this, we must reimagine education as a lifelong collaboration between human curiosity and machine precision. We must invest not only in infrastructure, but in imagination. We must teach our children not merely how to use tools, but how to ask questions no tool can answer. And we ourselves must be willing to unlearn, to evolve, to become something more than users—we must become co-authors of the future. And so, with each innovation we embrace, let us also embrace humility. With each challenge we face, let us summon empathy. With each system we build, let us embed within it the memory of what it means to be human—flawed, compassionate, and infinitely curious. For in the end, the legacy of our age will not be measured by the intelligence of our machines, but by the wisdom with which we choose to use them. As we look ahead into the next frontier of human potential, the convergence of biological insight, computational precision, and environmental consciousness will define not only our capabilities but our collective character. The boundaries between disciplines—once rigid walls—are dissolving into bridges, enabling breakthroughs that could eradicate disease, reverse environmental degradation, and redefine how we understand life itself. Already, bioengineers are harnessing machine learning to map genetic pathways in unprecedented detail, predicting not only the onset of hereditary illness but potential interventions that were inconceivable a decade ago. In parallel, ecologists are deploying AI-driven drones to monitor endangered species, assess biodiversity, and model the impact of climate interventions in real time. But in this era of hyperconnected intelligence, where the data of billions flows through invisible channels, we must ask: who owns this knowledge? Who decides how it is used, monetized, or restricted? Who ensures that the benefits of these revolutions are distributed justly and equitably? For if we allow innovation to outpace inclusion, if we design for efficiency without empathy, we risk building systems that replicate the very injustices we claim to solve. Technology must not merely serve the loudest voices or the wealthiest stakeholders—it must listen for the silenced, account for the overlooked, and empower the disempowered. To move toward this vision, we must cultivate a culture of intentional design—where every product, process, and policy is shaped not only by feasibility and profit, but by purpose and principle. And in parallel, we must confront the inner transformation required of ourselves. The age of intelligent machines demands an equally intelligent citizenry—one that can critically engage with complex systems, discern signal from noise, and make decisions not only based on what is possible, but on what is right. In this light, education must evolve. No longer can we rely solely on memorization or rote methods; we must instead empower learners to ask better questions, embrace ambiguity, and navigate uncertainty with ethical courage and emotional intelligence. This is not a call for techno-utopianism, nor a lament for a bygone analog age. It is a call for balance—for re-centering the human amidst the algorithmic. It is an invitation to participate fully in the design of our destiny, to recognize that the future is not something that happens to us, but something we shape through every act of awareness, advocacy, and alignment. Each line of code, each policy decision, each act of resistance or cooperation. And so"
    
    input_ids = processor.tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    embed_layer = language_model.get_input_embeddings()

    # Baseline (Pure PyTorch) with inputs_embeds
    torch_out, torch_total, torch_steps = timed_generate_steps_embeds(
        language_model,
        embed_layer,
        input_ids.clone(),
        processor.tokenizer.eos_token_id,
        2048+128,
        is_trt=False,
    )
    print("PyTorch generated text   :", processor.tokenizer.decode(torch_out[0], skip_special_tokens=True))

    print("\nCompiling Eagle2 language model (inputs_embeds) with Torch-TensorRT …")

    # Build a dummy embeds tensor for export/compile
    dummy_embeds = embed_layer(input_ids.clone())
    trt_model = compile_eagle2_lm_with_trt_embed(language_model, dummy_embeds, device)

    trt_out, trt_total, trt_steps = timed_generate_steps_embeds(
        trt_model,
        embed_layer,
        input_ids.clone(),
        processor.tokenizer.eos_token_id,
        2048+128,
        is_trt=True,
    )

    # Results
    print("\n================  RESULTS  ================")
    print("PyTorch generated text   :", processor.tokenizer.decode(torch_out[0][input_ids.shape[1]:], skip_special_tokens=True))
    print("TensorRT generated text  :", processor.tokenizer.decode(trt_out[0][input_ids.shape[1]:], skip_special_tokens=True))
    print("Tokens identical         :", torch.equal(torch_out[0][input_ids.shape[1]:], trt_out[0][input_ids.shape[1]:]))

    # Aggregate timings
    print(f"PyTorch total time   : {torch_total:.3f} s")
    print(f"TensorRT total time  : {trt_total:.3f} s")
    if trt_total > 0:
        print(f"Speed-up (×)        : {torch_total / trt_total:.2f}")

    # Per-token breakdown (ms)
    def _fmt(ts):
        return [f"{t*1000:.2f}" for t in ts]

    print("\nPyTorch per-token times (ms):", _fmt(torch_steps))
    print("TensorRT per-token times (ms):", _fmt(trt_steps))
    print("===========================================") 