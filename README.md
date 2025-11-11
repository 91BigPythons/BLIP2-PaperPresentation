# BLIP-2 — Bootstrapping Language–Image Pretraining with Frozen Image Encoders and LLMs

> **Course:** DS 5690 — Gen AI Models in Theory & Practice (2025F)  
> **Presenter:** *(your name)*  
> **Paper:** Li et al., 2023 — “BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models” (arXiv:2301.12597)

---

## TL;DR
BLIP‑2 makes multimodal training **much cheaper** by keeping the **image encoder** and the **LLM** **frozen**, and learning only a lightweight **Querying Transformer (Q‑Former)** to bridge images → language. It pretrains Q‑Former in **two stages**: (1) representation learning with an image encoder (ITC/ITM/ITG), and (2) generative learning by prompting a frozen LLM with projected visual queries. Despite far fewer *trainable* parameters, BLIP‑2 matches or beats larger end‑to‑end models on zero‑shot VQA, captioning, and retrieval.

---

## Five‑Minute Overview (Context → Problem → Approach → Results)
- **Context.** Vision–language pretraining (VLP) models got huge and expensive; end‑to‑end training is prohibitive.  
- **Problem.** How to achieve strong multimodal performance **without** full end‑to‑end training?  
- **Approach.** Freeze a **strong image encoder** (e.g., CLIP ViT) and a **strong LLM** (e.g., Flan‑T5/OPT), and train a small **Q‑Former** that extracts a compact set of **learned visual queries** and **prompts** the LLM.  
- **Results.** State‑of‑the‑art **zero‑shot** VQA, strong captioning and retrieval—while using vastly fewer *trainable* params than e2e models.

---

## Architecture (High‑Level)

```
Image            Q‑Former (trainable)                     Frozen LLM
Encoder          ┌───────────────┐   Linear proj.        ┌───────────┐
(ViT)  ───────►  │ 32 learned    ├───────────────►       │ Text gen. │
features         │ query tokens  │                      │ or QA     │
                 │ (self‑attn)   │                      └───────────┘
                 │ + cross‑attn  │
                 └───────────────┘
                       ▲
                 Frozen image features
```

**Key piece: Q‑Former**  
- Maintains a small, fixed number of trainable **query tokens** (e.g., 32×768).  
- **Cross‑attends** to frozen ViT features to pull out language‑relevant information.  
- Feeds projected queries as **soft visual prompts** to the frozen LLM.

---

## Two‑Stage Pretraining

### Stage 1 — *Vision–Language Representation Learning* (with frozen image encoder)
Jointly optimize three objectives to make queries language‑relevant:  
- **ITC** (Image–Text Contrastive): align global image/text embeddings.  
- **ITM** (Image–Text Matching): binary “match?” classification with hard negatives.  
- **ITG** (Image‑Grounded Generation): force queries to contain all info needed to generate text.

### Stage 2 — *Vision→Language Generative Learning* (with frozen LLM)
- Project query features to the LLM’s token dim and **prepend** them to the text tokens (soft prompts).  
- Train only **Q‑Former + projection**, keeping LLM **frozen**, to enable captioning/QA conditioned on visual prompts.

---

## Formal Pseudocode

```python
# Notation:
#   E_img: frozen image encoder (ViT)
#   Q: Q‑Former (trainable)
#   P: linear projection from Q‑Former output to LLM token dim (trainable)
#   LLM: frozen large language model (decoder or encoder‑decoder)
#   ITC, ITM, ITG: stage‑1 losses; LM_loss / PrefixLM_loss: stage‑2 losses

# ---------- Stage 1: Representation Learning ----------
for image, text in D_image_text:
    V = E_img(image)                  # frozen features
    Z = Q(image_feats=V, text=None)   # queries attend to V via cross‑attention
    loss = ITC(Z, text) + ITM(Z, text) + ITG(Z, text)
    update(Q)                         # only Q‑Former is updated

# ---------- Stage 2: Generative Learning ----------
for image, text in D_image_text:
    V = E_img(image)                  # frozen features
    Z = Q(image_feats=V, text=None)   # extract visual queries (language‑relevant)
    V_prompt = P(Z)                   # project to LLM embedding size
    loss = LM_loss(LLM(prompt=V_prompt, text=text))  # or PrefixLM for encoder‑decoder
    update(Q, P)                      # only Q‑Former + projection are updated
```

---

## Critical Analysis (What’s strong / What’s missing)
**Strengths**
- Large *frozen* components preserve pretrained knowledge; few trainable params → **compute‑efficient**.
- Two‑stage scheme reduces **catastrophic forgetting** and improves zero‑shot performance.
- Modular: can “harvest” better ViTs/LLMs over time.

**Limitations / Open Questions**
- Single‑pair pretraining lacks multi‑image interleaving → weak **in‑context** multimodal examples; limited few‑shot gains.  
- Quality still bounded by the LLM’s knowledge (bias, hallucination).  
- Visual reasoning can fail on novel or complex scenes; struggles with very long visual contexts.

---

## Impact
- Helped establish the **“frozen LLM + visual adapter”** recipe used by **LLaVA**, **InstructBLIP**, **MiniGPT‑4**, etc.  
- Lowered the barrier to building visual assistants on modest compute while staying competitive with very large e2e models.

---

## Demo (Captioning + VQA)
Use the notebook [`demo.ipynb`](./demo.ipynb). It loads `Salesforce/blip2-flan-t5-xl` from 🤗 Transformers, captions an image, and answers a visual question.

**Environment (suggested)**
```bash
pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # or CPU wheels
pip install -U transformers accelerate pillow safetensors
```

**Run**
1. Open the notebook, set `IMAGE_PATH` to a local file (or URL).  
2. Run the caption cell.  
3. Set a `question` string and run the VQA cell.

---

## Resource Links
1. Paper: https://arxiv.org/abs/2301.12597  
2. BLIP‑2 in LAVIS (Salesforce): https://github.com/salesforce/LAVIS/tree/main/projects/blip2  
3. 🤗 Model Card (Flan‑T5 XL): https://huggingface.co/Salesforce/blip2-flan-t5-xl  
4. 🤗 Model Card (OPT 2.7B): https://huggingface.co/Salesforce/blip2-opt-2.7b  
5. Colab‑style starter (community): https://colab.research.google.com/github/salesforce/LAVIS/blob/main/docs/source/tutorials/BLIP2_captioning.ipynb

---

## Citation
```bibtex
@article{li2023blip2,
  title   = {BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models},
  author  = {Li, Junnan and Li, Dongxu and Savarese, Silvio and Hoi, Steven},
  journal = {arXiv preprint arXiv:2301.12597},
  year    = {2023}
}
```

---

## Appendix: Notes for Presentation (Rubric‑friendly)
- **Overview:** keep to 5 min; state problem clearly (cost), highlight two‑stage idea & wins.  
- **Architecture:** paste the pseudocode; keep diagram simple; emphasize frozen ViT/LLM and trainable Q‑Former.  
- **Critical Analysis:** cover in‑context limitation + reliance on LLM knowledge.  
- **Impacts:** mention how it shaped later visual chatbots.  
- **Two audience questions:**  
  1) *Why freeze the LLM instead of finetuning it end‑to‑end?*  
  2) *How does Stage‑1 representation learning prevent catastrophic forgetting in Stage‑2?*
