# lora-finetune-kaggle

Fine-tune Qwen3-4B-Instruct-2507 into "DesiTutor" — a Hinglish AI-engineering tutor — using QLoRA on a free Kaggle T4 GPU. A 120B teacher model on Groq (`openai/gpt-oss-120b`) generates the 320-example synthetic dataset, Unsloth + TRL train LoRA adapters in ~12 minutes, a judged eval proves the tuned model beats the base, and the result exports to GGUF for local Ollama inference. Total cost: ₹0.

This project runs entirely in a Kaggle notebook — there is no local `src/`. Follow [GUIDE.md](GUIDE.md) cell by cell.
