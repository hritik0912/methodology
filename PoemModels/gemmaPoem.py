import subprocess
import time
import os
import re
import json
import statistics
import ollama
import chromadb
from sentence_transformers import SentenceTransformer
import logging

# 🟢 IMPORT vLLM
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# ==========================================
# 1. BACKGROUND OLLAMA SERVER SETUP
# ==========================================
print("🚀 Starting Ollama background server on an EMPTY GPU...")
ollama_path = os.path.expanduser("~/.local/bin/ollama") 

env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = "2" # Dedicated GPU for Ollama Evaluator
env["OLLAMA_HOST"] = "127.0.0.1:11434"

try:
    subprocess.Popen(
        [ollama_path, "serve"], 
        env=env, 
        stdout=subprocess.DEVNULL, 
        stderr=subprocess.DEVNULL
    )
    time.sleep(5) 
    print("✅ Ollama server is now running on isolated GPU!")
except Exception as e:
    print(f"❌ Failed to start Ollama: {e}")

# ==========================================
# 2. MAIN LLM SETUP (Gemma-2-9B via vLLM)
# ==========================================
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# 🟢 CHANGE: We only need ONE GPU for Gemma-2-9B. 
# It fits easily into 48GB VRAM.
os.environ["CUDA_VISIBLE_DEVICES"] = "5" 

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    handlers=[
        logging.FileHandler("gemma2_execution.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

logger.info("🔗 Connecting to ChromaDB...")
client = chromadb.PersistentClient(path="poetry_db")
collection = client.get_collection(name="hindwi_poems") 

# Load Embedder on CPU to save VRAM
logger.info("⏳ Loading Embedder (multilingual-e5-large) on CPU...")
embedder = SentenceTransformer('intfloat/multilingual-e5-large', device='cpu')

logger.info("⏳ Loading google/gemma-2-9b-it using vLLM...")
# 🟢 CHANGE: New Model ID
MODEL_ID = "google/gemma-2-9b-it"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# 🟢 Initialize vLLM Engine
llm = LLM(
    model=MODEL_ID,
    tensor_parallel_size=1,          # 🟢 CHANGE: Fits on 1 GPU easily
    dtype="bfloat16",
    trust_remote_code=True,
    gpu_memory_utilization=0.90,     
    enforce_eager=False              # Gemma 2 works fine with standard CUDA graphs
)
logger.info("✅ Gemma-2-9B loaded and ready!")

# ==========================================
# 3. POETRY EXPERIMENTER CLASS
# ==========================================
class PoetryExperimenter:
    def __init__(self, llm_engine, tokenizer, collection, embedder, logger):
        self.llm = llm_engine
        self.tokenizer = tokenizer
        self.collection = collection
        self.embedder = embedder
        self.logger = logger

    def _generate(self, system_prompt, user_prompt, temperature=0.7): 
        start_time = time.time()
        
        # 🟢 Gemma specific system prompt tweak
        # Gemma doesn't natively support "system" roles in its official template perfectly in all versions,
        # but vLLM handles the mapping. We make the instruction very clear in the user prompt if needed.
        
        messages = [
            {"role": "user", "content": f"{system_prompt}\n\nTask: {user_prompt}"}
        ]
        
        # Build prompt using tokenizer's template
        prompt_str = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # 🟢 vLLM Sampling Parameters
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=0.9,
            max_tokens=1024, # Gemma is concise, 1024 is plenty
            stop_token_ids=[self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<end_of_turn>")]
        )
        
        self.logger.info(f"   [Inference] Requesting vLLM generation...")
            
        outputs = self.llm.generate([prompt_str], sampling_params, use_tqdm=False)
        output_text = outputs[0].outputs[0].text.strip()
        
        elapsed_time = time.time() - start_time
        self.logger.info(f"   [Inference] Completed in {elapsed_time:.2f} seconds.")
        
        return output_text

    def find_best_poet(self, topic):
        try:
            query_vector = self.embedder.encode([f"कविता: {topic}"], convert_to_tensor=False)
            results = self.collection.query(query_embeddings=query_vector, n_results=1)
            if results['metadatas'] and results['metadatas'][0]:
                return results['metadatas'][0][0]['poet_slug']
        except Exception:
            pass
        return "ramdhari-singh-dinkar" 

    def zero_shot(self, topic):
        return self._generate("You are an expert Hindi poet. Write in Devanagari script.", f"Topic: '{topic}'")

    def few_shot(self, topic, style="ramdhari-singh-dinkar"):
        query_vector = self.embedder.encode([f"कविता: {topic}"], convert_to_tensor=False)
        res = self.collection.query(query_embeddings=query_vector, n_results=2, where={"poet_slug": style})
        
        examples_text = ""
        if res['documents'] and res['documents'][0]:
            for i, text in enumerate(res['documents'][0]):
                example_lines = "\n".join(text.split("\n")[:6]).strip()
                examples_text += f"Example {i+1}:\n{example_lines}\n\n"
                
        sys = "You are a famous Hindi poet. Analyze the style of the examples below and write a NEW poem on the given topic."
        user = (
            f"Style Examples:\n{examples_text}\n"
            f"Now write a 4-8 line poem on '{topic}'. Do not copy the examples."
        )
        return self._generate(sys, user)

    def rag_style_conditioned(self, topic, style="ramdhari-singh-dinkar"):
        query_vector = self.embedder.encode([f"कविता: {topic}"], convert_to_tensor=False)
        res = self.collection.query(query_embeddings=query_vector, n_results=2, where={"poet_slug": style})
        
        context = ""
        if res['documents'] and res['documents'][0]:
            for i, text in enumerate(res['documents'][0]):
                context += f"Context {i+1}:\n{text[:300]}...\n"
                
        user = (
            f"Study this poetic style deeply:\n{context}\n\n"
            f"Write a NEW poem on '{topic}' adopting this specific tone and vocabulary."
        )
        return self._generate("You are a creative poet.", user)
    
    def plan_then_generate(self, topic):
        sys = "You are a poet and critic. Plan the poem first, then write it."
        user = (
            f"Topic: '{topic}'\n"
            "1. Decide Mood\n2. Choose 5 keywords\n3. Choose a Metaphor\n4. Write the Poem in Hindi."
        )
        return self._generate(sys, user)

    def self_critique(self, topic):
        draft = self.zero_shot(topic)
        critique_prompt = f"Critique this Hindi poem and find 2 flaws:\n{draft}"
        critique = self._generate("You are a harsh literary critic.", critique_prompt)
        
        refine_prompt = f"Original:\n{draft}\n\nCritique:\n{critique}\n\nWrite a better version."
        final = self._generate("You are a master poet.", refine_prompt)
        return f"--- DRAFT ---\n{draft}\n\n--- CRITIQUE ---\n{critique}\n\n--- FINAL ---\n{final}"

    def constraint_based(self, topic):
        user = (
            f"Topic: '{topic}'\n"
            "Rules:\n1. Exactly 4 lines.\n2. Do NOT use the word 'आसमान' (Sky).\n3. End with the word 'कहानी'."
        )
        return self._generate("You follow rules strictly.", user)

    def temp_experiment(self, topic):
        low_temp = self._generate("You are a poet.", f"Topic: '{topic}'", temperature=0.2)
        high_temp = self._generate("You are a poet.", f"Topic: '{topic}'", temperature=1.2)
        return f"--- TEMP 0.2 ---\n{low_temp}\n\n--- TEMP 1.2 ---\n{high_temp}"

    def persona_based(self, topic):
        sys = "You are a 19th-century melancholic Hindi poet using archaic vocabulary (Tatsam)."
        return self._generate(sys, f"Write about: '{topic}'")

    def prompt_variants(self, topic):
        variant_a = self._generate("Write a poem.", f"Topic: {topic}")
        variant_b = self._generate(
            "You are a Sahitya Akademi Award winner. Your language is symbolic and heart-touching.", 
            f"Write a masterpiece on '{topic}'."
        )
        return f"--- BASIC ---\n{variant_a}\n\n--- ENGINEERED ---\n{variant_b}"

    def multi_agent(self, topic):
        sys_a = "You are Poet A (Calm, Nature lover). Write Stanza 1 (4 lines)."
        stanza_1 = self._generate(sys_a, f"Topic: '{topic}'")
        
        sys_b = "You are Poet B (Fiery, Revolutionary). Write Stanza 2 (4 lines) continuing Poet A's work."
        stanza_2 = self._generate(sys_b, f"Poet A wrote:\n{stanza_1}\n\nNow write your stanza in a rebellious tone.")
        
        return f"--- STANZA 1 ---\n{stanza_1}\n\n--- STANZA 2 ---\n{stanza_2}"

    def auto_eval(self, topic, generated_poem, poet_name, reference_poems, num_evals=5):
        # 🟢 Using your existing Ollama evaluator logic
        ollama_client = ollama.Client(host='http://127.0.0.1:11434')
        self.logger.info("   [Evaluation] Starting Dual-LLM evaluation via Ollama...")

        system_instruction = "You are an expert Hindi literary critic. Output ONLY valid JSON."
        evaluation_prompt = (
            f"Topic: {topic}\nTarget Style: {poet_name}\n"
            f"--- REFERENCE ---\n{reference_poems}\n--- GENERATED ---\n{generated_poem}\n"
            "Rate 1-10: fluency, coherence, relevance, creativity, style_similarity.\n"
            'Return JSON: {"fluency": 0, "coherence": 0, "relevance": 0, "creativity": 0, "style_similarity": 0}'
        )

        final_scores = {"Llama-3.1": {}, "Gemma-2": {}}
        metrics = ['fluency', 'coherence', 'relevance', 'creativity', 'style_similarity']

        def clean_json(text): 
            text = text.strip()
            if text.startswith("```json"): text = text[7:]
            if text.endswith("```"): text = text[:-3]
            match = re.search(r'\{.*\}', text, re.DOTALL)
            return match.group(0) if match else text

        for model_name, dict_key in [('llama3.1', 'Llama-3.1'), ('gemma2', 'Gemma-2')]:
            raw_scores = {m: [] for m in metrics}
            for i in range(num_evals):
                try:
                    response = ollama_client.chat(
                        model=model_name, 
                        messages=[{'role': 'system', 'content': system_instruction}, {'role': 'user', 'content': evaluation_prompt}], 
                        options={'temperature': 0.7, 'num_ctx': 4096},
                        keep_alive=0 
                    )
                    parsed = json.loads(clean_json(response['message']['content']))
                    for m in metrics:
                        if m in parsed: raw_scores[m].append(float(parsed[m]))
                except Exception: pass
            
            aggregated = {}
            for m in metrics:
                vals = raw_scores[m]
                if len(vals) > 0:
                    aggregated[m] = {"mean": round(statistics.mean(vals), 2), "std_dev": round(statistics.stdev(vals), 2) if len(vals) > 1 else 0}
            final_scores[dict_key] = aggregated

        return json.dumps(final_scores, indent=2, ensure_ascii=False)

    def run_all(self, topics):
        experiments = {
            "Zero-Shot": self.zero_shot,
            "Few-Shot": self.few_shot,
            "RAG Style": self.rag_style_conditioned,
            "Plan-Then-Generate": self.plan_then_generate,
            "Self-Critique": self.self_critique,
            "Constraints": self.constraint_based,
            "Temperature": self.temp_experiment,
            "Persona": self.persona_based,
            "Prompt Variants": self.prompt_variants,
            "Multi-Agent": self.multi_agent
        }

        output_dir = "Gemma2_Output"
        os.makedirs(output_dir, exist_ok=True)
        self.logger.info(f"📂 Execution started. Saving to '{output_dir}/'")
        
        for topic in topics:
            self.logger.info(f"\n{'='*50}\n🌟 TOPIC: {topic}\n{'='*50}")
            best_poet = self.find_best_poet(topic)
            safe_filename = re.sub(r'[^\w\s-]', '', topic).strip().replace(' ', '_')
            file_path = os.path.join(output_dir, f"{safe_filename}.txt")
            
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(f"TOPIC: {topic}\nPOET MATCH: '{best_poet}'\n{'='*50}\n\n")
                
                for exp_name, exp_func in experiments.items():
                    self.logger.info(f"🧪 {exp_name}")
                    f.write(f"🧪 {exp_name}\n{'-'*50}\n")
                    
                    try:
                        # Fetch RAG context only if needed
                        context = ""
                        if exp_name in ["RAG Style", "Few-Shot"]:
                            q_vec = self.embedder.encode([f"कविता: {topic}"], convert_to_tensor=False)
                            res = self.collection.query(query_embeddings=q_vec, n_results=3, where={"poet_slug": best_poet})
                            if res['documents'] and res['documents'][0]:
                                context = "\n".join([t[:300] for t in res['documents'][0]])

                        poem = exp_func(topic, style=best_poet) if exp_name in ["RAG Style", "Few-Shot"] else exp_func(topic)
                        eval_json = self.auto_eval(topic, poem, best_poet, context)
                        
                        f.write(f"📜 POEM:\n{poem}\n\n🧠 EVALUATION:\n{eval_json}\n\n{'='*50}\n\n")
                    except Exception as e:
                        self.logger.error(f"❌ Error in {exp_name}: {e}")
                        f.write(f"❌ ERROR: {e}\n\n")

        self.logger.info("✅ Done!")

# ==========================================
# 4. EXECUTION
# ==========================================
TEST_TOPICS = [
    "🌿 प्रकृति (Nature): बारिश की पहली बूंद",
    "❤️ भावनात्मक (Emotional): अधूरी मोहब्बत",
    "🌍 सामाजिक (Social): नारी शक्ति",
    "🧠 दार्शनिक (Philosophical): समय का चक्र",
    "🎭 रचनात्मक / अनोखा (Creative): टूटी हुई घड़ी की कहानी",
    "🌅 आशावादी (Optimistic): ख्वाबों का आसमान",
    "😔 उदास (Sad): सूनी राहें",
    "🌾 नॉस्टैल्जिक (Nostalgic): मिट्टी की खुशबू",
    "💔 दर्दभरा (Painful): चुप्पी का बोझ",
    "✨ प्रेरणादायक (Inspirational): उम्मीद की किरण",
    "🏡 स्मृतिपूर्ण (Reminiscent): बचपन की गलियाँ",
    "🌆 अकेलापन (Lonely): अजनबी शहर",
    "❤️ रोमांटिक (Romantic): दिल की दस्तक",
    "🤝 भावुक (Emotional): रिश्तों की डोर",
    "⏳ दार्शनिक (Reflective): वक्त की रेत",
    "🕊️ उत्साहपूर्ण (Energetic): सपनों की उड़ान",
    "🎈 निराशाजनक (Hopeless): टूटी हुई पतंग",
    "🪞 गंभीर (Serious): सच का आईना",
    "📜 विरहपूर्ण (Separation): आख़िरी ख़त",
    "🌄 सकारात्मक (Positive): नई सुबह"
]

experimenter = PoetryExperimenter(llm, tokenizer, collection, embedder, logger)
experimenter.run_all(TEST_TOPICS)