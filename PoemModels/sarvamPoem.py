import subprocess
import time
import os
import re
import json
import statistics
import ollama
import chromadb
from pathlib import Path
from tqdm.auto import tqdm
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
env["CUDA_VISIBLE_DEVICES"] = "2" # MUST be different from sarvam-m GPUs
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
# 2. MAIN LLM SETUP (Sarvam-M via vLLM & Embedder)
# ==========================================
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5,6" # 🟢 Exposing 2 GPUs for vLLM

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    handlers=[
        logging.FileHandler("sarvam_execution.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

logger.info("🔗 Connecting to ChromaDB...")
client = chromadb.PersistentClient(path="poetry_db")
collection = client.get_collection(name="hindwi_poems") 

# 🟢 Moving Embedder to CPU to save VRAM for vLLM
logger.info("⏳ Loading Embedder (multilingual-e5-large) on CPU...")
embedder = SentenceTransformer('intfloat/multilingual-e5-large', device='cpu')

logger.info("⏳ Loading sarvamai/sarvam-m using vLLM (This takes a minute)...")
MODEL_ID = ""

# Load Tokenizer separately to handle prompt formatting
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, fix_mistral_regex=True)

# 🟢 Initialize vLLM Engine
llm = LLM(
    model=MODEL_ID,
    tensor_parallel_size=2,          # Uses both A6000s simultaneously (GPUs 5 & 6)
    dtype="bfloat16",
    trust_remote_code=True,
    gpu_memory_utilization=0.90,     # Leaves 10% VRAM buffer to prevent crashes
    enforce_eager=True               # Recommended for some specific reasoning architectures
)
logger.info("✅ All models loaded and ready for SUPER FAST experiments!")

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

    def _generate(self, system_prompt, user_prompt, temperature=0.5): 
        start_time = time.time()
        
        # 🟢 PROMPT HACK: Force the model to think less and write more
        system_prompt += " (संक्षेप में सोचें और सीधे उत्तर दें।)"
            
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # vLLM needs a raw string, so we use the tokenizer to build the template string
        prompt_str = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # 🟢 vLLM Sampling Parameters
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=0.9,
            max_tokens=1536, # Hard cap to prevent endless generation
            stop_token_ids=[self.tokenizer.eos_token_id]
        )
        
        self.logger.info(f"   [Inference] Requesting vLLM generation...")
            
        # 🟢 Generate using vLLM (use_tqdm=False hides the progress bar to keep logs clean)
        outputs = self.llm.generate([prompt_str], sampling_params, use_tqdm=False)
        raw_output = outputs[0].outputs[0].text.strip()
        
        if "</think>" in raw_output:
            parts = raw_output.split("</think>")
            thoughts = parts[0].replace("<think>", "").strip() 
            self.logger.info(f"   [Thoughts] Model reasoned for {len(thoughts.split())} words.")
            clean_output = parts[-1].strip()
        else:
            clean_output = raw_output
        
        elapsed_time = time.time() - start_time
        self.logger.info(f"   [Inference] Completed in {elapsed_time:.2f} seconds.")
        
        return clean_output

    def find_best_poet(self, topic):
        try:
            query_vector = self.embedder.encode([f"कविता: {topic}"], convert_to_tensor=False)
            results = self.collection.query(query_embeddings=query_vector, n_results=1)
            if results['metadatas'] and results['metadatas'][0]:
                poet = results['metadatas'][0][0]['poet_slug']
                self.logger.info(f"   [RAG] Found best poet match: {poet}")
                return poet
        except Exception as e:
            self.logger.error(f"⚠️ Error finding best poet: {e}")
        return "ramdhari-singh-dinkar" 

    def zero_shot(self, topic):
        return self._generate("आप एक उत्कृष्ट हिंदी कवि हैं।", f"विषय: '{topic}' पर एक कविता लिखें।")

    def few_shot(self, topic, style="ramdhari-singh-dinkar"):
        query_vector = self.embedder.encode([f"कविता: {topic}"], convert_to_tensor=False)
        res = self.collection.query(query_embeddings=query_vector, n_results=2, where={"poet_slug": style})
        
        examples_text = ""
        if res['documents'] and res['documents'][0]:
            for i, text in enumerate(res['documents'][0]):
                example_lines = "\n".join(text.split("\n")[:6]).strip()
                examples_text += f"उदाहरण {i+1}:\n{example_lines}\n\n"
                
        sys = "आप एक प्रख्यात हिंदी कवि हैं। आपका काम दी गई शैली को समझना और उसी अंदाज़ में एक नई रचना करना है।"
        user = (
            f"यहाँ कुछ उदाहरण दिए गए हैं:\n{examples_text}\n"
            f"अब, '{topic}' विषय पर 4 से 8 पंक्तियों की एक नई और मौलिक (original) कविता लिखें। "
            f"ध्यान रहे, आपको उदाहरणों की पंक्तियाँ नहीं दोहरानी हैं, केवल उनकी भावना और लय का उपयोग करके नए शब्द लिखने हैं।"
        )
        return self._generate(sys, user)

    def rag_style_conditioned(self, topic, style="ramdhari-singh-dinkar"):
        query_vector = self.embedder.encode([f"कविता: {topic}"], convert_to_tensor=False)
        res = self.collection.query(query_embeddings=query_vector, n_results=2, where={"poet_slug": style})
        
        context = ""
        if res['documents'] and res['documents'][0]:
            for i, text in enumerate(res['documents'][0]):
                context += f"संदर्भ {i+1}:\n{text[:300]}...\n"
                
        user = (
            f"इस शैली का गहराई से अध्ययन करें:\n{context}\n\n"
            f"अब '{topic}' विषय पर एक बिल्कुल नई कविता लिखें। "
            f"संदर्भ से पंक्तियाँ न चुराएं, बल्कि अपनी कल्पना से उसी शैली में नए शब्द पिरोएं।"
        )
        return self._generate("आप एक रचनात्मक और मौलिक (original) कवि हैं।", user)
    
    def plan_then_generate(self, topic):
        sys = "आप एक कवि और आलोचक हैं। पहले कविता की योजना बनाएं, फिर कविता लिखें।"
        user = (
            f"विषय: '{topic}'\n"
            "1. भाव (Mood) तय करें。\n"
            "2. 5 मुख्य शब्द (Vocabulary) चुनें。\n"
            "3. अलंकार (Metaphor) सोचें。\n"
            "4. अंत में 'कविता:' शीर्षक के साथ कविता लिखें।"
        )
        return self._generate(sys, user)

    def self_critique(self, topic):
        draft = self.zero_shot(topic)
        critique_prompt = f"इस कविता की आलोचना करें और 2 कमियां निकालें (लय या शब्द चयन):\n{draft}"
        critique = self._generate("आप एक कठोर आलोचक हैं।", critique_prompt)
        
        refine_prompt = f"मूल कविता:\n{draft}\n\nआलोचना:\n{critique}\n\nआलोचना को ध्यान में रखते हुए एक श्रेष्ठ संस्करण लिखें।"
        final = self._generate("आप एक मास्टर कवि हैं जो अपनी गलतियों को सुधारता है।", refine_prompt)
        return f"--- DRAFT ---\n{draft}\n\n--- CRITIQUE ---\n{critique}\n\n--- FINAL ---\n{final}"

    def constraint_based(self, topic):
        user = (
            f"विषय: '{topic}'\n"
            "नियम:\n"
            "1. कविता में ठीक 4 पंक्तियां (lines) होनी चाहिए。\n"
            "2. 'आसमान' शब्द का प्रयोग वर्जित है。\n"
            "3. अंतिम पंक्ति 'कहानी' शब्द पर खत्म होनी चाहिए।"
        )
        return self._generate("आप नियमों का सख्ती से पालन करने वाले कवि हैं।", user)

    def temp_experiment(self, topic):
        low_temp = self._generate("आप एक कवि हैं।", f"विषय: '{topic}'", temperature=0.2)
        high_temp = self._generate("आप एक कवि हैं।", f"विषय: '{topic}'", temperature=1.2)
        return f"--- TEMP 0.2 (Predictable) ---\n{low_temp}\n\n--- TEMP 1.2 (Creative/Chaotic) ---\n{high_temp}"

    def persona_based(self, topic):
        sys = "आप 19वीं सदी के एक उदास, दार्शनिक कवि हैं जो पुरानी हिंदी (तद्भव/तत्सम बहुल) में लिखते हैं।"
        return self._generate(sys, f"इस विषय पर अपने विचार प्रकट करें: '{topic}'")

    def prompt_variants(self, topic):
        variant_a = self._generate("कवि बनो।", f"{topic} पर लिखो।")
        variant_b = self._generate(
            "आप साहित्य अकादमी पुरस्कार विजेता हैं। आपकी भाषा हृदय को छू लेने वाली और प्रतीकात्मक है।", 
            f"कृपया '{topic}' विषय पर एक मर्मस्पर्शी रचना प्रस्तुत करें।"
        )
        return f"--- BASIC PROMPT ---\n{variant_a}\n\n--- ENGINEERED PROMPT ---\n{variant_b}"

    def multi_agent(self, topic):
        sys_a = "आप 'कवि A' हैं। आप बहुत ही शांत और प्रकृति-प्रेमी हैं। विषय पर केवल पहली 4 पंक्तियां (Stanza 1) लिखें।"
        stanza_1 = self._generate(sys_a, f"विषय: '{topic}'")
        
        sys_b = "आप 'कवि B' हैं। आपका स्वभाव उग्र और क्रांतिकारी है। 'कवि A' की कविता को आगे बढ़ाते हुए अगली 4 पंक्तियां (Stanza 2) लिखें।"
        stanza_2 = self._generate(sys_b, f"कवि A ने यह लिखा है:\n{stanza_1}\n\nअब आप इसे अपने विद्रोही अंदाज में पूरा करें।")
        
        return f"--- STANZ 1 (Calm Agent) ---\n{stanza_1}\n\n--- STANZA 2 (Fiery Agent) ---\n{stanza_2}"

    def auto_eval(self, topic, generated_poem, poet_name, reference_poems, num_evals=5):
        ollama_client = ollama.Client(host='http://127.0.0.1:11434')
        self.logger.info("   [Evaluation] Starting Dual-LLM evaluation via Ollama...")

        system_instruction = (
            "You are an expert Hindi literary critic and NLP evaluation judge. "
            "Your task is to evaluate a generated Hindi poem. "
            "You must output ONLY a valid JSON object. Do not include markdown formatting, explanations, or introductory text."
        )
        
        evaluation_prompt = (
            f"Topic: {topic}\n"
            f"Target Poet Style: {poet_name}\n\n"
            f"--- REFERENCE POEMS BY {poet_name} ---\n"
            f"{reference_poems}\n\n"
            f"--- GENERATED POEM TO EVALUATE ---\n"
            f"{generated_poem}\n\n"
            "Evaluate the generated poem on a scale of 1 to 10 for the following metrics:\n"
            "1. 'fluency': Language fluency, correct Hindi grammar, and natural rhythm.\n"
            "2. 'coherence': Logical flow, structural integrity, and how well the stanzas connect.\n"
            "3. 'relevance': How accurately it addresses the exact Topic.\n"
            "4. 'creativity': Originality of metaphors, vivid imagery, and avoiding cliches.\n"
            "5. 'style_similarity': How closely the vocabulary, tone, and sentence structure match the Reference Poems provided above.\n\n"
            "Return EXACTLY this JSON format and nothing else:\n"
            '{"fluency": 0, "coherence": 0, "relevance": 0, "creativity": 0, "style_similarity": 0}'
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
                        messages=[
                            {'role': 'system', 'content': system_instruction},
                            {'role': 'user', 'content': evaluation_prompt}
                        ], 
                        options={'temperature': 0.7, 'num_ctx': 4096},
                        keep_alive=0 
                    )
                    clean_text = clean_json(response['message']['content'])
                    parsed = json.loads(clean_text)
                    for m in metrics:
                        if m in parsed:
                            raw_scores[m].append(float(parsed[m]))
                except Exception as e:
                    self.logger.error(f"⚠️ Eval Error on {model_name} (Attempt {i+1}): {e}")
            
            aggregated = {}
            for m in metrics:
                vals = raw_scores[m]
                if len(vals) > 1:
                    aggregated[m] = {"mean": round(statistics.mean(vals), 2), "std_dev": round(statistics.stdev(vals), 2)}
                elif len(vals) == 1:
                    aggregated[m] = {"mean": round(vals[0], 2), "std_dev": 0.0}
                else:
                    aggregated[m] = {"error": "Failed all 5 attempts"}
            final_scores[dict_key] = aggregated

        return json.dumps(final_scores, indent=2, ensure_ascii=False)

    def run_all(self, topics):
        experiments = {
            "Zero-Shot": self.zero_shot,
            "Few-Shot": self.few_shot,
            "RAG Style (Best Match)": self.rag_style_conditioned,
            "Plan-Then-Generate": self.plan_then_generate,
            "Self-Critique": self.self_critique,
            "Constraints": self.constraint_based,
            "Temperature (0.2 vs 1.2)": self.temp_experiment,
            "Persona (19th Century)": self.persona_based,
            "Prompt Variants": self.prompt_variants,
            "Multi-Agent": self.multi_agent
        }

        output_dir = "SarvamM_Output"
        os.makedirs(output_dir, exist_ok=True)
        self.logger.info(f"📂 Execution started. Saving to '{output_dir}/'")
        
        for topic in topics:
            self.logger.info(f"\n{'='*50}\n🌟 STARTING TOPIC: {topic}\n{'='*50}")
            best_poet_slug = self.find_best_poet(topic)
            safe_filename = re.sub(r'[^\w\s-]', '', topic).strip().replace(' ', '_')
            file_path = os.path.join(output_dir, f"{safe_filename}.txt")
            
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(f"🌟 TOPIC: {topic}\n🎯 BEST POET MATCH: '{best_poet_slug}'\n{'='*50}\n\n")
                
                for exp_name, exp_func in experiments.items():
                    self.logger.info(f"🧪 Running Experiment: {exp_name}")
                    f.write(f"🧪 EXPERIMENT: {exp_name}\n{'-'*50}\n")
                    
                    try:
                        query_vector = self.embedder.encode([f"कविता: {topic}"], convert_to_tensor=False)
                        res = self.collection.query(query_embeddings=query_vector, n_results=3, where={"poet_slug": best_poet_slug})
                        
                        reference_context = ""
                        if res['documents'] and res['documents'][0]:
                            for i, text in enumerate(res['documents'][0]):
                                reference_context += f"Reference {i+1}:\n{text[:300]}...\n"

                        poem = exp_func(topic, style=best_poet_slug) if exp_name in ["RAG Style (Best Match)", "Few-Shot"] else exp_func(topic)
                        
                        eval_scores_json = self.auto_eval(topic, poem, best_poet_slug, reference_context)
                        
                        f.write(f"📜 GENERATED POEM:\n{poem}\n\n🧠 DUAL-AI EVALUATION (JSON):\n{eval_scores_json}\n\n{'='*50}\n\n")
                    except Exception as e:
                        self.logger.error(f"❌ FAILED! Error in {exp_name}: {e}")
                        f.write(f"❌ ERROR GENERATING POEM: {str(e)}\n\n{'='*50}\n\n")
            
            self.logger.info(f"💾 Completed topic. Saved to: {file_path}")

        self.logger.info("✅ All experiments complete!")

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

# Note: llm replaces model
experimenter = PoetryExperimenter(llm, tokenizer, collection, embedder, logger)
experimenter.run_all(TEST_TOPICS)