import time # measure response time
import uuid # generates unique ID ( name of collection in Chroma to prevent mixing data )
import torch # PyTorch library where torch is GPU / CPU to calculate probability 
from langchain_chroma import Chroma # vector database to store embeddings ( numerical representation of text )
from langchain_text_splitters import RecursiveCharacterTextSplitter # intelligently splits text ( by not cutting in the middle of word / sentence) into smaller parts ( chunks )
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer # loads generative model of ecoder-decoder type  and its tokenizer ( many models have their own tokenizer and rarely use the versatile ones, for better quality data processing )

# every file needs its own imports
from config import (
    EMBEDDING_MODEL, LLM_MODEL,
    PROMPT_TOKEN_BUDGET, QUESTION_RESERVE, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K,
    MIN_RETRIEVAL_SIM, MIN_GEN_CONF, MIN_VERIFY,
    ANSWER_PROMPT, VERIFY_PROMPT,
)
from embeddings import E5Embeddings
from result import Result


# main system class ( __init__ is a constructor which performs during creating object of DescriptionRAG() ), it loads models and sets beginning state
class DescriptionRAG:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu" # checks which device is aviable 
        self.embeddings = E5Embeddings( # initialising embedding object, loading e5
            model_name=EMBEDDING_MODEL,
            encode_kwargs={"normalize_embeddings": True}, # normalizing vectors to a length of 1 ( probability values are set to predictable scale for MIN_RETRIEVAL_SIM )
        )
        self.tok = AutoTokenizer.from_pretrained(LLM_MODEL) # first launch loads tokenizer for generative model ( flan-t5 ), converts text into numbers ( ID tokens, needed for token limit ) 
        self.model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL).to(self.device).eval() # loads model's weights ( from_pretrained(LLM_MODEL) ), moves model to available device ( .to(self.device) ), switches model into evaulation mode ( .eval() ), e.g. turns off dropout for stable results ( conclusion standard )
        self.splitter = RecursiveCharacterTextSplitter( # creates an object for splitting chunks, defining CHUNK_SIZE and CHUNK_OVERLAP, 'separators' is a list of separators, first splitter cuts off after each double enter ( if part is too long, then single enter ), it enables chunks to be splitted naturally
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self.description = None 
        self.store = None
        self.use_retrieval = False
        self.history: list[dict] = [] # conversation history as a list of dictionaries ( e.g. question-answer )


    def _count_tokens(self, text: str) -> int: # _count_tokens is an inner method ( not supposed to be invoked outside )
        return len(self.tok(text, add_special_tokens=False).input_ids) # tokenizes text, add_special_tokens prevents from inserting special characters such as </s> ( counts only tokens of text )

    def set_description(self, description: str) -> str: # this method recieves description, decides which mode to choose and eventually builds an index 
        if self.store is not None: # if there are any remainings from previous conversation
            self.store.delete_collection() # it's being deleted
            self.store = None # cleaning up the reference 
        self.history = [] # resets conversation history by defining empty list

        self.description = description.strip() # .strip() removes any whitespaces and enters from description
        if not self.description: # empty description is not considered at all
            self.use_retrieval = False # sets to False ( none of modes )
            return "Empty description - please paste some text."
        overhead = self._count_tokens(ANSWER_PROMPT.format(context="", question="")) # counts tokens of prompt template ( instruction, words and 'build-in' NO_ANSWER )
        needed = self._count_tokens(self.description) + overhead + QUESTION_RESERVE # estimating how many tokens will be used by prompt if the whole description would be added ( description tokens and saved 64 tokens for question )
        self.use_retrieval = needed > PROMPT_TOKEN_BUDGET # if estimation crosses 512 tokens, the search ( retrieval ) mode is turned on ( not the whole description is attached to prompt, rather the most similiar parts of it )

        if not self.use_retrieval: # if the description is short, then there's no need to create embeddings
            return f"Short description ({needed} tokens needed) -> full-context mode."

        chunks = self.splitter.split_text(self.description) # splits description into smaller units ( 500 characters along with 80 of overlap ) based off configuration made by __init__ 
        self.store = Chroma( 
            collection_name=f"desc_{uuid.uuid4().hex[:8]}", # unique name of collection desc_ along with 8 random characters from UUID ( no collision of descriptions )
            embedding_function=self.embeddings, # defines embedding function made specifically for e5 which needs prefixes 
            collection_metadata={"hnsw:space": "cosine"}, # sets metric of distance for cosinus 
        )
        # important to note: there's no persist_directory so database runs only in session memory! ( intentional tho )
        self.store.add_texts(chunks) # converts every chunk into vector and saves it in database
        return f"Long description ({needed} tokens needed) -> retrieval mode, {len(chunks)} chunks." 


    def _generate(self, prompt: str) -> tuple[str, float, int, int]: # generates answers and returns tuple ( immutable but indexed? )
        inputs = self.tok(prompt, return_tensors="pt", truncation=True, max_length=1024).to(self.device) # tokenizes prompt into PyTorch tensors ( the most dimensional type ), by enabling truncation it cuts off above 1024 tokens, then it moves tensors to avaiable device ( CPU / GPU )
        with torch.no_grad(): # turns off gradient count ( during concluding there's no need for it, therefore it saves memory )
            out = self.model.generate(
                **inputs, # defined earlier it unpacks dictionary as arguments ( ** args / kwargs )
                max_new_tokens=128, # limits answer length
                do_sample=False, 
                num_beams=1, # greedy generation - only the most likely token 
                output_scores=True, # returns logits from each step of generation 
                return_dict_in_generate=True, # returns dictionary object with fields ( out.sequences, out.scores )
            )
        log_probs = self.model.compute_transition_scores( # takes every logarithm of porbability from each chosen answer 
            out.sequences, out.scores, normalize_logits=True # log-softmax for out.sequences and out.scores ( no raw logits )
        )[0] # takes the first and only element of batch
        confidence = log_probs.mean().exp().item() if len(log_probs) else 0.0 # protects from empty answer ( which would result in NaN value ), converts tensor to float
        text = self.tok.decode(out.sequences[0], skip_special_tokens=True).strip() # decodes token identificators back to text, removes tags such as <pad>, </s>
        return text, confidence, inputs["input_ids"].shape[1], len(log_probs) # returns answer, length of token input and number of generated tokens ( stats )

    def _verify(self, context: str, question: str, answer: str) -> tuple[float, int]: # checks whether answer is related to the actual context
        prompt = VERIFY_PROMPT.format(context=context, question=question, answer=answer) # build VERIFY_PROMPT and tokenizes it
        inputs = self.tok(prompt, return_tensors="pt", truncation=True, max_length=1024).to(self.device) # answer prompt and this prompt are not equal ( this prompt consists of context as well as prompt itself so it might be longer )
        start = torch.tensor([[self.model.config.decoder_start_token_id]], device=self.device) # since T5 is made of enocder ( reads the prompt ) and decoder ( generates an answer ), creating tensor [[id]] with shape of (1, 1) - 1 element of batch and one token 
        with torch.no_grad(): # encoder recieves prompt, decoder recieves first token
            logits = self.model(**inputs, decoder_input_ids=start).logits[0, -1] # recieves vector of logits for the first answer token ( how model started to shape its answer )
        # loads identificators for 'yes' and 'no' tokens, where [0] takes the first word token ( for both words it's usually one token, but if there would be more - then the first one would be the only one recieved )
        yes_id = self.tok("yes", add_special_tokens=False).input_ids[0]
        no_id = self.tok("no", add_special_tokens=False).input_ids[0]
        p_yes = torch.softmax(logits[[yes_id, no_id]], dim=0)[0].item() # chooses only two logits for 'yes' and 'no', turns it into probability summing up to 1 ( softmax? ) and compares 'yes' with 'no' 
        return p_yes, inputs["input_ids"].shape[1] # return result of verification


    def session_averages(self) -> dict:
        n = len(self.history) # number of currently asked questions ( each question assigns to self.history one dictionary of statistics )
        if n == 0: 
            return {"n": 0} # protects from dividing by zero
        keys = ["input_tokens", "output_tokens", "total_tokens", "latency"] # list of statistics to be averaged
        avg = {k: sum(h[k] for h in self.history) / n for k in keys} # dict comprehension goes thorugh keys, summs value of 'k' from every question
        avg["n"] = n # adds number of questions to result
        return avg
    

    def ask(self, question: str) -> Result:
        if not self.description: # if set_description has not been invoked yet or it's empty
            return Result(None, reason="no description provided yet") # then returns None value ( this question doesn't even reach history since it didn't go through 'finish' )

        t_start = time.perf_counter() # to measure latency / time
        stats = { # much safer to initialize with 0 values rather than nothing ( unpredictable behaviour )
            "gen_in": 0, "gen_out": 0, "verify_in": 0,
            "t_retrieval": 0.0, "t_generation": 0.0, "t_verification": 0.0,
        }
        # initializing lists for results of confidence and sources 
        scores: dict = {}
        sources: list = []

        def finish(answer, mode, reason=""): # nested function, all outputs go through this one
            stats["latency"] = time.perf_counter() - t_start # time from 'ask' till now
            stats["input_tokens"] = stats["gen_in"] + stats["verify_in"] # sum of generation and verification 
            stats["output_tokens"] = stats["gen_out"] # only sum of generation ( '_verify' only does one lap ( przebieg ) forward )
            stats["total_tokens"] = stats["input_tokens"] + stats["output_tokens"] # sum of both
            self.history.append(stats)  
            return Result(answer, mode, reason, scores, sources, stats) # shapes the final result of stats

        
        if self.use_retrieval: # retrieval mode ( long desciption )
            mode = "retrieval"
            t0 = time.perf_counter()
            hits = self.store.similarity_search_with_score(question, k=TOP_K) # converts question into vector ( with prefixes! ) and returns top 3 closest chunks
            stats["t_retrieval"] = time.perf_counter() - t0 # time of searching
            sources.extend(sorted(((1 - dist, doc.page_content) for doc, dist in hits), reverse=True)) # for each pair, a tuple is being created, Chroma returns cosinus probability 
            scores["top_similarity"] = round(sources[0][0], 3) if sources else 0.0 # sorts probability of best chunk ( rounded ), if no source it sets to 0
            if not sources or sources[0][0] < MIN_RETRIEVAL_SIM: # gate number one, efficient decision making
                return finish(None, mode, "no sufficiently relevant passage found") # if the probability of the best chunk is below 0.80 the answer is set to None, this way generative model is not being launched at all
            kept = list(sources) # creates copy of a list
            # always left with one source ( only the best chunk )
            while len(kept) > 1 and self._count_tokens( # repeates loop untill each condition is meet
                ANSWER_PROMPT.format( # building prompt from actual chunks and connecting it together 
                    context="\n---\n".join(t for _, t in kept), question=question
                )
            ) > PROMPT_TOKEN_BUDGET:
                kept.pop() # sources are sorted desc, so the last one is the weakest 
            sources[:] = kept # stats will display only used elements
            context = "\n---\n".join(text for _, text in kept) # connects text of chunks in one context 
        else: # for short desciption, context is full description
            mode = "full-context"
            sources.append((None, self.description)) # tuple with 'None' and 'opis' ( no semantic search, no probability )
            context = self.description

        t0 = time.perf_counter()
        answer, gen_conf, stats["gen_in"], stats["gen_out"] = self._generate( # inserts context along with question
            ANSWER_PROMPT.format(context=context, question=question) # returns four-element tuple ( two last values are added to 'stats' )
        )
        # saves generation time, model's confidence 
        stats["t_generation"] = time.perf_counter() - t0
        scores["generation_confidence"] = round(gen_conf, 3)


        if not answer or "don't know" in answer.lower().replace("’", "'"): # gate number two
            return finish(None, mode, "model says the answer is not in the description") # it rejects empty answer, captured by NO_ANSWER
        if gen_conf < MIN_GEN_CONF: # gate number three
            return finish(None, mode, "low generation confidence") # if average probability of tokens is below 0.60, the answer is rejected 

        t0 = time.perf_counter()
        # model judges its answer based off the same context ( result is saved along with number of tokens and time )
        p_yes, stats["verify_in"] = self._verify(context, question, answer)
        stats["t_verification"] = time.perf_counter() - t0
        scores["verification"] = round(p_yes, 3)
        if p_yes < MIN_VERIFY: # gate number three
            return finish(None, mode, f"answer not supported by the context (candidate: {answer})") # answer rejected if 'p_yes' is lower than 0.75

        return finish(answer, mode) # answer succesfully passed verification so it's sent to the user