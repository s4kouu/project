# multilingual embedding model ( prefixes recomended for better quality search e.g "query: ", "passage: ") 
EMBEDDING_MODEL = "intfloat/multilingual-e5-base" # or "intfloat/e5-large-v2"
# sequence to sequence instruct model - small, mainly english-targeted and its window limit is 512 tokens
LLM_MODEL = "google/flan-t5-large" # or "google/flan-t5-large"


PROMPT_TOKEN_BUDGET = 512 # maximum length of a whole prompt in tokens ( adapted to flan-T5 capabilities )
QUESTION_RESERVE = 64 # token allowance ( przydzial ) for asked question ( along with instructions / prompt template )    
CHUNK_SIZE = 300 # length of excerpt ( fragment )             
CHUNK_OVERLAP = 80 # overlap ( nałożenie ) for next units of text ( to prevent cutting of information at the end of chunk )
TOP_K = 3 # number of the most similiar parts ending up in the context


MIN_RETRIEVAL_SIM = 0.80 # minimum similarity of question to found excerpts ( treshold ( próg ) which defines the minimum similarity - if given part crosses it, then it's being actually used / considered in answer. 
MIN_GEN_CONF = 0.60 # treshold of minimum confidence ( if model is not sure its answer then it will be rejected by treshold )        
MIN_VERIFY = 0.75 # treshold of verification, model recieves context, question and answer and judges it with 'yes' or 'no'.          
NO_ANSWER = "I don't know based on the description." # fallback situation ( settled in prompt )


# prompt template
ANSWER_PROMPT = (
    "Answer the question using the context below. "
    # f"If the answer is not in the context, say: {NO_ANSWER}\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)


# copmares probability of 'yes' and 'no' and based off it, calculates MIN_VERIFY
VERIFY_PROMPT = (
    "Context:\n{context}\n\n"
    "Question: {question}\n"
    "Proposed answer: {answer}\n\n"
    "Is the proposed answer supported by the context? Answer yes or no."
)