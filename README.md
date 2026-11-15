Recommended Python version: 3.12 ( used in this project )




HOW TO INSTALL?

1. Open terminal from the level of main.py. 

2. Create virtual environment:
python -m venv venv

3. Activate ( after succes it should show (venv). at the beginning ):
venv\Scripts\activate <- Windows
source venv/bin/activate <- macOS / Linux

4. In terminal use this command ( might take a while ): 
pip install -r requirements.txt

5. Then proceed with this line:
uvicorn server:app

7. Then open address:
http://127.0.0.1:8000




FOR EXAMPLES OF HOW THIS PROGRAM WORKS PLEASE PROCEED TO THE SCREENSHOTS FOLDER




INFORMATION ABOUT THIS PROJECT

This is a simple english project made for fun using RAG mechanism to generate answers based off provided knowledge along with three validation gates build with help of Claude :D. How does it work? First of all program  inserts given description, then automatically decides whether to choose retrieval mode ( long descrpition -> embeddings, semantic search) or full-context ( short description -> whole text in prompt ). After generating answer, it goes thorugh few verification gates which check different requirments. Additioanlly LLM itself checks if the context actually supports the given answer. User recieves answer as well as statistics ( latency, tokens, mode etc. ). What I find really cool in this project is that it has many verification gates with its own tresholds, which makes the answer more valuable and higher quality. The model doesn't just send whatever has been generated but it's actually being checked to meet requriements. One of the gates is basically a process of self-judgement by model while comparing it with given context. Moreover another interesting part is the structure of the prompt, which is kinda splitted into two seperate prompts - one prompt serves as usual prompt with rules, instruction etc. and the second one is also instruction but closely related to the mechanism of self-judgement. When model finishes its answer and it passes every gate, then VERIFY_PROMPT comes in to instruct model what it's supposed to do with finished answer. These two prompt templates have very similiar strucutre but serve with diferrent purpose. 




USED TECHNOLOGIES/TOOLS:

FastAPI - high-performance Python framework to enable connection between server and code.

Python - most of code is written in Python since it's one of the easiest programming languages and very often used in AI/ML field.

LangChain - framework which provides with ready to use wrap-ups for embedding models, vector databases, text splitting etc ( it supports easy integration / configuration ).

ChromaDB - vector database which stores vector embeddings ( numerical representation of text )

Hugging Face - used to import embedding model, generative model necessary classes to initialize them.

Google Flan T5 - sequence to sequence model from sentence transformers library ( encoder-decoder) which is capable of generating ( mainly ) english text.

Multilingual E5 - embedding model, converts text into vectors while reamining their semantic meaning.




NOTES MADE DURING CREATING PROJECT:

EOS - End Of Sequence, special token which means 'end of text'. For T5 model it looks like </s>. Model generates tokens one after another, at some point it chooses EOS token. It signalises to stop, otherwise model wouln't know when to stop and would add text untill reaching max_new_tokens limit. However it is not counted within the lentgh of text. 

VERIFY_PROMPT - second prompt which serves as answer verification ( instead of generating another one ). Model adds context, question and answer, then judges it based off given context with 'yes' or 'no'. These parameters are then compared with tresholf of MIN_VERIFY ( prevents from making up answers / hallucinating ).

MIN_GEN_CONF - geometric mean of probabilities, each generated token has its probability ( 0-1 scale ) with which model decided to choose it. To analyse the whole answer, they have to be summed up. It's much more strict than usual average mean because one token with low confidence will strongly affect overall result ( it enables to protect from overlooking weaker tokens? ).

Indexing - enables fast searching thorugh database by assigning them with identificator. In this code description is splittied into chunks, converted each one into vector and saved in databse.

List/Dict comprehension - shortcut for creating a list / dictionary in one line.

CPU/GPU - processrors on which model calculations are performed. CPU are great for diversity of tasks, but struggle with huge amount of repetitive ones. GPU handles performing many same tasks ( great for neuron models when it comes to multipltying matrixes ).

__init__ - construcor, special method which launches automatically when class object is being created. Python creates an empty object for each class initialisation and invokes on him __init__, which sets beginning state ( here loading models, launching splitter, tokenizer etc ).

Tensor - multidimensional array, basic data type for PyTorch, they are fast in calculations and are capable of tracking gradients.

Greedy Generation - for each generation counts probability for each token in dictionary. It chooses only the best token ( the highest probability ), same question -> same answer. 

Sampling - randomly picks token appropiately to probabilities, so the answers are more diverse, but less predictable.

Beam Search - tracks few best variants at the same time, at the end chooses the best whole string.

Parser - program which analizes text according to rules and changes it into organized structure. 

Cleaning the reference - it's a label poinitng to an object in memory ( if it points to None, Python automatically frees memory by garbage collector ). In code it plays a role of flag which states whether index exists or there's no index ( if description is short there will be no indexing ).

Sequence to Sequence (seq2seq) - type of model which converts one string of characters into other string ( both can have different lengths ). Input is a sequence and output is sequence as well - great for translation purposes or answering. 

Encoder - first part of seq2seq, it reads all input and turns it into inner number representation ( understanding prompt ), it sees the whole thing so context remains. 

Decoder - second part of seq2seq models, generates answer token after token. In each step it looks at representation made by encoder and what it generated itself, then it chooses next token. 

Token - the smallest unit of text, which LLM's work on. It's neither a word nor letter - part of text chosen by tokenizer. Short words are usually represented as one token, meanwhile longer or rare words are splitted into few smaller units. Each token has its identification number in model's dictionary. Limits are measured with them.

Chunk - excerpt of document, created from splitting long text into smaller pieces. These are later saved in vector database. Created by TextSplitter.

Argparse - Python standard library for arguments management from command line. Thanks to it flags can be launched with no modifications in the program itself ( such as --debug, which additionally checks corectness of args and eventually generates --help automatically ).

Why e5 needs prefixes? - this model has been trained on pairs ( question -> answer ) with prefixes throughout the whole training. Prefix is kinda like label, it defines whether its a question or a document to be found. Then it creates vector in such way that these pairs are close to each other. 

PyTorch - library used for performing calculations on tensors ( neruon networks run on them ). It provides to calculating on GPU, automatic counting of gradients as well as tensors themselves. Moreover transformers library is actually build on PyTorch.

LangChain - framework used for convinient building of applications based on LLM's. It provides with ready to use wrap-ups for embedding models, vector databases, text splitting etc. By using LangChain it's easy to switch some tools with no need for huge changes it the code itself. It's kinda like automized setup with ready to use tools ( no need to build everything from scratch ).

Overlap - when the limit of chunk is reached at the end of document, it might cut off in the middle of a sentence. Overlap prevents such situations by defining particular amount of supply - information from the end of document remains in one part.

Sematnic search - based on cosinus distance, enables models to percieve meaning rather than just occurrence of a specific word. It enables to search for chunks with similiar meaning providing better 'understanding' of language structure.

Why values are not settled right in the class? - each object needs to have its own state, if it would be assigned explicitly then for every object of this class, the value woul be the same. Also some values are depnedant and might change impacted by actual environemnt.

Logit - raw result given by model before changing it into probability. Softmax is then converting these into scale of 0-1. That's also why normalize_logits=True, because even if logits themselves might already hint with probability, they might be negative as well.

*Args - saves all postiion arguments into tuple ( invoked with * will unpack ).

**Kwargs - saves all arguments into dictionary ( invoked with ** will unpack ).

Why normalizing vector to the lentgh of 1? - embeddings have destination as well as length and meaning of a text holds onto destination. So after normalization results have usual stable scale so every treshold works with the samle values ( length of a vector does not affect comparison ).

Flan T5 - small Google model of type text-to-text, which means that every request is saved as input text and output text. Flan means additional tunning for many differnet types of tasks described with instructions. It's build on encoder-decoder architecture 

Multilingual E5 - Microsoft embedding model, converts text into vector with 768 dimensions, it calculates relations bewteen chunks ( if two chunks are closely related then their distance will be smaller ).

Chunk probability - for each chunk there is probability of it being accurate to given context and answer.

Confidence of generation - model writes token after token, so that's why tokens can have its own probabilties as well. This measurment defines how confident model was when writing particular word / sentence ( it depends whether word was short or long so it could be a half of word, word or even small sentence possibly? ).

Framework - it's a ready structure or set of tools. These are fundaments on which real application programms can be build. It gives ready to use, checked elements to work with by only modyfing specific areas with no need to build from scratch.

Model's parameters - weights ( numbers ) inside model, which it learned during training porcess. They kinda store basic knowledge of a model.

Hiperparameters - these are settings to be set up manually, can be decided upon such as TOP_K or CHUNK_SIZE - models adapt to them and work appropiately to these configurations.

CLI - Command Line Interface, terminal which inserts command lines.

Venv - Virtual Environment, isolated folder with its own copy of Python and libraries assigned for one project. It's used becasue some progarms work on different versions of tools and libraries and some other will also work on other versions - in one Python there's only possibility to have one version of library / tool - the purpose of venv is to enable multitiple isolated environemnts for each project.

Why list? - this data type is organized and can dynamically change, so there are modifications allowed.

Why tuple? - organized group of data which is static and can't be changes after initialisation.

config.py - conveinient file which usually stores all settings for models, chunk sizes, tresholds as well as pormpts, basically a place for configuration and set up settings.

Tokenization to PyTorch tensor - closely related to how machines percieve semantic meaning of text through numbers. Text is converted into list of numbers ( which represent and remain meaning of text ), then it's being returned in the form of tensor ( PyTorch data requirement ).

Why max_length=1024 if PROMPT_TOKEN_BUDGET=512? - first is estimated  project budget, second one is safety limit. In moment of tokenization ( not the description itself but also prompt template, user's question )the first limit might get crossed so instead of cutiing it off and losing context the limit kinda expands but with cost of losing precision tho!

Truncation - cutting off input if it's longer than defined limit. If truncation=True, tokenizer keeps everything that fit into the limit and rejects the rest from the end.

Gradient - information in which direction and with what magnitude change each weight of model in order to decrease loss function. 

Backward Propagation - process which enables model to 'go back' after training, model makes a prediction, measures loss, counts gradients and eventually corrects weights' values.

NaN - Not a Number, special value which cannot be calculated ( like 0/0 ). It's truly dangerous because its invasious - each action with NaN generates NaN, and they always identify as False values.

Why float? - it's convinient data type which can be easily rounded and compared. On the other hand tensor is like a wrap-up around this number which cannot be converted. 

Endpoint - specific address in server which performs particular actions. It's connection between address ( path ) and HTTP method ( which defines what kind of operation it is ):
GET/ - retunrs index.html. 
POST/description - takes description and builds index.
POST/ask - rakes question and returns answer.

API - Application Programmin Interface, set of endpoints which enable communication between servers and programms, by connecting and integrating tools.

/docs - automatically generated content by FastAPI when it analyzes code ( can be tested on ). Imporant to note: if server would go public, then anyone can modify /docs, so it's crucial to secure it beforehand. 

/redoc - the same documentation but more 'formal' with no possibility to test.

/openapi.json - raw API description in JSON format, which creates /redocs as well as /doc.

Swagger UI - interface placed in /docs address, which consists of list of all endpoints, what kind of data each one of them accepts and what possible errors might return. 

Logs - order list of all actions performed by program, usually used when errors occur in order to investigate cause or to monitor / track. These are list of all occurrences while statistics are counted numbers - they both describe how system works:
INFO - normal actions such as POST /ask 200 OK 
WARNING - odd behaviour, but program continues running
ERROR - crash out of program such as code erro.

JSON - Java Script Object Notation, text format to wirte and save data. Normal text with restrictive syntax which consists of:
Object {...} - set of pairs '"klucz":wartość', basically dictionary in Python.
Array [...] - organized list of values, list in Python.
Values - text in "text", numbers (0.9), boolean values true / false, null other nested obejcts.
Important to note: JSON doesn't have tuples, that's why they are usually converted into lists.

Why use JSON? - server is written in Python, while website in JavaScript. Two different programming languages with different type of data, so it's impossible to exchange  objects between these explicitly. JSON is a special layer which is defined as common format undesrtood by both sides. 

Serialization - turning object into JSON text.

Deserialization - turning JSON format into object. 

Pydantic - Python data validation library which processes data with core logic written in Rust.




HELPFUL COMMANDS/FILES:

Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned - if venv acitavte won't work because of PowerShell ( Windows blocked activation script, even tho the venv was created succesfully )

pip freeze > requirements.txt - writes all necessary libraries along with specific versions used in the project ( important to note: first activate venv and JUST THEN run this command! ( learning on mistakes ) )

.gitignore - file used when project is supposed to be shared, it informs Git which files should NOT be tracked or uploaded.

deactivate - to deactivate venv.

pycache - folder made by Python automatically, consits of compiled version of code saved in .pyc files.  However Python doesn't execute the code directly. It transforms it into bytecode ( simplified intermediate ( pośrednia ) form ).

python main.py --debug - terminal version of this program