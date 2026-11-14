from pathlib import Path # convinient way of working with path files ( work on Windows, macOS/Linux the same way )
from fastapi import FastAPI, HTTPException # main class of framework, HTTPException enables to use HTTP error code
from fastapi.responses import FileResponse # response which is send to a browser ( index.html )
from pydantic import BaseModel # base class to describe shape of data, used for automatic validation
from rag import DescriptionRAG # class with logic / core of pipeline
import threading # start module to work with threads

HERE = Path(__file__).parent # __file__ is path to actual file ( server.py ), very cool because .parent treats these files on the same level even if they are actually not in the project folder 
app = FastAPI(title="Test") # creates application 
rag = DescriptionRAG() # creates system object when server is launched ( __init__ invoked )


class DescriptionIn(BaseModel): # defines JSON standards for requests for description
    description: str


class QuestionIn(BaseModel): # here the same thing for question
    question: str


@app.get("/") # decorator defines what kind of requests are handled from this address
def index():
    return FileResponse(HERE / "index.html") # connects elements in Path


lock = threading.Lock() # FastAPI allows for mutiple thread working at the same time, to work with only one we initialize lock


@app.post("/description") # handles POST requests
def set_description(body: DescriptionIn): # converts JSON from request into object of this class ( after validation )
    if not body.description.strip(): # if there's no content provided after removing whitespaces 
        raise HTTPException(status_code=400, detail="Empty description")
    with lock: # takes it for the time of block execution ( once at a time )
        status = rag.set_description(body.description)
    return {"status": status, "mode": "retrieval" if rag.use_retrieval  else "full-context"} # returns FastAPI dictionary and turns it into JSON format ( field "mode" is dependent )


@app.post("/ask") # whole pipeline is being invoked here ( mean is calculated in the same block which fits this particular question )
def ask(body: QuestionIn):
    with lock:
        res = rag.ask(body.question)
        averages = rag.session_averages()
    return { # builds JSON answer with fields of Result object
        "answer": res.answer,
        "mode": res.mode,
        "reason": res.reason,
        "scores": res.scores,
        "sources": [{"similarity": sim, "text": text} for sim, text in res.sources], # data type here is a tuple, but since JSON has no tuples there has to be list comprehension performed ( dictionary )
        "stats": res.stats,
        "averages": averages,
    }