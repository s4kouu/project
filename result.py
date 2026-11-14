from dataclasses import dataclass, field # used for data structures ( answer result with fields such as answer, confidence etc. )


@dataclass # decorator which automtically generates build in python objects ( such as __init__, __repr__ and __eq__ ) based off declared fields
class Result:
    answer: str | None # string answer or None ( no default value, gotta assign when object is being created )         
    mode: str = "" # field defining answer mode ( full or retrieval )              
    reason: str = "" # field which defines reason ( e.g for NO_ANSWER ) and which gate rejected it          
    scores: dict = field(default_factory=dict) # creates empty dictionary for each object ( dataclass frobids explicit declaration {} of dicrionaries )
    sources: list = field(default_factory=list) # list of used excerpts from which answer was created
    stats: dict = field(default_factory=dict) # statistics ( such as response time, number of tokens )
# important rule: fields with default values have to go last