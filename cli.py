import argparse # user firendly command-line interface 


from rag import DescriptionRAG
from result import Result


def read_multiline(prompt: str) -> str: # function takes a message and returns entered text as one string 
    print(prompt + " (finish with two empty lines)")
    lines = []
    blank = 0 # counter for blank lines
    while True: # 'infinite' loop
        try:
            line = input() # each iteration input() awaits for user's input
        except EOFError: # Ctrl+D / Ctrl+Z is handled by finishing rather than unexpected crash
            break
        if not line.strip(): # removes whitespaces from input and if there's nothing left
            blank += 1
            if blank == 2: # it breaks out of the loop after the second empty line in a row
                break
        else:
            blank = 0
        lines.append(line) # each line is added to list ( empty ones too, so paragraphs stay separated )
    return "\n".join(lines).strip() # connects each line together into one text ( .strip() removes the empty lines )


def print_stats(rag: DescriptionRAG, res: Result) -> None: # function recieves RAG object and result of last question  
    s = res.stats # 's' stands for dictionary stats )
    print(
        f"  mode: {res.mode} | latency: {s['latency']:.2f}s "
        f"(retrieval {s['t_retrieval']:.2f}s, generation {s['t_generation']:.2f}s, "
        f"verification {s['t_verification']:.2f}s)"
    )
    print(
        f"  tokens this question: {s['input_tokens']} in / {s['output_tokens']} out "
        f"= {s['total_tokens']} total "
        f"(generation {s['gen_in']} in + {s['gen_out']} out, verification {s['verify_in']} in)"
    )
    avg = rag.session_averages() # loads averages from conversation history
    print(
        f"  average over {avg['n']} question(s): {avg['input_tokens']:.0f} in, "
        f"{avg['output_tokens']:.1f} out, {avg['total_tokens']:.0f} total per question, "
        f"{avg['latency']:.2f}s latency"
    )
    if res.sources: # sources are printed only if list is not empty 
        print("  sources (most reliable first):") # result from descening sorting
        for i, (sim, text) in enumerate(res.sources, 1): # starts from one, unpacks tuple ( 'podobieństwo, tekst' ) explicitly in loop
            flat = " ".join(text.split()) # with no arg provided it splits by whitespaces, then connects with spaces 
            snippet = flat[:90] + ("..." if len(flat) > 90 else "") # cuts off to 90 characters and adds space if something was cut off
            label = f"similarity {sim:.3f}" if sim is not None else "entire description" # in retrieval mode it will display probability, otherwise in full-context mode it's set to None ( no sematnic search )
            star = " *best*" if i == 1 and sim is not None else "" # assigns star for first ( the best ) part ONLY if probability exists ( nothing to be compared in full-context mode ) 
            print(f'   {i}. {label}{star}: "{snippet}"')


def main():
    parser = argparse.ArgumentParser() # creates an object of parser 
    parser.add_argument("--debug", action="store_true", help="show validation scores and refusal reasons") # defines optional flag which doesn't take values ( if args.debug then True, otherwise False )
    args = parser.parse_args() # reads args from sys.argv and saves them in args object 

    print("Loading models...")
    rag = DescriptionRAG() # creating object of system ( here __init__ is initialized - loading embeddigs, tokenizer, models )

    print(rag.set_description(read_multiline("\nPaste your description:")))
    print("Ask questions ('new' = new description, 'exit' = quit).\n")

    while True: # 'infinite' loop
        try:
            q = input("You: ").strip() # awaits user's input, then removes whitespaces with .strip()
        except (EOFError, KeyboardInterrupt): # tuple for two specific exceptions ( Ctl + C, Ctrl + Z )
            print() 
            break # breaks out of the loop so main ends with no crash out
        if not q: # if no question provided ( only Enter )
            continue # it skips the rest of loop and asks again ( prevents from loading empty text into model )
        if q.lower() in {"exit", "quit"}:
            break
        if q.lower() == "new":
            print(rag.set_description(read_multiline("\nPaste your new description:")))
            continue

        res = rag.ask(q) # launches the whole pipeline ( searching, generating, gates )
        if res.answer: # if answer is not empty
            print(f"Bot: {res.answer}")
        else: # if any gate rejected answer 
            print("Bot: I'm not sure, so I'd rather not answer.")
        if args.debug: # works only with --debug flag
            print(f"     [reason={res.reason or 'ok'} scores={res.scores}]") # dictionary with similarity, confidence of generation and verification ( contains only values which were calculated before potential rejection )
        if res.stats:
            print_stats(rag, res)
        print()