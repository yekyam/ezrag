import os
import logging
import argparse
import textwrap
import readline
from pathlib import Path
from multiprocessing.dummy import Pool


import faiss
import ollama
import pymupdf4llm
import numpy as np
from rich.console import Console

logging.basicConfig(level=logging.WARN)

console = Console()

EMBEDDING_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.2:1b"

def check_model_available(model):
    """Checks to see if the model is installed locally through ollama.

    :param model: the name of the model to check.
    :return: true if the model is installed locally; else false.
    """
    try:
        ollama.show(model)
        return True
    except Exception:
        return False

def turn_string_into_chunks(source: str, string: str, size_of_chunks: int = 1000) -> list[list[str]]:
    """Turns a string into a list of chunks.

    :param string: the string
    :param size_of_chunks: the size of the chunks
    :return: the chunks
    """
    return [[str(s), str(source)] for s in textwrap.wrap(string, size_of_chunks)]

def add_string_to_store(store, string):
    """Creates an embedding from a string, and stores it in the current vector store.

    :param store: the store. if store is None, then just returns the embedding from the string
    :param string: the string to embed and add to the store
    :return: either just the string's embeddings or the updated store
    """
    response = ollama.embed(EMBEDDING_MODEL, string)
    if store is None:
        return response.embeddings
    return store + response.embeddings

def get_text_from_pdf(filepath) -> str:
    """Extracts the text from a file as a string.

    :param filepath: the file path
    :return: the content of the file
    """
    return pymupdf4llm.to_markdown(filepath)

def create_db_from_items(items: list[str]) -> list[list[str]]:
    """Creates a content "database" from a list of sources. A content database tracks a string and it's source.

    :param items: A list of sources. A source can either be a file or a directory containing files.
    """

    def _create_db_from_file(file_path):
        db = []
        logging.info(f"getting doc...: {file_path}")
        text = ""

        try:
            text = get_text_from_pdf(file_path)
        except Exception:
            logging.info("couldn't get text using pymupdf4llm; trying to read normally")
            try:
                with open(file_path, "r") as f:
                    text = f.read()
            except Exception as e:
                logging.warning(f"couldn't open file {file_path}! encountered error: {e}")
                return []
        logging.info("...got doc")
        logging.info("chunking...")
        chunks = turn_string_into_chunks(file_path, text)
        logging.info("...done chunking")
        db.extend(chunks)
        return db

    def _create_db_from_item(item):

        if os.path.isfile(item):
            return _create_db_from_file(item)

        db = []
        files = []
        content_path = Path(item)
        for (_, _, filenames) in os.walk(content_path):
            files = filenames
            break

        for file in files:
            _db = _create_db_from_file(content_path / file)
            db.extend(_db)
        return db

    db = []

    with console.status("[bold green]loading sources..."):
        with Pool() as pool:
            logging.info("trying to create docs...")
            results = pool.map(_create_db_from_item, items)
            logging.info("...created docs")

        for result in results:
            db.extend(result)

        logging.info(db)
    return db

def main():
    logging.info("starting...")


    parser = argparse.ArgumentParser(description="Initalizes a RAG system and allows an LLM chat using local LLMs through ollama.")
    parser.add_argument("sources", nargs="+", help="A source is either a file or a directory containing those files.")

    args = parser.parse_args()

    if not check_model_available(EMBEDDING_MODEL):
        logging.fatal(f"couldn't get model {EMBEDDING_MODEL}! quitting... :(")
        quit()
    logging.info("embedding model available!")

    if not check_model_available(CHAT_MODEL):
        logging.fatal(f"couldn't get model {CHAT_MODEL}! quitting... :(")
        quit()
    logging.info("chat model available!")


    content_db = create_db_from_items(args.sources)
    logging.info(content_db)
    logging.info(len(content_db))
    store = None

    with console.status("[bold green]creating embeddings..."):
        for s in content_db:
            logging.info(f"adding string to store...:{s[0]}")
            logging.info(f"{{file name: {s[1]}; content: {s[0]}}}")
            store = add_string_to_store(store, f"{s[1]}:{s[0]}")
            logging.info("...added string")
    # vector database
    # store = add_string_to_store(None, "bobby sucks")
    # store = add_string_to_store(store, "many say that the goat is jimmy")
    # store = add_string_to_store(store, "candice is on the suns")
    # store = add_string_to_store(store, "dont ddrink and drive")
    # store = add_string_to_store(store, "just shut up and dribble")
    # store = add_string_to_store(store, "jimmy is the goat!")
    # print(np.array(store).shape)

    store = np.array(store)

    dimensions = store.shape[1]

    index = None
    with console.status("[bold green]creating index..."):
        index = faiss.IndexFlatL2(dimensions)
        index.add(store)

    while True:
        logging.info(f"db size = {len(content_db)}; dimensionality: {dimensions}")
        try:
            search = console.input("[bold blue]: ")

            if len(search) == 0:
                continue
        except KeyboardInterrupt:
            quit()
        response = ollama.embed(EMBEDDING_MODEL, search)

        query_vector = np.array(response.embeddings)

        D, I = index.search(query_vector, 5)

        logging.info(D)
        logging.info(I)
        logging.info(I[0])

        indices = I[0]

        result = [content_db[i] for i in indices]

        logging.info(f"{search}:{result}")

        info = [pair[0] for pair in result]
        files = set(pair[1] for pair in result)

        with console.status("[bold green]thinking..."):
            final_response = ollama.chat(CHAT_MODEL, messages=[
                {
                    "role": "system",
                    "content": f"the user asked to find {search}, the information retrieved was {info}, and the information was retrieved from these files: {files}. summarize it for them based ONLY on the result. be sure to tell them the name of the files. add a little sass too; chastise the user for being too lazy to read the text."
                },
            ])

        console.print(final_response.message.content, style="bold yellow")

if __name__ == "__main__":
    main()
