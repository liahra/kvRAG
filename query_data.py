import argparse
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama

from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Svar på spørsmålet basert kun på følgende kontekst og bruk norsk språk:

{context}

---

Svar på spørsmålet ovenfor på norsk. Gi et detaljert svar og forklaring: {question}
"""


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="Spørsmålet eller forespørselen.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)

# Funksjon som tar én parameter query_text, som er en streng.
# Representerer spørsmålet eller forespørselen brukeren vil stille.
def query_rag(query_text: str):
    # Forebereder databasen.
    embedding_function = get_embedding_function() # Konverterer tekst til vektor.
    # Gjør det mulig å søke i databasen basert på likhet i embeddingen.
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Søker i databasen.
    # Finner de 3 mest relevante dokumentene basert på likhet med query_text
    results = db.similarity_search_with_score(query_text, k=3)

    # Kombinerer innholdet fra de returnerte dokumentene til en enkelt kontekststreng.
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    # Forbereder tekst som sendes til språkmodell
    prompt = prompt_template.format(context=context_text, question=query_text)
    # print(prompt)

    model = Ollama(model="mistral")
    # Sender prompten til modellen og får tilabke en respons
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Respons: {response_text}\nKilder: {sources}"
    print(formatted_response)
    return response_text


if __name__ == "__main__":
    main()