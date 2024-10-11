import gradio as gr
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

# Funksjon som tar én parameter query_text, som er en streng.
# Representerer spørsmålet eller forespørselen brukeren vil stille.
def query_rag(message, history=None):
    # Initialiser historikk hvis den ikke er satt
    history = history or []
    
    # Forebereder databasen.
    embedding_function = get_embedding_function()  # Konverterer tekst til vektor.
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Søker i databasen.
    results = db.similarity_search_with_score(message, k=3)

    # Kombinerer innholdet fra de returnerte dokumentene til en enkelt kontekststreng.
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    # Forbereder tekst som sendes til språkmodellen
    prompt = prompt_template.format(context=context_text, question=message)

    # Kjør spørsmålet gjennom modellen
    model = Ollama(model="llama3.2")
    response_text = model.invoke(prompt)

    # Hent kilder fra de returnerte dokumentene
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"{response_text}\n\nKilder:\n" + "\n".join([f"- {source}" for source in sources if source])

    # Legg til spørsmålet og svaret i historikken
    history.append((message, formatted_response))

    # Returner historikken for chatbot og oppdatert state
    return history, history  # Én for chatbot og én for state

# Gradio UI
with gr.Blocks() as demo:
    # Legg til en tittel og beskrivelse
    gr.Markdown("# RAG-søk for produktspesifikasjoner")
    gr.Markdown("Skriv inn et spørsmål eller forespørsel, og modellen vil gi et detaljert svar basert på kontekst.")
    
    # Chatbot-komponent for chat-lignende UI
    chatbot = gr.Chatbot()
    
    # Input-felt for spørsmål
    query_text = gr.Textbox(label="Skriv ditt spørsmål her", placeholder="Hva består arter av nasjonal forvaltningsinteresse av?")
    
    # Knapp for å sende inn og klarere spørsmål
    submit_btn = gr.Button("Send inn")
    clear_btn = gr.Button("Tøm")

    # Skjult tilstand for å holde samtalehistorikk
    state = gr.State([])  # Initialiserer med en tom liste for historikk

    # Handlinger for knappene
    submit_btn.click(fn=query_rag, inputs=[query_text, state], outputs=[chatbot, state])
    clear_btn.click(lambda: [], None, chatbot)  # Nullstiller chatten

if __name__ == "__main__":
    demo.launch()