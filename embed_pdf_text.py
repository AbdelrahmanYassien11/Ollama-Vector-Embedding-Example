import ollama
import chromadb
import fitz  # PyMuPDF
import math

# --------- Helpers ---------


def extract_text_from_pdf(pdf_path):
    """Extract raw text from each page and return one big string."""
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        pages.append(page.get_text())
    doc.close()
    return "\n".join(pages)


def chunk_text(text, chunk_size=400, overlap=50):
    """
    Create overlapping chunks of approximately chunk_size characters.
    Returns list of non-empty chunks.
    """
    if not text:
        return []
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap  # overlap some characters
    # filter out overly short junk
    filtered = [c for c in chunks if len(c) >= 40]  # keep reasonably-sized chunks
    return filtered


def safe_embed(text):
    """
    Call ollama.embed for a single input and return the embedding vector (not nested extra).
    Ollama returns {"embeddings": [[...]]} for a single input; return that inner vector.
    """
    resp = ollama.embed(model="mxbai-embed-large", input=text)
    embs = resp.get("embeddings")
    # embs is typically a list of vectors; return the first vector
    if not embs:
        raise RuntimeError("Embedding API returned no embeddings.")
    return embs[0]  # single vector


# --------- Main flow ---------

pdf_path = "Abdelrahman_Yassien_CV.pdf"  # adjust path
raw_text = extract_text_from_pdf(pdf_path)

# chunk into paragraph-like pieces
chunks = chunk_text(raw_text, chunk_size=600, overlap=100)
print(f"Created {len(chunks)} chunks from PDF")

# create / connect to Chroma
client = chromadb.Client()
collection = client.create_collection(name="docs")

# Add chunks to Chroma (one by one). Use string IDs.
for i, chunk in enumerate(chunks):
    # embed the chunk (returns single vector)
    vec = safe_embed(chunk)  # vec is a list of floats
    # collection.add expects a list-of-vectors for embeddings,
    # and lists for ids/documents (batch form)
    collection.add(ids=[str(i)], embeddings=[vec], documents=[chunk])

print("Indexed chunks into Chroma.")

# Example user question
prompt_text = "What is abdelrahman's college?"

# embed question
q_resp = ollama.embed(model="mxbai-embed-large", input=prompt_text)
# q_resp["embeddings"] is [[...]] -> pass it directly (list-of-vectors)
query_embedding = q_resp["embeddings"]  # this is the format Chroma expects: [[...]]
# or use query_embedding = [safe_embed(prompt_text)]

# search top k matches
top_k = 5
results = collection.query(query_embeddings=query_embedding, n_results=top_k)

# results["documents"] is a list-of-lists: one query -> list of docs
matched_docs = results.get("documents", [[]])[0]  # list of matching doc strings
matched_ids = results.get("ids", [[]])[0]
distances = results.get("distances", [[]])[0]

print(f"Retrieved {len(matched_docs)} docs. IDs: {matched_ids}, distances: {distances}")

# Build the context by joining the retrieved chunks (keep in order of closeness)
context = "\n\n---\n\n".join(matched_docs).strip()

# Prepare a safe RAG prompt: instruct the LLM not to hallucinate and to answer only based on context
rag_prompt = (
    "You are given the following extracted passages from a resume/CV as context.\n\n"
    f"{context}\n\n"
    "Using ONLY the information provided above, answer the user's question below. "
    "If the answer is not present in the context, reply exactly: 'I don't know.'\n\n"
    f"Question: {prompt_text}\n\n"
    "Answer concisely and cite (by chunk index) any passages you used if possible."
)

# Generate the answer using llama3.2
output = ollama.generate(
    model="llama3.2",
    prompt=rag_prompt,
)

print("LLM response:\n", output.get("response", ""))
