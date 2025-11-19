import os
import re
import torch
import faiss
import pandas as pd
import streamlit as st

from pathlib import Path
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# ============================================================
# CONFIGURACIÓN
# ============================================================
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Cliente OpenAI
# client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])



# ============================================================
# CARGAR CSVs (PRODUCTS + INVENTORY)
# ============================================================
@st.cache_data
def load_csvs():
    products = pd.read_csv("data/products.csv")
    inventory = pd.read_csv("data/inventory.csv")
    return products, inventory

products_df, inventory_df = load_csvs()


# ============================================================
# FUNCIONES PARA SKUS
# ============================================================
def find_sku(query: str):
    """
    Detecta si la consulta contiene un SKU.
    Condición: una palabra que comience por 'SKU'.
    """
    query_upper = query.upper()
    words = query_upper.replace("?", " ").replace(",", " ").split()

    for w in words:
        if w.startswith("SKU") and len(w) > 3:  # ejemplo: SKU12345
            return w.strip()

    return None


# ============================================================
# FUNCIONES SOBRE CSVs
# ============================================================
def check_stock(sku_input):
    sku_input = sku_input.upper().strip()
    df = inventory_df.copy()
    df["sku"] = df["sku"].astype(str).str.upper().str.strip()

    match = df[df["sku"] == sku_input]

    if match.empty:
        return f"No encontré inventario para el SKU {sku_input}."

    stock = int(match.iloc[0]["stock"])

    if stock > 0:
        return f"Sí, hay {stock} unidades del SKU {sku_input}."
    else:
        return f"SKU {sku_input} encontrado, pero no hay stock disponible."


def check_impermeable(sku_input):
    sku_input = sku_input.upper().strip()
    df = products_df.copy()
    df["sku"] = df["sku"].astype(str).str.upper().str.strip()

    match = df[df["sku"] == sku_input]

    if match.empty:
        return f"No encontré información de producto para el SKU {sku_input}."

    val = match.iloc[0]["impermeable"]

    if val == 1:
        return f"Sí, el producto con SKU {sku_input} es impermeable."
    else:
        return f"No, el producto con SKU {sku_input} no es impermeable."


# ============================================================
# MODELO DE EMBEDDINGS
# ============================================================
@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBED_MODEL, device=DEVICE)

embedder = load_embedder()


# ============================================================
# CHUNKING DE DOCUMENTOS PARA RAG
# ============================================================
@st.cache_data
def load_and_chunk(folder="data", chunk_size=400, overlap=100):
    docs = []
    for file in Path(folder).glob("*"):
        if file.suffix not in [".txt", ".md"]:
            continue

        text = file.read_text(encoding="utf-8")
        start = 0
        while start < len(text):
            end = start + chunk_size
            docs.append({
                "chunk": text[start:end],
                "source": file.name
            })
            start = max(end - overlap, 0)
    return docs

docs = load_and_chunk()


# ============================================================
# FAISS VECTOR STORE
# ============================================================
@st.cache_resource
def build_faiss_index(docs):
    texts = [d["chunk"] for d in docs]
    vectors = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    dim = vectors.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    return index

index = build_faiss_index(docs)


# ============================================================
# RETRIEVE
# ============================================================
def retrieve(query, k=3):
    q_emb = embedder.encode([query], convert_to_numpy=True)
    distances, idxs = index.search(q_emb, k)
    return [
        {
            "score": float(dist),
            "chunk": docs[idx]["chunk"],
            "source": docs[idx]["source"]
        }
        for dist, idx in zip(distances[0], idxs[0])
    ]


# ============================================================
# OPENAI LLM
# ============================================================
def build_prompt(query, retrieved):
    context = ""
    for r in retrieved:
        context += f"[Fuente: {r['source']}]\n{r['chunk']}\n\n"

    return f"""
Eres un asistente RAG.
Responde SOLO con información del contexto recuperado.
Si la respuesta no está en el contexto, responde:
"No encontré información en los documentos".

Contexto:
{context}

Pregunta:
{query}

Respuesta (incluye citas de fuente):
"""
# ============================================================
# RESPEUSTA HIBRIDA SKU SEARCH + RAG
# ============================================================

def generate_answer(prompt):
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    return res.choices[0].message.content

def answer_with_sku_and_rag(query, sku):
    # 1. info estructurada del CSV
    stock_info = check_stock(sku)
    imp_info = check_impermeable(sku)

    # 2. quitar el SKU de la pregunta antes del RAG
    query_clean = query.replace(sku, "").strip()

    # 3. recuperar documentos FAQ
    retrieved = retrieve(query_clean)
    context_docs = ""
    for r in retrieved:
        context_docs += f"[Fuente: {r['source']}]\n{r['chunk']}\n\n"

    # 4. construir mega-contexto combinado
    prompt = f"""
Eres un asistente corporativo.
Primero usa la información del producto (SKU) y luego la información de los documentos.

Información del SKU:
- {stock_info}
- {imp_info}

Información de documentos recuperados:
{context_docs}

Pregunta del usuario:
{query}

Respuesta final (natural, integrada, profesional):
"""

    # 5. generar respuesta LLM
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return res.choices[0].message.content, retrieved

# ============================================================
# UI STREAMLIT
# ============================================================
st.title("Asistente GenAI – RAG (BERT + FAISS + CSV)")

# Modo debug
debug = st.sidebar.checkbox("🔍 Modo Debug (ver chunks recuperados)")

query = st.text_input("Haz una pregunta:")

if query:

    # DETECTAR SKU primero
    sku_detectado = find_sku(query)

    if sku_detectado:
        # --- RESPUESTA HÍBRIDA: SKU + RAG ---
        answer, retrieved = answer_with_sku_and_rag(query, sku_detectado)

        st.subheader("Respuesta:")
        st.write(answer)

        # Debug opcional
        if debug:
            st.subheader("Fragmentos recuperados:")
            for r in retrieved:
                st.markdown(f"**{r['source']} – score: {r['score']:.4f}**")
                st.write(r["chunk"])

            st.subheader("Prompt enviado al LLM:")
            st.code(answer)

        st.stop()   # IMPORTANTE porque evita que pase al RAG normal

    # SI NO HAY SKU se usar RAG puro
    retrieved = retrieve(query)
    prompt = build_prompt(query, retrieved)
    answer = generate_answer(prompt)

    st.subheader("Respuesta:")
    st.write(answer)

    if debug:
        st.subheader("Fragmentos recuperados:")
        for r in retrieved:
            st.markdown(f"**{r['source']} – score: {r['score']:.4f}**")
            st.write(r["chunk"])

        st.subheader("Prompt enviado al LLM:")
        st.code(prompt)
