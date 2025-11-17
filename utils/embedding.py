import re
import time
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# Normalization

def normalize_for_embedding(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[\n\r\t]", " ", text)
    text = re.sub(r"[^a-zçğıöşü0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text



def generate_embeddings_for_unique_texts(
    df: pd.DataFrame,
    text_column: str,
    normalize_fn,
    embed_model_name: str = "distiluse-base-multilingual-cased-v2",
    batch_size: int = 32,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, SentenceTransformer]:
    """
    - Textleri normalize edip unique hale getirir.
    - SentenceTransformer ile embedding üretir.
    - emb_matrix'i float16'e cast ederek bellek kullanımını düşürür.

    Returns:
        emb_matrix: shape (N_unique, d), dtype=float16
        unique_texts: np.ndarray (N_unique,)
        mapping_df: text -> emb_idx eşleşmesi
        embed_model: yüklü model
    """
    print(f"[INFO] Embedding modeli yükleniyor: {embed_model_name}")
    embed_model = SentenceTransformer(embed_model_name)

    normalized_series = df[text_column].astype(str).apply(normalize_fn)
    unique_texts = normalized_series.unique()
    print(f"[INFO] Unique (normalized) text sayısı: {len(unique_texts)}")

    print("[INFO] Embedding oluşturuluyor...")
    emb_matrix = embed_model.encode(
        list(unique_texts),
        batch_size=batch_size,
        show_progress_bar=True
    )

    emb_matrix = emb_matrix.astype("float16")

    mapping_df = pd.DataFrame({
        text_column: unique_texts,
        "emb_idx": np.arange(len(unique_texts), dtype=int)
    })

    print("[INFO] Embedding işlemi tamamlandı.")
    print(f"[INFO] emb_matrix shape: {emb_matrix.shape}, dtype: {emb_matrix.dtype}")

    return emb_matrix, unique_texts, mapping_df, embed_model



def _retrieve_topk_indices_and_sims(
    query_text: str,
    embed_model: SentenceTransformer,
    emb_matrix: np.ndarray,
    unique_texts: np.ndarray,
    normalize_fn,
    top_k: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Query'yi embed eder, tüm emb_matrix ile cosine similarity hesaplar,
    en benzer top_k index'i ve skorlarını döner.

    - np.argpartition kullanarak O(N) top_k seçimi yapar (tam sort'tan daha hızlı).
    """
    q_norm = normalize_fn(query_text)
    query_emb = embed_model.encode([q_norm])

    sims = cosine_similarity(query_emb, emb_matrix)[0].astype("float32")  # (N,)

    if top_k >= len(sims):
        top_idx = np.argsort(-sims)
    else:
        # Önce top_k'yi partition ile seç (O(N)), sonra kendi içinde sort et.
        top_idx_part = np.argpartition(-sims, top_k - 1)[:top_k]
        top_idx = top_idx_part[np.argsort(-sims[top_idx_part])]

    sims_top = sims[top_idx]
    return top_idx, sims_top



def most_similar_feedback(
    query_text: str,
    top_k: int,
    embed_model: SentenceTransformer,
    emb_matrix: np.ndarray,
    unique_texts: np.ndarray,
    normalize_fn=normalize_for_embedding,
) -> pd.DataFrame:
    """
    Saf semantic search:
    - Query embedding + cosine similarity
    - En benzer top_k normalized feedback
    """
    top_idx, sims_top = _retrieve_topk_indices_and_sims(
        query_text=query_text,
        embed_model=embed_model,
        emb_matrix=emb_matrix,
        unique_texts=unique_texts,
        normalize_fn=normalize_fn,
        top_k=top_k,
    )

    result = pd.DataFrame({
        "Feedback_norm": unique_texts[top_idx],
        "similarity": sims_top,
    })

    return result



def rag_query(
    df: pd.DataFrame,
    query: str,
    embed_model: SentenceTransformer,
    emb_matrix: np.ndarray,
    unique_texts: np.ndarray,
    top_k: int = 20,
    feedback_col: str = "Feedback_norm",
    sentiment_col: str = "sentiment_score_model",
    score_col: str = "Score",
    normalize_fn=normalize_for_embedding,
):
    """
    Optimize edilmiş RAG sorgusu + zaman ölçümü.

    Returns:
        sub_df (pd.DataFrame)
        summary (dict)
        retrieval_time (float, seconds)
        stats_time (float, seconds)
        total_time (float, seconds)
    """


    t0 = time.perf_counter()

    top_idx, sims_top = _retrieve_topk_indices_and_sims(
        query_text=query,
        embed_model=embed_model,
        emb_matrix=emb_matrix,
        unique_texts=unique_texts,
        normalize_fn=normalize_fn,
        top_k=top_k,
    )
    retrieved_texts = unique_texts[top_idx]


    sub_df = df[df[feedback_col].isin(retrieved_texts)].copy()


    sim_map: Dict[str, float] = dict(zip(retrieved_texts, sims_top))
    sub_df["similarity"] = sub_df[feedback_col].map(sim_map)

    t1 = time.perf_counter()
    retrieval_time = t1 - t0

    if sentiment_col in sub_df.columns:
        sentiment_mean = float(sub_df[sentiment_col].mean())
        sentiment_std = float(sub_df[sentiment_col].std())
    else:
        sentiment_mean = None
        sentiment_std = None

    if score_col in sub_df.columns:
        score_mean = float(sub_df[score_col].mean())
        score_std = float(sub_df[score_col].std())
    else:
        score_mean = None
        score_std = None

    t2 = time.perf_counter()
    stats_time = t2 - t1

    total_time = retrieval_time + stats_time

    summary = {
        "query": query,
        "n_comments": int(len(sub_df)),
        "mean_sentiment_model": sentiment_mean,
        "std_sentiment_model": sentiment_std,
        "mean_user_score": score_mean,
        "std_user_score": score_std,
        "retrieval_time_s": retrieval_time,
        "stats_time_s": stats_time,
        "total_time_s": total_time,
    }

    return sub_df, summary, retrieval_time, stats_time, total_time



# -----------------------------
# Latency karşılaştırma (retrieval + RAG pipeline)
# -----------------------------
def compare_rag_vs_nonrag(
    df: pd.DataFrame,
    query: str,
    embed_model: SentenceTransformer,
    emb_matrix: np.ndarray,
    unique_texts: np.ndarray,
    normalize_fn=normalize_for_embedding,
    top_k: int = 20,
) -> Tuple[float, float]:
    """
    Non-RAG (most_similar_feedback) vs RAG (rag_query) latency karşılaştırması.
    Grafik üretimi yok, sadece retrieval + RAG'in kendi iç işlemleri ölçülür.
    """
    # Non-RAG
    t0 = time.perf_counter()
    _ = most_similar_feedback(
        query_text=query,
        top_k=top_k,
        embed_model=embed_model,
        emb_matrix=emb_matrix,
        unique_texts=unique_texts,
        normalize_fn=normalize_fn,
    )
    t1 = time.perf_counter()
    nonrag_time = t1 - t0

    # RAG
    sub_df, summary, retrieval_time, stats_time, total_time = rag_query(
        df=df,
        query=query,
        embed_model=embed_model,
        emb_matrix=emb_matrix,
        unique_texts=unique_texts,
        top_k=top_k,
        feedback_col="Feedback_norm",
        sentiment_col="sentiment_score_model",
        score_col="Score",
        normalize_fn=normalize_fn,
    )

    rag_time = retrieval_time

    return nonrag_time, rag_time
