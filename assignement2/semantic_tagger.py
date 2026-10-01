import os
import string
import math
import numpy as np
import gensim.downloader as api


# ==============================================================================
# Vector Class Implementation (Part E requirement)
# ==============================================================================
class Vector:
    """
    Vector abstraction representing mathematical vectors using tuples for storage.
    Supports dot product, Euclidean norm, and cosine similarity.
    """
    def __init__(self, elements=None):
        if elements is None:
            self.elements = ()
        elif isinstance(elements, Vector):
            self.elements = elements.elements
        elif isinstance(elements, np.ndarray):
            self.elements = tuple(float(x) for x in elements.flat)
        else:
            self.elements = tuple(float(x) for x in elements)

    def dot(self, other: "Vector") -> float:
        if len(self.elements) != len(other.elements):
            raise ValueError(f"Vector dimensions do not match: {len(self.elements)} vs {len(other.elements)}")
        return sum(a * b for a, b in zip(self.elements, other.elements))

    def norm(self) -> float:
        return math.sqrt(sum(x * x for x in self.elements))

    def cosine_similarity(self, other: "Vector") -> float:
        norm_self = self.norm()
        norm_other = other.norm()
        if norm_self == 0.0 or norm_other == 0.0:
            raise ValueError("Cosine similarity is undefined for zero vectors.")
        return self.dot(other) / (norm_self * norm_other)

    def __len__(self):
        return len(self.elements)

    def __repr__(self):
        return f"Vector({self.elements[:3]}...)" if len(self.elements) > 3 else f"Vector({self.elements})"


# ==============================================================================
# Part A — Choose 20 Tags
# ==============================================================================
TAGS = [
    "research", "innovation", "education", "university", "students",
    "faculty", "campus", "engineering", "medicine", "technology",
    "curriculum", "collaboration", "publication", "laboratory", "scholarship",
    "mentorship", "internship", "entrepreneurship", "accreditation", "alumni"
]


def build_tag_matrix(model, tags):
    """
    Construct the tag matrix T in R^(20 x 50).
    Handles single-word and multi-word tags.
    Raises ValueError if any tag (or component word) is out-of-vocabulary.
    """
    tag_vectors = []
    for tag in tags:
        words = tag.strip().lower().split()
        if not words:
            raise ValueError("Tag cannot be empty.")
        
        comp_vectors = []
        for w in words:
            if w not in model.key_to_index:
                raise ValueError(f"Tag component '{w}' in tag '{tag}' is not in model vocabulary.")
            comp_vectors.append(model[w])
        
        # Mean vector for multi-word tag, or single vector for single-word tag
        tag_vec = np.mean(comp_vectors, axis=0)
        tag_vectors.append(tag_vec)

    T = np.vstack(tag_vectors)
    return tags, T


# ==============================================================================
# Part B & C — Text Loading and Preprocessing Pipeline
# ==============================================================================
STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to",
    "in", "on", "for", "is", "are", "was", "were",
    "with", "at", "by", "from",
    "it", "this", "that", "our", "we", "they",
    "their", "has", "have", "had", "as", "be",
    "its", "all", "so", "into", "across", "through"
}


def load_text(filename: str) -> str:
    """Load text file relative to current script directory."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, filename)
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def preprocess_text(text: str) -> list[str]:
    """
    Preprocess text strictly in required order:
    1. convert to lowercase
    2. remove punctuation
    3. split into whitespace tokens
    4. remove custom stopwords
    5. remove tokens with len <= 2
    """
    text_lower = text.lower()
    text_clean = text_lower.translate(str.maketrans("", "", string.punctuation))
    tokens = text_clean.split()
    tokens_no_stop = [tok for tok in tokens if tok not in STOPWORDS]
    tokens_filtered = [tok for tok in tokens_no_stop if len(tok) > 2]
    return tokens_filtered


# ==============================================================================
# Part D — Build Text Matrix W
# ==============================================================================
def build_text_matrix(model, tokens: list[str]):
    """
    Construct text matrix W in R^(n x 50).
    Filters tokens by model vocabulary, preserving order and duplicates.
    """
    in_vocab_tokens = []
    oov_tokens = []
    word_vectors = []

    for token in tokens:
        if token in model.key_to_index:
            in_vocab_tokens.append(token)
            word_vectors.append(model[token])
        else:
            oov_tokens.append(token)

    n = len(in_vocab_tokens)
    if n == 0:
        raise ValueError("The input text contains no in-vocabulary words. Cannot construct text matrix W.")

    W = np.vstack(word_vectors)
    return in_vocab_tokens, oov_tokens, W


# ==============================================================================
# Part E — Cosine Similarity Using Vector Class (Loops)
# ==============================================================================
def similarity_matrix_vector_class(W: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    Computes S in R^(n x 20) using the custom Vector class and nested Python loops.
    Every entry S[i, j] = Vector(W[i]).cosine_similarity(Vector(T[j])).
    """
    n_words = W.shape[0]
    n_tags = T.shape[0]
    S = np.zeros((n_words, n_tags), dtype=float)

    for i in range(n_words):
        for j in range(n_tags):
            S[i, j] = Vector(W[i]).cosine_similarity(Vector(T[j]))

    return S


# ==============================================================================
# Part F — Cosine Similarity Using Vectorized NumPy
# ==============================================================================
def similarity_matrix_numpy(W: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    Computes S in R^(n x 20) via row normalization and matrix multiplication:
    S = W_hat @ T_hat.T
    """
    W_norm = np.linalg.norm(W, axis=1, keepdims=True)
    T_norm = np.linalg.norm(T, axis=1, keepdims=True)

    if np.any(W_norm == 0) or np.any(T_norm == 0):
        raise ValueError("Zero-norm vector encountered in W or T.")

    W_hat = W / W_norm
    T_hat = T / T_norm
    S = W_hat @ T_hat.T
    return S


# ==============================================================================
# Part H — Rank Tags Using Max Pooling
# ==============================================================================
def rank_tags(tag_names: list[str], S: np.ndarray, in_vocab_tokens: list[str]):
    """
    Computes max-pool scores r_j = max_i S[i, j] for each tag.
    Returns ranked tags sorted descending by score with best-matching words.
    """
    ranked = []
    for j, tag in enumerate(tag_names):
        col = S[:, j]
        max_idx = int(np.argmax(col))
        max_score = float(col[max_idx])
        best_word = in_vocab_tokens[max_idx]
        is_exact = (tag == best_word)
        ranked.append({
            "tag": tag,
            "score": max_score,
            "best_word": best_word,
            "is_exact": is_exact
        })

    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked


# ==============================================================================
# Main Full Pipeline & Execution
# ==============================================================================
def main():
    print("=" * 65)
    print("  AME 5151 - Assignment 2: Semantic Tagging with GloVe & NumPy")
    print("=" * 65)

    # 1. Load Pretrained Model
    print("\n[1] Loading GloVe 50-d embeddings (glove-wiki-gigaword-50)...")
    model = api.load("glove-wiki-gigaword-50")
    print("    Model successfully loaded.")

    # 2. Build Tag Matrix T
    print("\n[2] Constructing Tag Matrix T...")
    tag_names, T = build_tag_matrix(model, TAGS)
    print(f"    Tags defined: {len(tag_names)}")
    print(f"    T.shape = {T.shape}")

    # 3. Load & Preprocess Input Text
    print("\n[3] Loading & Preprocessing 'manipal_text.txt'...")
    raw_text = load_text("manipal_text.txt")
    raw_tokens = raw_text.split()
    processed_tokens = preprocess_text(raw_text)

    # 4. Build Text Matrix W & Filter OOV
    in_vocab_tokens, oov_tokens, W = build_text_matrix(model, processed_tokens)
    distinct_oov = sorted(list(set(oov_tokens)))

    print("\n--- Token Statistics ---")
    print(f"Tokens before preprocessing : {len(raw_tokens)}")
    print(f"Tokens after preprocessing  : {len(processed_tokens)}")
    print(f"In-vocabulary tokens (n)    : {len(in_vocab_tokens)}")
    print(f"Out-of-vocabulary tokens    : {len(oov_tokens)}")
    print(f"Distinct OOV tokens ({len(distinct_oov)}): {distinct_oov}")

    print("\n--- Matrix Dimension Checkpoint ---")
    print(f"T.shape = {T.shape}")
    print(f"W.shape = {W.shape}")

    # 5. Compute Similarity Matrix using Vector class (Part E)
    print("\n[5] Computing Similarity Matrix using custom Vector class...")
    S_vector = similarity_matrix_vector_class(W, T)
    print(f"    S_vector.shape = {S_vector.shape}")

    # 6. Compute Similarity Matrix using NumPy (Part F)
    print("\n[6] Computing Similarity Matrix using NumPy matrix multiplication...")
    S_numpy = similarity_matrix_numpy(W, T)
    print(f"    S_numpy.shape  = {S_numpy.shape}")

    # 7. Numerical Verification (Part G)
    print("\n[7] Numerical Verification between Vector class and NumPy...")
    matches = np.allclose(S_vector, S_numpy, atol=1e-6)
    max_diff = float(np.max(np.abs(S_vector - S_numpy)))
    print(f"    np.allclose(S_vector, S_numpy) : {matches}")
    print(f"    Max absolute numerical diff    : {max_diff:.2e}")

    # 8. Tag Ranking via Max Pooling (Part H)
    print("\n[8] Max-Pool Tag Ranking (Top 8 Tags):")
    ranked = rank_tags(tag_names, S_numpy, in_vocab_tokens)
    
    print("-" * 65)
    print(f"{'Rank':<6}{'Tag':<18}{'Score':<10}{'Best-Matching Text Word':<22}{'Match Type'}")
    print("-" * 65)
    for rank, item in enumerate(ranked[:8], start=1):
        mtype = "Identical (Exact)" if item["is_exact"] else "Semantic (Diff Word)"
        print(f"{rank:<6}{item['tag']:<18}{item['score']:<10.4f}{item['best_word']:<22}{mtype}")
    print("-" * 65)


if __name__ == "__main__":
    main()