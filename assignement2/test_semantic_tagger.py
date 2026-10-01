import pytest
import numpy as np
from semantic_tagger import (
    Vector,
    TAGS,
    build_tag_matrix,
    load_text,
    preprocess_text,
    build_text_matrix,
    similarity_matrix_vector_class,
    similarity_matrix_numpy,
)


@pytest.fixture(scope="module")
def loaded_components():
    import gensim.downloader as api
    model = api.load("glove-wiki-gigaword-50")
    tag_names, T = build_tag_matrix(model, TAGS)
    raw_text = load_text("manipal_text.txt")
    processed_tokens = preprocess_text(raw_text)
    in_vocab_tokens, oov_tokens, W = build_text_matrix(model, processed_tokens)
    S_vector = similarity_matrix_vector_class(W, T)
    S_numpy = similarity_matrix_numpy(W, T)
    return {
        "model": model,
        "tag_names": tag_names,
        "T": T,
        "in_vocab_tokens": in_vocab_tokens,
        "oov_tokens": oov_tokens,
        "W": W,
        "S_vector": S_vector,
        "S_numpy": S_numpy,
    }


# Test 1: Verify T.shape == (20, 50)
def test_1_tag_matrix_shape(loaded_components):
    T = loaded_components["T"]
    assert T.shape == (20, 50), f"Expected T.shape to be (20, 50), got {T.shape}"


# Test 2: Verify W.shape == (n, 50)
def test_2_text_matrix_shape(loaded_components):
    W = loaded_components["W"]
    n = len(loaded_components["in_vocab_tokens"])
    assert W.shape == (n, 50), f"Expected W.shape to be ({n}, 50), got {W.shape}"


# Test 3: Verify S.shape == (n, 20)
def test_3_similarity_matrix_shape(loaded_components):
    S_numpy = loaded_components["S_numpy"]
    n = len(loaded_components["in_vocab_tokens"])
    assert S_numpy.shape == (n, 20), f"Expected S.shape to be ({n}, 20), got {S_numpy.shape}"


# Test 4: Pairwise similarity verification between Vector class and NumPy
def test_4_pairwise_vector_vs_numpy(loaded_components):
    W = loaded_components["W"]
    T = loaded_components["T"]
    S_numpy = loaded_components["S_numpy"]

    # Select pair (word index 0, tag index 0)
    w_vec = Vector(W[0])
    t_vec = Vector(T[0])
    cosine_val = w_vec.cosine_similarity(t_vec)

    assert abs(cosine_val - S_numpy[0, 0]) < 1e-6, (
        f"Vector cosine similarity {cosine_val} does not match NumPy {S_numpy[0, 0]}"
    )


# Test 5: Verify np.allclose(S_vector, S_numpy)
def test_5_allclose_verification(loaded_components):
    S_vector = loaded_components["S_vector"]
    S_numpy = loaded_components["S_numpy"]
    assert np.allclose(S_vector, S_numpy, atol=1e-6), "S_vector and S_numpy are not allclose within tolerance!"


# Test 6: Verify out-of-vocabulary token excluded from W and reported in oov_tokens
def test_6_oov_handling(loaded_components):
    model = loaded_components["model"]
    tokens = ["research", "supercalifragilisticexpialidocious_oov_xyz", "students"]
    in_vocab, oov, W = build_text_matrix(model, tokens)

    assert "supercalifragilisticexpialidocious_oov_xyz" in oov
    assert "supercalifragilisticexpialidocious_oov_xyz" not in in_vocab
    assert W.shape == (2, 50)


# Test 7: Verify tag vocabulary with OOV component raises an error in build_tag_matrix
def test_7_invalid_tag_raises_error(loaded_components):
    model = loaded_components["model"]
    invalid_tags = ["research", "completely_unknown_tag_word_xyz123"]
    with pytest.raises(ValueError, match="is not in model vocabulary"):
        build_tag_matrix(model, invalid_tags)
