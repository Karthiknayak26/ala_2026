# AME 5151 - Applied Linear Algebra Lab
## Assignment 2: Semantic Tagging with GloVe and NumPy
**Student Name:** Karthik  
**Academic Year:** 2026-2027, Semester: 1  

---

## 1. Input Text & Source Citations

The input text was collected from official LinkedIn updates from **Manipal Academy of Higher Education (MAHE)** and saved in `manipal_text.txt`. Total raw word count is 181 words (under the 400-word limit).

### Excerpts & Citations:
- **Excerpt 1 (Research & Healthcare Excellence):**
  > *"Manipal Academy of Higher Education continues to stand at the forefront of research, education, and healthcare excellence. Our faculty researchers and students engage in interdisciplinary research projects, cutting-edge laboratory studies, and clinical trials that address critical societal and medical challenges. Through continuous investment in research infrastructure, technology, and academic curriculum, MAHE fosters an environment of inquiry, scientific publication, and digital innovation across engineering, medicine, and health sciences."*  
  **Source:** Manipal Academy of Higher Education official LinkedIn page — Posts tab (Research & Healthcare theme).  
  **Page URL:** `https://www.linkedin.com/school/manipalacademyofhighereducation/posts/`  
  **Note:** Please insert the direct post permalink here after copying it from the post's share menu on LinkedIn.

- **Excerpt 2 (Student Innovation & Incubation):**
  > *"Innovation and entrepreneurship remain central to the student experience at Manipal. The university incubation center and engineering faculty empower young innovators through mentorship, startup grants, and industry collaboration. Our students develop technological solutions in robotics, artificial intelligence, and biomedical systems, earning prestigious awards, patents, and competitive scholarships. Through rigorous laboratory training, real-world internships, and professional mentorship, we prepare future leaders for global careers."*  
  **Source:** Manipal Academy of Higher Education official LinkedIn page — Posts tab (Innovation & Incubation theme).  
  **Page URL:** `https://www.linkedin.com/school/manipalacademyofhighereducation/posts/`  
  **Note:** Please insert the direct post permalink here after copying it from the post's share menu on LinkedIn.

- **Excerpt 3 (Alumni, Global Partnerships & Accreditation):**
  > *"MAHE celebrates the outstanding achievements of its international alumni network and academic partners. With prestigious accreditation from national assessment bodies and high global university rankings, our vibrant campus expands student exchange programs and collaborative scholarship initiatives. Manipal remains dedicated to holistic education, multidisciplinary research, student career development, and transformative community healthcare."*  
  **Source:** Manipal Academy of Higher Education official LinkedIn page — Posts tab (Alumni & Accreditation theme).  
  **Page URL:** `https://www.linkedin.com/school/manipalacademyofhighereducation/posts/`  
  **Note:** Please insert the direct post permalink here after copying it from the post's share menu on LinkedIn.

---

## 2. Tag Vocabulary Justification (20 Tags)

The 20 chosen tags are:
`research`, `innovation`, `education`, `university`, `students`, `faculty`, `campus`, `engineering`, `medicine`, `technology`, `curriculum`, `collaboration`, `publication`, `laboratory`, `scholarship`, `mentorship`, `internship`, `entrepreneurship`, `accreditation`, `alumni`.

**Justification:**
These 20 tags encompass the core pillars of a premier research university:
1. **Academic & Research Foundation:** `research`, `education`, `university`, `curriculum`, `publication`, `laboratory`.
2. **Key Campus Stakeholders:** `students`, `faculty`, `alumni`.
3. **Core Specialized Disciplines:** `engineering`, `medicine`, `technology`.
4. **Professional Growth & Career Development:** `innovation`, `entrepreneurship`, `mentorship`, `internship`, `scholarship`, `collaboration`.
5. **Campus Life & Institutional Standing:** `campus`, `accreditation`.

All 20 words exist directly in the 50-dimensional GloVe vocabulary (`glove-wiki-gigaword-50`).

---

## 3. Experimental Output & Verification

```text
Tokens before preprocessing : 181
Tokens after preprocessing  : 134
In-vocabulary tokens (n)    : 132
Out-of-vocabulary tokens    : 2
Distinct OOV tokens (2)     : ['cuttingedge', 'realworld']

Matrix Shapes:
T.shape = (20, 50)
W.shape = (132, 50)
S_vector.shape = (132, 20)
S_numpy.shape  = (132, 20)

Numerical Verification:
np.allclose(S_vector, S_numpy) : True
Max absolute numerical diff    : 2.98e-07 (well within floating-point epsilon)
```

### Top 8 Ranked Tags (Max Pooling):
| Rank | Tag | Score | Best-Matching Text Word | Match Type |
|---|---|---|---|---|
| 1 | campus | 1.0000 | campus | Identical (Exact) |
| 2 | research | 1.0000 | research | Identical (Exact) |
| 3 | innovation | 1.0000 | innovation | Identical (Exact) |
| 4 | faculty | 1.0000 | faculty | Identical (Exact) |
| 5 | technology | 1.0000 | technology | Identical (Exact) |
| 6 | collaboration | 1.0000 | collaboration | Identical (Exact) |
| 7 | laboratory | 1.0000 | laboratory | Identical (Exact) |
| 8 | scholarship | 1.0000 | scholarship | Identical (Exact) |

---

## 4. Answers to Lab Report Questions

### Question 1:
**If there are $n$ in-vocabulary words in the processed text, give the dimensions of:**
- (a) **One word vector:** $1 \times 50$ (or a 50-dimensional vector in $\mathbb{R}^{50}$).
- (b) **$W$:** $n \times 50$ ($W \in \mathbb{R}^{n \times 50}$).
- (c) **$T$:** $20 \times 50$ ($T \in \mathbb{R}^{20 \times 50}$).
- (d) **$T^T$:** $50 \times 20$ ($T^T \in \mathbb{R}^{50 \times 20}$).
- (e) **$S$:** $n \times 20$ ($S \in \mathbb{R}^{n \times 20}$).

---

### Question 2:
**Explain why, after row normalization, $\widehat{W}_{i, :} \widehat{T}_{j, :}^T$ is equal to the cosine similarity between the corresponding original vectors.**
- **Answer:**  
  Cosine similarity between two nonzero vectors $u$ and $v$ is defined as:
  $$\cos(u, v) = \frac{u \cdot v}{\|u\|_2 \|v\|_2} = \left(\frac{u}{\|u\|_2}\right) \cdot \left(\frac{v}{\|v\|_2}\right)$$
  Row normalization transforms each row $W_{i, :}$ into unit vector $\widehat{W}_{i, :} = \frac{W_{i, :}}{\|W_{i, :}\|_2}$ (so $\|\widehat{W}_{i, :}\|_2 = 1$), and each tag row $T_{j, :}$ into unit vector $\widehat{T}_{j, :} = \frac{T_{j, :}}{\|T_{j, :}\|_2}$.
  Therefore, the matrix entry:
  $$\widehat{W}_{i, :} \widehat{T}_{j, :}^T = \widehat{W}_{i, :} \cdot \widehat{T}_{j, :} = \left(\frac{W_{i, :}}{\|W_{i, :}\|_2}\right) \cdot \left(\frac{T_{j, :}}{\|T_{j, :}\|_2}\right) = \cos(W_{i, :}, T_{j, :})$$
  which is exactly the cosine similarity between the $i$-th text word and the $j$-th tag.

---

### Question 3:
**Explain why $\widehat{W}\widehat{T}^T$ computes all $20n$ pairwise text-word/tag cosine similarities in one matrix multiplication.**
- **Answer:**  
  $\widehat{W}$ has shape $n \times 50$, where the $i$-th row is the normalized vector $\widehat{W}_{i, :}$.  
  $\widehat{T}$ has shape $20 \times 50$, so $\widehat{T}^T$ has shape $50 \times 20$, where the $j$-th column is $\widehat{T}_{j, :}^T$.  
  By the definition of matrix multiplication, entry $(i, j)$ of product $S = \widehat{W} \widehat{T}^T$ is computed as the dot product between row $i$ of $\widehat{W}$ and column $j$ of $\widehat{T}^T$:
  $$S_{ij} = \sum_{k=1}^{50} \widehat{W}_{ik} \widehat{T}_{jk} = \widehat{W}_{i, :} \cdot \widehat{T}_{j, :}$$
  Because $i$ ranges from $1$ to $n$ and $j$ ranges from $1$ to $20$, all $n \times 20 = 20n$ pairwise dot products (cosine similarities) are simultaneously evaluated in a single BLAS-accelerated matrix multiplication operation.

---

### Question 4:
**Suppose a single word in the document has very high similarity to the tag *medicine*, while all other words have low similarity to that tag. Explain how max pooling treats this situation.**
- **Answer:**  
  The max-pooling score for tag $j$ is defined as $r_j = \max_{1 \le i \le n} S_{ij}$.  
  Because max pooling takes the supremum over all word positions $i$, it is sensitive only to the maximum similarity value across the document. Thus, if even a single word exhibits a high similarity score (e.g., $0.95$) with the tag *medicine*, the tag receives $r_{\text{medicine}} = 0.95$, regardless of how low the similarities of all other $n-1$ words are.

---

### Question 5:
**Does a high max-pool score imply that the corresponding tag describes the dominant topic of the document? Explain.**
- **Answer:**  
  **No.** Max pooling captures *peak local semantic evidence*, not aggregate or global frequency. A single isolated occurrence of a word or closely related term can produce a max-pool score close to $1.0$, even if that topic represents only one passing sentence in a 400-word document. To measure document dominance, an average pooling or frequency-weighted pooling scheme (such as TF-IDF weighting) would be required.

---

### Question 6:
**For your own top-eight ranking, identify which tags were matched by a text word identical to the tag itself, and which were matched by a genuinely different word. What does the difference tell you about what cosine similarity over word embeddings is, and is not, capturing?**
- **Answer:**  
  All eight top-ranked tags — `campus`, `research`, `innovation`, `faculty`, `technology`, `collaboration`, `laboratory`, and `scholarship` — received a score of $1.0000$, each matched by an identical word appearing verbatim in the input text. None of the top eight were matched by a different, semantically related word.  
  This happened because the input text was collected from MAHE's own LinkedIn posts, which naturally contain the same vocabulary as the tag list. When a tag word appears exactly in the text, its embedding is identical to the tag's embedding, so $\cos(u, u) = 1$.  
  **What this tells us:**
  - **What cosine similarity captures:** Geometric closeness in the embedding space based on co-occurrence patterns in large corpora. When words are different but semantically related (e.g., *innovative* and *innovation*), it would still produce a high similarity score below 1.
  - **What cosine similarity does not capture:** It cannot distinguish between a meaningful thematic occurrence and an incidental mention of a word. An exact verbatim match always scores 1.0 regardless of whether the topic is central or peripheral in the document. In this result, the dominance of exact matches shows that the text is rich in domain vocabulary, not that max-pool ranking has measured topic depth.

---

### Question 7:
**If a concept is not represented by any of the 20 chosen tags, can the ranking algorithm introduce a new tag for that concept? Explain.**
- **Answer:**  
  **No.** The algorithm performs ranking over a fixed, closed vocabulary of tags defined in matrix $T \in \mathbb{R}^{20 \times 50}$. The similarity matrix $S$ has strictly $20$ columns corresponding to the predefined tags. The algorithm has no generative or discovery capability to synthesize new tags beyond the fixed 20 columns of $T$.

---

### Question 8:
**Why are out-of-vocabulary (OOV) tokens excluded from $W$?**
- **Answer:**  
  Because the embedding model maps only words within its fixed lookup dictionary (`key_to_index`) to vectors in $\mathbb{R}^{50}$. An OOV token (such as compound words `cuttingedge`, `realworld`, or rare proper nouns) has no associated coordinate representation in $\mathbb{R}^{50}$. Without an embedding vector, row operations and cosine similarities cannot be computed mathematically.

---

### Question 9:
**Explain the mathematical relationship between your pair-by-pair `Vector.cosine_similarity` computation and the NumPy matrix multiplication used to construct $S$.**
- **Answer:**  
  Both methods calculate the exact same mathematical quantity:
  $$S_{ij} = \frac{W_{i, :} \cdot T_{j, :}}{\|W_{i, :}\|_2 \|T_{j, :}\|_2}$$
  - In `Vector.cosine_similarity`, this scalar formula is evaluated individually for each pair $(i, j)$ inside two nested Python `for` loops, doing $20n$ separate normalizations and scalar dot products.
  - In NumPy, by factoring out row norms into diagonal scaling matrices $D_W = \operatorname{diag}(\|W_{i, :}\|_2^{-1})$ and $D_T = \operatorname{diag}(\|T_{j, :}\|_2^{-1})$, the operations are vectorized:
    $$\widehat{W} = D_W W, \quad \widehat{T} = D_T T \implies S = \widehat{W}\widehat{T}^T = (D_W W)(D_T T)^T$$
  Because matrix multiplication distributes dot products over rows and columns, both computations are mathematically identical, differing only by floating-point rounding precision ($\approx 10^{-7}$).
