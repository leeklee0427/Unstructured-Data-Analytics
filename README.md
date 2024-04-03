<a name="top"></a>



# 95-865: Unstructured Data Analytics



## Links
[95-865: Unstructured Data Analytics (Spring 2024 Mini 4)](https://www.andrew.cmu.edu/user/georgech/95-865/)


## Course Outline

### Part I: Exploratory data analysis
Identify structure present in "unstructured" data
- Frequency and co-occurrence analysis
- Visualizing high-dimensional data/dimensionality reduction
- Clustering
- Topic modeling

### Part II: Predictive data analysis
Make predictions using known structure in data
- Basic concepts and quality assessment of prediction models
- Neural networks and deep learning for analyzing texts and image



## Table of Contents
1. [Overview](#1-overview)
2. [Basic Text Analysis](#2-basic-text-analysis)
3. 
4. 
5. 
6. 
7. [Clustering](#7-clustering)


[Back to Top](#)

---


## 1. Overview

### Types of Data

1. **Structured data**: Well-defined elements, relationships between elements

2. **Unstructured data**: No pre-defined model; elements and relationships are ambiguous
    - Text
    - Images
    - Videos
    - Audio

### Unstructured Data Analysis
Solving a "murder mystery"
1. Question
2. Data/Evidence
3. Finding Structure: Exploratory data analysis
4. Insights: Answer original question


---


### Bag of Words (BoW)
- **Bag of Words (BoW)** model represents text data as a collection of words without considering the order.
    - Uses raw count frequency to represent the occurrence of words
- **Corpus**: a large and structured collection of text documents.
- **Collection Term Frequency (CTF)** refers to the frequency of a term across an entire collection of documents in a corpus.


#### Properties of Text
1. Split on blank spaces
2. Stopwords typically do not carry significant semantic meaning
    - Remove stopwords or not?


[Back to Top](#)



---



## 2. Basic Text Analysis


### NLP Tasks

For words in various forms, capitalization:
- **Lemmatization** reduces words to the base or canonical form, known as the lemma, while still preserving the meaning.

For words with multiple meanings:
- **Word Sense Disambiguation (WSD)** determines the correct meaning or sense of a word within a given context.

For named entities:
- **Named Entity Recognition (NER)** identifies and categorizes named entities within text data into predefined categories such as persons, organizations, locations, dates, quantities, and etc.

Other common tasks:
- **Tokenization** breaks down a text into smaller units, called tokens, which can be words, phrases, symbols, or other meaningful elements.

- **Part-of-speech (POS) tagging** assigns a part-of-speech tag to each word in a text based on syntactic role and grammatical category within a sentence, including nouns, verbs, adjectives, adverbs, pronouns, conjunctions, prepositions, and interjections.

- **Sentence boundary detection** (sentence recognition) determines where one sentence ends and the next begins in a given text.



### Bigram Model
**Bigram model** is a probabilistic language model predicts the probability of a word given the previous word in a sequence of words.
- 1 word at a time: Unigram model
- 3 words at a time: Trigram model
- n words at a time: n-gram model



### spaCy
[Jupyter notebook (basic text analysis)](https://gist.github.com/georgehc/932764d81cd246a60b85e28648cf05bc)

#### Load model
```python
import spacy
nlp = spacy.load('en_core_web_sm')  # Load spaCy's built-in English tokenizer, tagger, parser, NER, and word vectors
```

#### Processing
```python
parsed_text = nlp(text)
```

#### Display tokens
```python
for token in parsed_text:
    """
    Display the token's orthographic representation, lemma, part of speech, and entity type (empty if not part of a named entity)
    """
    print(token, token.lemma_, token.pos_, token.ent_type_)
```
- Including `_` means the string representation
- After the foreach loop, `token` will point to the last element from the parsed text, which is a period (`.`)


#### Iterate through the named entities
```python
for entity in parsed_text.ents:
    print(entity, entity.label_)
```

#### Iterate through the sentences
```python
idx = 0
for sentence in parsed_text.sents:
    print('Sentence number', idx, ':', sentence)
    idx += 1


for idx, sentence in enumerate(parsed_text.sents):
    print('Sentence number', idx, ':', sentence)
```

#### Tokens and counts

```python
histogram = {}
for token in parsed_text:
    if token.orth_ not in histogram:
        histogram[token.orth_] = 1
    else:
        histogram[token.orth_] += 1

# Convert to a list of tuples
list(histogram.items())

from operator import itemgetter
sorted_token_count_pairs = sorted(histogram.items(),
                                  reverse=True, # Sort in reverse order (largest to smallest)
                                  key=itemgetter(1)) # lambda x: x[1]
```

Alternative: `Counter.most_common()`

```python
from collections import Counter

histogram = Counter()
for token in parsed_text:
    histogram[token.orth_] += 1

sorted_token_count_pairs = histogram.most_common()
for token, count in sorted_token_count_pairs:
    print(token, ":", count)
```


#### Remove stopwords, punctuations, and spaces
```python
from collections import Counter

histogram_with_filtering = Counter()
for token in parsed_text:
    lemma = token.lemma_.lower() # Convert lemma to lowercase
    # Ignore junk
    if not (nlp.vocab[lemma].is_stop or token.pos_ == 'PUNCT' or token.pos_ == 'SPACE' or token.pos_ == 'X'):
        histogram_with_filtering[lemma] += 1

sorted_lemma_count_pairs = histogram_with_filtering.most_common()
for lemma, count in sorted_lemma_count_pairs:
    print(lemma, ":", count)
```


#### Manually remove stopwords
```python
manual_stop_words = {'jump', 'b', '-', 'c'} # Create a set

# Keep lemma that are not the stopwords
histogram_filtered_twice = Counter({lemma: count
                                    for lemma, count in histogram_with_some_filtering.items()
                                    if lemma not in manual_stop_words})

twice_filtered_sorted_lemma_count_pairs = histogram_filtered_twice.most_common()
for lemma, count in twice_filtered_sorted_lemma_count_pairs:
    print(lemma, ":", count)
```

#### Plot the top 20 most frequently occurring lemmas
```python
%matplotlib inline
import matplotlib.pyplot as plt

num_top_lemmas_to_plot = 20
top_lemmas = [lemma for lemma, count in twice_filtered_sorted_lemma_count_pairs[:num_top_lemmas_to_plot]]
top_counts = [count for lemma, count in twice_filtered_sorted_lemma_count_pairs[:num_top_lemmas_to_plot]]
plt.bar(range(num_top_lemmas_to_plot), top_counts)
plt.xticks(range(num_top_lemmas_to_plot), top_lemmas, rotation=90)
plt.xlabel('Lemma')
plt.ylabel('Raw count')
```


### Summary
- Represent each document as a histogram/probability distribution
- Feature vector: vector representation of the document
    - Feature vectors are high-dimensional
    - Dimensions = number of terms








## 3. Co-occurrence Analysis












[Back to Top](#)
---









## 4.






[Back to Top](#)



---





## 5. PCA






[Back to Top](#)



---








## 6. Manifold learning





[Back to Top](#)



---












## 7. Clustering

### Overview

- **Clustering** is a method of unsupervised learning, a type of machine learning where the system learns to **identify patterns without prior labeling of the data**.

- Clustering methods aim to group together data points that are "similar" into "clusters", while having different clusters be "dissimilar".
    - Similarity is inversely related to distance (two points being more similar $\rightarrow$ closer in distance)
    - Use **Euclidean distance** between feature vectors

- Clustering structure often occurs
    - Crime happens more often in specific spots
    - Users share similar tastes in a recommendation system
    - 2-D t-SNE plot of handwritten digit images shows clumps that correspond to real digits


---


### Drug Consumption Data
Source: https://archive.ics.uci.edu/dataset/373/drug+consumption+quantified

#### Demo


---

### Similarity/Distance Functions

1. **Euclidean distance** between feature vectors

2. **Levenshtein distance** (edit distance): the minimum number of single-letter insertions, deletions, or substitutions required to convert one string into another.
    - `kitten` and `sitting` has a Levenshtein distance of 3


---




### Effectiveness Assessment of Similarity/Distance Functions

- Step 1: Select a data point, which can be chosen randomly or deliberately.

- Step 2: Compute the similarity or distance between the selected data point and all other data points in the dataset and sort the data points based on their similarity or distance, from most similar to least similar (or smallest distance to largest).

- Step 3: **Manually examine** the data points that are most similar or closest to the selected point by inspecting their raw data.
    - <font color='red'>Similarity/distance functions is likely not good if the most similar/closest points are not interpretable.</font>


---



### Clustering Methods


| Generative Models | Hierarchical Clustering |
|-------------------|-------------------------|
| 1. Pretend data generated by specific model with parameters. <br> 2. Learn the parameters ("fit model to data"). <br> 3. Use fitted model to determine cluster assignments. | **Top-down**: Start with everything in 1 cluster and decide on how to recursively split. <br> **Bottom-up**: Start with everything in $n$ cluster and decide on how to iteratively merge. |
| - | <font color='red'>Requires certain termination criteria.</font> |


---


### K-Means Clustering

1. **Initialization**: Start by selecting $k$ initial centroids randomly.
    - One common approach is to randomly choose $k$ data points from the dataset as the initial centroids.

2. **Assignment**: Assign each data point to the nearest centroid.
    - The most common distance metric used is the Euclidean distance.

3. **Update**: Recalculate the centroids of the clusters by taking the mean of all the data points assigned to each cluster.
    - The mean (center of mass) becomes the new centroid of each cluster.

4. **Iteration**: Repeat the Assignment and Update steps until one of the following conditions is met:
    - The centroids do not change (or below a certain threshold), indicating convergence.
    - The assignments of data points to clusters remain the same between iterations.
    - A predefined number of iterations has been reached.

5. **Termination**: Return the centroids of the clusters and the assignments of each data point to a cluster.


---



### K-Means++

- K-Means++ uses a **weighted probability distribution** for the initialization step to improve the convergence of the K-Means algorithm, leading to better clustering outcomes and faster convergence.

- After choosing the first centroid randomly, K-Means++ selects subsequent centroids from the remaining data points with a probability proportional to the square of the distance from each point to the nearest existing centroid.
    - <font color='red'>This step biases the selection towards points that are the furthest from the existing centroids, aiming to spread out the initial centroids.</font>


---


### Gaussian Mixture Model (GMM)

- GMM is a **probabilistic** (not deterministic) model that assumes all the data points are generated from a mixture of a finite number of Gaussian distributions with unknown parameters. 

- Data Assignment (Soft Clustering): Given a new data point, a GMM can calculate the probability of it belonging to each of the Gaussian components in the mixture.

#### General Case
- GMM is the sum of $k$ different $d$-dimensional Gaussian distributions.
    - The overall probability distribution looks like $k$ mountains
    - Each mountain corresponds to a different cluster
    - Different mountains can have different peak heights
    - Different mountains can have different ellipse shapes that captures correlation/covariance information


#### Learning a GMM

- Step 0: Guess $k$

- Step 1: Guess cluster probabilities, means, and covariances

Repeat until convergence:

- Compute probability of each point being in each of the $k$ clusters

- Update cluster probabilities, means, and covariances accounting for probabilities of each point belonging to each of the clusters


#### Limitations
1. In reality, data points are unlikely generated the same way!
2. In reality, data points might not be independent!


#### Caveat
"All models are wrong, but some are useful."

- Models are approximations/simplifications of the reality.
- Some models provide insights, make predictions, or enable decisions that are sufficiently accurate for practical purposes.