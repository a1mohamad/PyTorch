# 📖 Chapter 15: Processing Sequential Data with Recurrent Neural Networks (RNNs)

This comprehensive README summarizes the core concepts and practical applications covered in **Chapter 15** of *Machine Learning with PyTorch and Scikit-Learn*. This chapter focuses on neural networks designed specifically to handle **sequential data** (like text, time series, and speech) by incorporating an internal **memory** mechanism.

---

## 🎯 Chapter Overview: Introducing Neural Memory

The primary challenge addressed in this chapter is that previous architectures (MLPs and CNNs) treat every input as independent. RNNs solve this by maintaining a **hidden state ($h_t$)** that carries information about all prior time steps, allowing the network to understand context and dependencies over time.

| Focus Area |
| :--- | :--- | :--- |
| **Mechanics** | Recurrence | Learning temporal dependencies and memory retention over time steps. |
| **Architecture** | LSTMs/GRUs | Solving the "vanishing gradient" problem for long-range dependencies. |
| **Data Prep** | Embedding & Packing | Converting tokens (words/characters) into dense vectors and handling variable sequence lengths. |
| **Applications** | Classification & Generation | Distinguishing between passive prediction (Sentiment Analysis) and active content creation (Language Modeling). |

---

## 🛠️ Section 1: The Core RNN Mechanics # 📖 Chapter 15: Processing Sequential Data with Recurrent Neural Networks (RNNs)

This comprehensive README summarizes the core concepts and practical applications covered in **Chapter 15** of *Machine Learning with PyTorch and Scikit-Learn*. This chapter focuses on neural networks designed specifically to handle **sequential data** (like text, time series, and speech) by incorporating an internal **memory** mechanism.

---

## 🎯 Chapter Overview: Introducing Neural Memory

The primary challenge addressed in this chapter is that previous architectures (MLPs and CNNs) treat every input as independent. RNNs solve this by maintaining a **hidden state ($h_t$)** that carries information about all prior time steps, allowing the network to understand context and dependencies over time.

| Focus Area | Core Concept | Significance |
| :--- | :--- | :--- |
| **Mechanics** | Recurrence | Learning temporal dependencies and memory retention over time steps. |
| **Architecture** | LSTMs/GRUs | Solving the "vanishing gradient" problem for long-range dependencies. |
| **Data Prep** | Embedding & Packing | Converting tokens (words/characters) into dense vectors and handling variable sequence lengths. |
| **Applications** | Classification & Generation | Distinguishing between passive prediction (Sentiment Analysis) and active content creation (Language Modeling). |

---

## 🛠️ Section 1: The Core RNN Mechanics (Notebook: `RNN_basics.ipynb`)

This section establishes the foundational mathematics and PyTorch implementation of a simple RNN layer.

### 1. The Recurrent Step

The fundamental mechanism is the calculation of the current hidden state ($h_t$), which is a function of the current input ($x_t$) and the hidden state from the previous time step ($h_{t-1}$).

* **RNN Formula:** The core state update is often defined as $h_t = \tanh(W_{xh}x_t + W_{hh}h_{t-1} + b)$.
* **Unrolling:** The notebook demonstrates **unrolling** the RNN manually, showing that the model is applied sequentially to each element of the input sequence. 
* **Tensors and `batch_first`:** The inputs are structured as a 3D Tensor: **(Batch Size, Sequence Length, Feature Size)**. The `batch_first=True` parameter is confirmed as the standard PyTorch convention.

### 2. Manual Inspection

The notebook inspects the four key weight tensors that govern the RNN's behavior: `weight_ih_l0` (Input-to-Hidden weights, $W_{xh}$) and `weight_hh_l0` (Hidden-to-Hidden weights, $W_{hh}$), confirming a direct understanding of the layer's internal parameters.

---

## 🧠 Section 2: Solving the Memory Problem with LSTMs

Simple RNNs struggle with the **vanishing gradient problem**, making them unable to learn dependencies that span many time steps (e.g., in a long review). The solution is Gated RNNs.

### 1. Long Short-Term Memory (LSTM)

* **Architecture:** The **LSTM** is the workhorse of modern sequence processing (used in the `IMDB.ipynb` and `Character_level_language_model.ipynb` notebooks).
* **The Cell State ($C_t$):** LSTMs introduce a separate **Cell State**—a dedicated path for carrying long-term memory down the sequence.
* **Gating Mechanism:** LSTMs use three specialized gates to control the flow of information: **Forget Gate** (decides what information to discard), **Input Gate** (decides what new information to store), and **Output Gate** (decides what to expose as the new hidden state). 

### 2. Bidirectional LSTMs

* **Context:** The `IMDB.ipynb` uses a **Bidirectional LSTM**, where the sequence is processed simultaneously by two LSTMs: one running forward and one running backward.
* **Benefit:** This allows the model to capture context from both **past** (causal) and **future** information in the sequence, which is highly beneficial for text classification tasks where the sentiment of a word might depend on what follows it.

---

## 📝 Section 3: Text Data Preprocessing for RNNs

All text applications require specialized preprocessing to convert sparse text into dense numerical features.

* **Tokenization & Vocabulary:** Text is broken down into tokens (words or characters), and a vocabulary maps these tokens to unique integer indices.
* **Embedding Layer (`nn.Embedding`):** The numerical indices are converted into **dense word vectors**. This step is crucial because it allows the model to learn semantic and syntactic similarities between words.
* **Handling Variable Lengths:**
    * **Padding:** Sequences are padded to the maximum length in a batch.
    * **Packing Utility:** The critical `nn.utils.rnn.pack_padded_sequence` utility (seen in `IMDB.ipynb`) is used to tell the RNN to **ignore padding tokens** during computation, ensuring accurate gradients and saving computation time.

---

## 🚀 Section 4: Applications in Prediction and Generation

Chapter 15 demonstrates two major categories of sequence modeling tasks.

### 1. Sentiment Classification (Notebook: `Sentiment_analysis_on_IMDB_with_LSTMs.ipynb`)

* **Task Type:** Sequence-to-Label (Classification).
* **Workflow:** The Bidirectional LSTM processes the entire movie review, and the final combined hidden state (representing the review's summary) is fed into a final fully connected layer for **Binary Classification** (positive/negative sentiment) using **`nn.BCEWithLogitsLoss`**.

### 2. Character-Level Language Modeling (Notebook: `Character_level_language_model.ipynb`)

* **Task Type:** Sequence-to-Sequence (Generation).
* **Workflow:** The model is trained to predict the **next character** in a sequence.
* **Output:** The final layer has $V$ outputs (where $V$ is the size of the character vocabulary), and **`nn.CrossEntropyLoss`** is used.
* **Generative Inference:** The trained model is used recursively: a predicted character is **sampled** from the output probability distribution and immediately fed back as the input for the next time step. This process creates new, novel text.
* **Temperature Sampling:** The notebook introduces **temperature** as a method to control the randomness (creativity) of the generated text.

This section establishes the foundational mathematics and PyTorch implementation of a simple RNN layer.

### 1. The Recurrent Step

The fundamental mechanism is the calculation of the current hidden state ($h_t$), which is a function of the current input ($x_t$) and the hidden state from the previous time step ($h_{t-1}$).

* **RNN Formula:** The core state update is often defined as $h_t = \tanh(W_{xh}x_t + W_{hh}h_{t-1} + b)$.
* **Unrolling:** The notebook demonstrates **unrolling** the RNN manually, showing that the model is applied sequentially to each element of the input sequence. 
* **Tensors and `batch_first`:** The inputs are structured as a 3D Tensor: **(Batch Size, Sequence Length, Feature Size)**. The `batch_first=True` parameter is confirmed as the standard PyTorch convention.

### 2. Manual Inspection

The notebook inspects the four key weight tensors that govern the RNN's behavior: `weight_ih_l0` (Input-to-Hidden weights, $W_{xh}$) and `weight_hh_l0` (Hidden-to-Hidden weights, $W_{hh}$), confirming a direct understanding of the layer's internal parameters.

---

## 🧠 Section 2: Solving the Memory Problem with LSTMs

Simple RNNs struggle with the **vanishing gradient problem**, making them unable to learn dependencies that span many time steps (e.g., in a long review). The solution is Gated RNNs.

### 1. Long Short-Term Memory (LSTM)

* **Architecture:** The **LSTM** is the workhorse of modern sequence processing (used in the `IMDB.ipynb` and `Character_level_language_model.ipynb` notebooks).
* **The Cell State ($C_t$):** LSTMs introduce a separate **Cell State**—a dedicated path for carrying long-term memory down the sequence.
* **Gating Mechanism:** LSTMs use three specialized gates to control the flow of information: **Forget Gate** (decides what information to discard), **Input Gate** (decides what new information to store), and **Output Gate** (decides what to expose as the new hidden state). 

### 2. Bidirectional LSTMs

* **Context:** The `IMDB.ipynb` uses a **Bidirectional LSTM**, where the sequence is processed simultaneously by two LSTMs: one running forward and one running backward.
* **Benefit:** This allows the model to capture context from both **past** (causal) and **future** information in the sequence, which is highly beneficial for text classification tasks where the sentiment of a word might depend on what follows it.

---

## 📝 Section 3: Text Data Preprocessing for RNNs

All text applications require specialized preprocessing to convert sparse text into dense numerical features.

* **Tokenization & Vocabulary:** Text is broken down into tokens (words or characters), and a vocabulary maps these tokens to unique integer indices.
* **Embedding Layer (`nn.Embedding`):** The numerical indices are converted into **dense word vectors**. This step is crucial because it allows the model to learn semantic and syntactic similarities between words.
* **Handling Variable Lengths:**
    * **Padding:** Sequences are padded to the maximum length in a batch.
    * **Packing Utility:** The critical `nn.utils.rnn.pack_padded_sequence` utility (seen in `IMDB.ipynb`) is used to tell the RNN to **ignore padding tokens** during computation, ensuring accurate gradients and saving computation time.

---

## 🚀 Section 4: Applications in Prediction and Generation

Chapter 15 demonstrates two major categories of sequence modeling tasks.

### 1. Sentiment Classification (Notebook: `Sentiment_analysis_on_IMDB_with_LSTMs.ipynb`)

* **Task Type:** Sequence-to-Label (Classification).
* **Workflow:** The Bidirectional LSTM processes the entire movie review, and the final combined hidden state (representing the review's summary) is fed into a final fully connected layer for **Binary Classification** (positive/negative sentiment) using **`nn.BCEWithLogitsLoss`**.

### 2. Character-Level Language Modeling (Notebook: `Character_level_language_model.ipynb`)

* **Task Type:** Sequence-to-Sequence (Generation).
* **Workflow:** The model is trained to predict the **next character** in a sequence.
* **Output:** The final layer has $V$ outputs (where $V$ is the size of the character vocabulary), and **`nn.CrossEntropyLoss`** is used.
* **Generative Inference:** The trained model is used recursively: a predicted character is **sampled** from the output probability distribution and immediately fed back as the input for the next time step. This process creates new, novel text.
* **Temperature Sampling:** The notebook introduces **temperature** as a method to control the randomness (creativity) of the generated text.