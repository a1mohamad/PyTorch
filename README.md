# Learning Journey: Deep Learning & PyTorch Mastery

This repository serves as a comprehensive technical log of my progression through the deep learning curriculum of Sebastian Raschka's authoritative book, **"Machine Learning with PyTorch and Scikit-Learn."**

## 🎯 Project Motivation
Having established a strong foundation in **TensorFlow** and core **Machine Learning** methodologies in the past, I sought to expand my expertise into the **PyTorch** ecosystem. I selected this specific text because of its rigorous balance between mathematical theory and production-grade implementation. 

Given my prior background, I intentionally bypassed the introductory Scikit-Learn chapters and began my journey at **Chapter 11**, which serves as the gateway to advanced Deep Learning and the PyTorch framework.

---

## 🛤️ Core Learning Path: Chapters 11–19

The following sections detail the evolution of my technical skills, from basic tensor manipulation to sophisticated reinforcement learning agents.

### 🔹 I. PyTorch Fundamentals (Chapters 11–13)
The transition focused on shifting from static functional graphs to PyTorch’s **Dynamic Computational Graphs**.
* **Tensor Mechanics:** In-depth mastery of multi-dimensional array operations and GPU acceleration.
* **Autograd & nn.Module:** Implementation of custom neural network layers using automatic differentiation.
* **Data Pipelines:** Architecting efficient `Dataset` and `DataLoader` classes to handle large-scale data preprocessing and batching.



### 🔹 II. Advanced Neural Architectures (Chapters 14–16)
This phase shifted focus toward specialized architectures for computer vision and sequential data.
* **Convolutional Neural Networks (CNNs):** Designed deep architectures for image recognition, optimizing hyperparameters like kernel size, padding, and pooling layers.
* **Recurrent Neural Networks (RNNs & LSTMs):** Solved vanishing gradient problems in sequential data for time-series forecasting and NLP.
* **Transformers:** Studied the shift from recurrence to **Self-Attention mechanisms**, implementing the building blocks of modern Large Language Models.




### 🔹 III. Generative & Graph Modeling (Chapters 17–18)
Explored cutting-edge frontiers in deep learning beyond standard supervised tasks.
* **Generative Adversarial Networks (GANs):** Orchestrated the competition between Generator and Discriminator models to synthesize realistic images.
* **Graph Neural Networks (GNNs):** Applied deep learning to non-Euclidean data structures, such as social graphs and molecular modeling.



### 🔹 IV. Reinforcement Learning (Chapter 19)
The capstone of this journey involved building agents capable of making autonomous decisions in complex environments.
* **Theoretical Foundation:** Deep dive into the **Markov Decision Process (MDP)**, Bellman Equations, and the concepts of Rewards, States, and Actions.
* **Tabular Q-Learning:** Implementation of the first agent to solve a discrete **Grid World** environment by optimizing a Q-table through experience.
* **Deep Q-Learning (DQN):** Advanced to continuous control problems by using a Neural Network as a function approximator. Successfully solved the **Gymnasium CartPole-v1** environment using **Experience Replay** and **Target Networks** to stabilize the training process.



---

## 🛠️ Technical Stack
* **Frameworks:** PyTorch, Numpy, NetworkX, Transformers(Hugging Face), Gymnasium (OpenAI Gym).
* **Visualization:** Matplotlib, Pygame (for Grid World rendering).
* **Architecture:** Deep Neural Networks (MLP), CNNs, RNNs, LSTMs, Transformers, GANs, GNNs, and RL structures like DQNs.

## 🎓 Conclusion
By starting from Chapter 11, I have successfully bridged my existing knowledge of ML into the PyTorch framework. This journey has not only improved my coding proficiency but has also deepened my understanding of the mathematical optimizations required to build stable and efficient deep learning models.