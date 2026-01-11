# Vision-to-language model
Week 1: Foundations of Computer Vision 
Topics Covered:

Image preprocessing and augmentation
Convolutional Neural Networks (CNNs)
Transfer learning and feature extraction
DenseNet architecture

Assignment:
Part 1: DenseNet-121 feature extraction on CIFAR-10
Froze all convolutional layers
Trained custom classifier (1024 → 512 → 10)
Achieved ~85-90% test accuracy
Applied data augmentation (flips, crops)
Used BatchNorm and Dropout for regularization

Part 2: Deep Neural Network (No CNN) on MNIST
Built 3-layer fully connected network (784 → 512 → 256 → 128 → 10)
Achieved ~97-98% test accuracy
Implemented regularization (Dropout, BatchNorm, data augmentation)
Compared performance: transfer learning vs training from scratch



Key Learnings:

Transfer learning dramatically reduces training time and improves accuracy
Freezing pretrained layers prevents catastrophic forgetting
Proper normalization (ImageNet stats) is critical for pretrained models
Regularization techniques (Dropout, BatchNorm) effectively prevent overfitting

Technologies: PyTorch, torchvision, sklearn, matplotlib

Week 2: Foundations of Natural Language Processing 
Topics Covered:

Text preprocessing pipeline (tokenization, lemmatization, stopword removal)
Word embeddings (Word2Vec, GloVe)
Sentence embeddings (Avg Word2Vec, BERT)
Recurrent Neural Networks (RNN, LSTM, GRU)
Transformer architecture
Hugging Face ecosystem

Assignment:

1: Twitter Sentiment Analysis with Word2Vec

Preprocessed 10K+ tweets (lowercase, URL removal, contraction expansion, lemmatization)
Loaded Google News Word2Vec model (300-dimensional vectors)
Implemented Avg Word2Vec for sentence embeddings
Trained Multiclass Logistic Regression
Achieved ~75-85% test accuracy
Created predict_tweet_sentiment() function for inference

2: BERT Fine-tuning with Hugging Face

Fine-tuned bert-base-uncased on IMDb dataset (50K reviews)
Implemented complete training pipeline using Hugging Face Trainer API
Achieved ~90-93% test accuracy, F1 score: ~0.91
Applied early stopping and learning rate scheduling
Saved and deployed fine-tuned model for inference
Used mixed precision training (fp16) for efficiency



Key Learnings:

Word2Vec captures semantic relationships (king - man + woman ≈ queen)
BERT's bidirectional context understanding outperforms traditional methods
Transfer learning applies to NLP: pretrained models save massive compute
Hugging Face ecosystem simplifies model loading, training, and deployment
Text preprocessing significantly impacts model performance

Technologies: NLTK, Gensim, Transformers, Datasets, PyTorch

Week 3: Explainable AI (XAI) for Vision Models 
Topics Covered:

Vision Transformers (ViT) introduction
Encoder-decoder architectures for multimodal tasks
Attention mechanisms
Explainability techniques (Grad-CAM, feature visualization)

Assignment:

1: Vision Model Setup

Loaded pretrained ResNet-50
Fine-tuned on CIFAR-10 classification
Achieved high accuracy with minimal training


2: Filter Visualization

Visualized first layer filters (edge/color detectors)
Visualized deeper layer filters (complex patterns)
Analyzed filter complexity vs network depth

3: Feature Map Visualization

Extracted intermediate layer activations
Visualized 16 different channel responses
Analyzed how channels specialize in different patterns

4: Grad-CAM Implementation

Implemented Gradient-weighted Class Activation Mapping from scratch
Generated heatmaps for correctly classified images
Overlayed attention maps on original images
Validated that model focuses on discriminative regions

5: Failure Case Analysis

Identified misclassified examples
Applied Grad-CAM to understand failure modes
Revealed when model focuses on wrong regions
Identified systematic errors and biases



Key Learnings:

CNNs learn hierarchical features: edges → textures → objects
Different channels specialize in detecting different patterns
Grad-CAM effectively explains CNN predictions
XAI reveals both strengths and weaknesses of models
Explainability is crucial for debugging and building trust

Technologies: PyTorch, OpenCV, Matplotlib, Grad-CAM





NLP Libraries:

NLTK - Text preprocessing
Gensim - Word2Vec embeddings
scikit-learn - Classical ML algorithms

Visualization & Analysis:

Matplotlib - Plotting and visualization
Seaborn - Statistical visualizations
OpenCV - Image processing
tqdm - Progress bars

Data & Datasets:

CIFAR-10 (50K images, 10 classes)
MNIST (60K handwritten digits)
Twitter US Airline Sentiment (~14K tweets)
IMDb Reviews (50K movie reviews)
