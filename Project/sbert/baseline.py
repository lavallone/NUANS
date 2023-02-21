import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.special import softmax
import logging
import pytorch_lightning as pl
from sentence_transformers import SentenceTransformer, util

"""
LexRank implementation
Source: https://github.com/crabcamp/lexrank/tree/dev
"""

# AUX FUNCTIONS
#################################################################################################################
def _power_method(transition_matrix, increase_power=True, max_iter=10000):
    eigenvector = np.ones(len(transition_matrix))
    if len(eigenvector) == 1:
        return eigenvector
    transition = transition_matrix.transpose()
    for _ in range(max_iter):
        eigenvector_next = np.dot(transition, eigenvector)
        if np.allclose(eigenvector_next, eigenvector):
            return eigenvector_next
        eigenvector = eigenvector_next
        if increase_power:
            transition = np.dot(transition, transition)
    logger = logging.getLogger(__name__)
    logger.warning("Maximum number of iterations for power method exceeded without convergence!")
    return eigenvector_next

def connected_nodes(matrix):
    _, labels = connected_components(matrix)
    groups = []
    for tag in np.unique(labels):
        group = np.where(labels == tag)[0]
        groups.append(group)
    return groups

def stationary_distribution(transition_matrix, increase_power=True, normalized=True,):
    n_1, n_2 = transition_matrix.shape
    if n_1 != n_2:
        raise ValueError('\'transition_matrix\' should be square')
    distribution = np.zeros(n_1)
    grouped_indices = connected_nodes(transition_matrix)
    for group in grouped_indices:
        t_matrix = transition_matrix[np.ix_(group, group)]
        eigenvector = _power_method(t_matrix, increase_power=increase_power)
        distribution[group] = eigenvector
    if normalized:
        distribution /= n_1
    return distribution

def create_markov_matrix(weights_matrix):
    n_1, n_2 = weights_matrix.shape
    if n_1 != n_2:
        raise ValueError('\'weights_matrix\' should be square')
    row_sum = weights_matrix.sum(axis=1, keepdims=True)
    # normalize probability distribution differently if we have negative transition values
    if np.min(weights_matrix) <= 0:
        return softmax(weights_matrix, axis=1)
    return weights_matrix / row_sum

def create_markov_matrix_discrete(weights_matrix, threshold):
    discrete_weights_matrix = np.zeros(weights_matrix.shape)
    ixs = np.where(weights_matrix >= threshold)
    discrete_weights_matrix[ixs] = 1
    return create_markov_matrix(discrete_weights_matrix)
#################################################################################################################

def degree_centrality_scores(similarity_matrix, threshold=None, increase_power=True,):
    if not (threshold is None or isinstance(threshold, float) and 0 <= threshold < 1):
        raise ValueError('\'threshold\' should be a floating-point number ' 'from the interval [0, 1) or None',)
    if threshold is None:
        markov_matrix = create_markov_matrix(similarity_matrix)
    else:
        markov_matrix = create_markov_matrix_discrete(similarity_matrix, threshold,)
    scores = stationary_distribution(markov_matrix, increase_power=increase_power, normalized=False,)
    return scores


# this is the BASELINE model used both for performance comparison and for the extraction of summary candidates
class SentenceBERT(pl.LightningModule):
    def __init__(self, hparams, predictions = None):
        super(SentenceBERT, self).__init__()
        self.save_hyperparameters(hparams)
        self.predictions = predictions
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def forward(self, texts):
        ris = {}
        # iterate over all the texts
        for name, text in texts.items():

            # for each story we compute the length of its extractive summary depending on its original length!
            original_length = len(text)
            print("Num sentences:", original_length)
            if self.hparams.sbert_mode == "extraction":
                # we use the output length predictions made by the trained regression model
                summary_length = self.predictions[name] + self.hparams.length_conf_int
            elif self.hparams.sbert_mode == "evaluation":
                summary_length = int(len(text)*0.3)

            # compute the sentence embeddings
            embeddings = self.model.encode(text, convert_to_tensor=True)

            # compute the pair-wise cosine similarities
            cos_scores = util.cos_sim(embeddings, embeddings).cpu().numpy()

            # compute the centrality for each sentence
            centrality_scores = degree_centrality_scores(cos_scores, threshold=None)
            #print(np.abs(np.sort(-centrality_scores)))
            
            # we 'argsort' so that the first element is the sentence with the highest score
            most_central_sentence_indices = np.argsort(-centrality_scores)
            most_central_sentence_indices = most_central_sentence_indices[0:summary_length].tolist()
            most_central_sentence_indices.sort()
            ris[name] = most_central_sentence_indices
            
        return ris