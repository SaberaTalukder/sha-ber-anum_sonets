from nltk import pos_tag, word_tokenize
import nltk
import numpy as np
import matplotlib.pyplot as plt

def pos_vis(hmm, obs_map, title=None):
    
    nltk.download('averaged_perceptron_tagger')
    pos = pos_tag(list(sorted(obs_map.keys(), key=lambda k : obs_map[k])))
    pos = list(map(lambda p : p[1], pos))
    pos_voc = sorted(set(pos))
    pos2idx = {u:i for i, u in enumerate(pos_voc)}
    idx2pos = np.array(pos_voc)
    
    label_matrix = np.zeros((len(pos),len(pos_voc)))
    label_matrix[range(len(pos)),list(map(lambda p : pos2idx[p], pos))] = 1

    states_pos = np.array(hmms[0].O) @ label_matrix

    #plt.figure(figsize=(8,3))
    plt.imshow(states_pos)
    plt.xticks(ticks=range(len(pos_voc)),labels=pos_voc,rotation='vertical')
    plt.xlabel('Part of Speech')
    plt.ylabel('State')
    if title == None:
        title = f'{states_pos.shape[0]} States'
    
    plt.savefig(f'Part of Speech for {title}')
