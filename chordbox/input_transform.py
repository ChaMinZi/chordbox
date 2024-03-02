import vamp
import librosa
import torch
import numpy as np
from model import *
# import librosa.display
# import matplotlib.pyplot as plt


def int2chord(chord_id):
    table = {0:'C:maj', 1:'C#:maj', 2:'D:maj', 3:'D#:maj', 4:'E:maj', 5:'F:maj', 6:'F#:maj', 7:'G:maj', 8:'G#:maj', 9:'A:maj', 10:'A#:maj', 11:'B:maj',
             12:'C:min', 13:'C#:min', 14:'D:min', 15:'D#:min', 16:'E:min', 17:'F:min', 18:'F#:min', 19:'G:min', 20:'G#:min', 21:'A:min', 22:'A#:min', 23:'B:min',
             24:'N', 25:'X'}
    return table[chord_id]

def idxes_of_change(chord_list):
    idxes_of_change = [0]
    c_prev = chord_list[0]
    for i, c in enumerate(chord_list):
        if c!=c_prev:
            idxes_of_change.append(i)
        c_prev = c
    return idxes_of_change
        
def get_chromagram(filepath):
    
    data, rate = librosa.load(filepath)
    chroma = vamp.collect(data, rate, "nnls-chroma:nnls-chroma", output="bothchroma")
    stepsize, chromadata = chroma["matrix"]
    
    segment_width = 21
    seg_n_pad = segment_width//2
    n_frames = chromadata.shape[0]
    segment_hop = 5
    n_steps = 100
    
    # [time + 2*seg_n_pad, 24]
    chroma_pad = np.pad(chromadata, [(seg_n_pad, seg_n_pad), (0, 0)], 'constant', constant_values=0.0) 
    # [n_segments, segment_width, 24]
    chroma_segment = np.array([chroma_pad[i-seg_n_pad:i+seg_n_pad+1] for i in range(seg_n_pad, seg_n_pad + n_frames, segment_hop)])
    # [n_segments, segment_width*24]
    chroma = np.reshape(chroma_segment[:, :, :], [-1, segment_width*24])
    
    n_frames = chroma.shape[0]
    n_pad = 0 if n_frames/n_steps == 0 else n_steps - (n_frames % n_steps)
    if n_pad != 0: # chek if need paddings
        chroma = np.pad(chroma, [(0, n_pad), (0, 0)], 'constant', constant_values=0)  ## padding at the end
        
    seq_hop = n_steps
    n_sequences = int((chroma.shape[0] - n_steps) / seq_hop) + 1
    _, feature_size = chroma.shape

    s0, s1 = chroma.strides
    chroma_reshape = np.lib.stride_tricks.as_strided(chroma, shape=(n_sequences, n_steps, feature_size), strides=(s0 * seq_hop, s0, s1))
    
    return n_pad, stepsize, segment_hop, chroma_reshape
    
    
def predict(n_pad, stepsize, segment_hop, chroma_reshape):
    
    device = torch.device('cpu')
    model_dir = '/home/chordbox2021/chordbox/ChordBox2_statedict.pt'
    model = ChordTransformer(n_classes=26, device=device)
    model.load_state_dict(torch.load(model_dir, map_location=device))
    model.eval()
    
    chroma_reshape = torch.tensor(chroma_reshape)
    S, T, F = chroma_reshape.shape
    pred_chords = torch.empty((S,T))
    
    for s in range(S):
        seg = chroma_reshape[s,:,:].unsqueeze(0)
        with torch.no_grad():
            pred = model(seg)
            pred_chords[s] = pred['o_dec']

    valid_pred_chords = pred_chords.flatten()[:-n_pad]
    valid_pred_chords = [int2chord(int(x)) for x in valid_pred_chords]
    
    change_idxes = idxes_of_change(valid_pred_chords)
    timestamps = torch.empty((S,T))
    
    for s in range(S):
        for t in range(T):
            timestamps[s,t] = float(stepsize)*segment_hop*(100*s+t)

    valid_timestamps = timestamps.flatten()[:-n_pad]
    
    result_chords = ""
    result_times = ""
    
    for i, idx in enumerate(change_idxes):
        chord = valid_pred_chords[idx]
        time_start = valid_timestamps[idx]
        if i+1 == len(change_idxes):
            time_end = valid_timestamps[-1]
        else:
            time_end = valid_timestamps[change_idxes[i+1]]
        
        result_times += f' {time_start:.1f}'
        result_chords += f' {chord}'
        
        # return result_times, result_chords
        # print(f'{time_start} ~ {time_end} : {chord}')
    print(result_times[1:])
    print(result_chords[1:])
    
    return result_times[1:], result_chords[1:]

    
