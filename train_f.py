import torch
from torch import nn
from torch import optim
from torch.autograd import Variable

import numpy as np

import os
import sys
import time
import optparse

import copy, itertools, collections
from pretty_midi import PrettyMIDI, Note, Instrument

STATE_RESOLUTION = 220
STATE_TEMP = 120
STATE_VELOCITY = 52
STATE_PITCH_RANGE = range(21, 109)
STATE_VELOCITY_RANGE = range(21, 109)

BEAT_LENGTH = 60 / STATE_TEMP
STATE_TIME_SHIFT_BINS = 1.15 ** np.arange(32) / 65
STATE_VELOCITY_STEPS = 32
STATE_NOTE_LENGTH = BEAT_LENGTH * 2
MIN_NOTE_LENGTH = BEAT_LENGTH / 2


STATE_WINDOW_SIZE = BEAT_LENGTH * 4
STATE_NOTE_DENSITY_BINS = np.arange(12) * 3 + 1


class Note_Seqce:
    def __init__(self, notes=[]):
        self.notes = []
        if notes:
            for note in notes:
                notes = filter(lambda note: note.end >= note.start, notes)
            self.push_notes(list(notes))
    
    def copy(self):
        return copy.deepcopy(self)

    def convert2midi(self):
        midi = PrettyMIDI(resolution=STATE_RESOLUTION, initial_tempo=STATE_TEMP)
        inst = Instrument(1, False, 'Note_Seqce')
        inst.notes = copy.deepcopy(self.notes)
        midi.instruments.append(inst)
        return midi
    
    def midi2noteseq(midi, programs=DEFAULT_LOADING_PROGRAMS):
        notes = itertools.chain(*[
            inst.notes for inst in midi.instruments
            if inst.program in programs and not inst.is_drum])
        return NoteSeq(list(notes))

    @staticmethod
    def midi2noteseq_file(path, *kargs, **kwargs):
        midi = PrettyMIDI(path)
        return NoteSeq.midi2noteseq(midi, *kargs, **kwargs)


    def convert2midi_file(self, path, *kargs, **kwargs):
        self.convert2midi(*kargs, **kwargs).write(path)


    def push_notes(self, notes):
        self.notes += notes
        self.notes.sort(key=lambda note: note.start)



class Event:

    def __init__(self, type, time, value):
        self.type = type
        self.time = time
        self.value = value
    
    def __repr__(self):
        return 'Event(type={}, time={}, value={})'.format(
            self.type, self.time, self.value)


class Event_Seqce:

    pitch_range = STATE_PITCH_RANGE
    velocity_range = STATE_VELOCITY_RANGE
    velocity_steps = STATE_VELOCITY_STEPS
    time_shift_bins = STATE_TIME_SHIFT_BINS
    
    
    @staticmethod
    def take_from_array(event_indeces):
        time = 0
        events = []
        for event_index in event_indeces:
            for event_type, feat_range in Event_Seqce.featur_ranges().items():
                if feat_range.start <= event_index < feat_range.stop:
                    event_value = event_index - feat_range.start
                    events.append(Event(event_type, time, event_value))
                    if event_type == 'time_shift':
                        time += Event_Seqce.time_shift_bins[event_value]
                    break

        return Event_Seqce(events)

    @staticmethod
    def dim():
        return sum(Event_Seqce.featur_dimen().values())

    @staticmethod
    def featur_dimen():
        featur_dimen = collections.OrderedDict()
        featur_dimen['note_on'] = len(Event_Seqce.pitch_range)
        featur_dimen['note_off'] = len(Event_Seqce.pitch_range)
        featur_dimen['velocity'] = Event_Seqce.velocity_steps
        featur_dimen['time_shift'] = len(Event_Seqce.time_shift_bins)
        return featur_dimen

    @staticmethod
    def featur_ranges():
        offset = 0
        featur_ranges = collections.OrderedDict()
        for feat_name, feat_dim in Event_Seqce.featur_dimen().items():
            featur_ranges[feat_name] = range(offset, offset + feat_dim)
            offset += feat_dim
        return featur_ranges

    @staticmethod
    def getting_veloc_basket():
        n = Event_Seqce.velocity_range.stop - Event_Seqce.velocity_range.start
        return np.arange(
                Event_Seqce.velocity_range.start,
                Event_Seqce.velocity_range.stop,
                n / (Event_Seqce.velocity_steps - 1))

    def __init__(self, events=[]):
        self.events = copy.deepcopy(events)
        time = 0
        for event in self.events:
            event.time = time
            if event.type == 'time_shift':
                time += Event_Seqce.time_shift_bins[event.value]
    
    def conv2note_seq(self):
        time = 0
        notes = []
        
        velocity = STATE_VELOCITY
        velocity_bins = Event_Seqce.getting_veloc_basket()

        last_notes = {}

        for event in self.events:
            if event.type == 'note_on':
                pitch = event.value + Event_Seqce.pitch_range.start
                note = Note(velocity, pitch, time, None)
                notes.append(note)
                last_notes[pitch] = note

            elif event.type == 'note_off':
                pitch = event.value + Event_Seqce.pitch_range.start

                if pitch in last_notes:
                    note = last_notes[pitch]
                    note.end = max(time, note.start + MIN_NOTE_LENGTH)
                    del last_notes[pitch]
            
            elif event.type == 'velocity':
                index = min(event.value, velocity_bins.size - 1)
                velocity = velocity_bins[index]

            elif event.type == 'time_shift':
                time += Event_Seqce.time_shift_bins[event.value]

        for note in notes:
            if note.end is None:
                note.end = note.start + STATE_NOTE_LENGTH

            note.velocity = int(note.velocity)

        return Note_Seqce(notes)

    def conv2array(self):
        feat_idxs = Event_Seqce.featur_ranges()
        idxs = [feat_idxs[event.type][event.value] for event in self.events]
        dtype = np.uint8 if Event_Seqce.dim() <= 256 else np.uint16
        return np.array(idxs, dtype=dtype)

class Control:

    def __init__(self, pitch_histogram, note_density):
        self.pitch_histogram = pitch_histogram # list
        self.note_density = note_density # int
    
    def __repr__(self):
        return 'Control(pitch_histogram={}, note_density={})'.format(
                self.pitch_histogram, self.note_density)
    
    def conv2array(self):
        featur_dimen = ControlSeq.featur_dimen()
        ndens = np.zeros([featur_dimen['note_density']])
        ndens[self.note_density] = 1. # [dens_dim]
        phist = np.array(self.pitch_histogram) # [hist_dim]
        return np.concatenate([ndens, phist], 0) # [dens_dim + hist_dim]


class ControlSeq:

    note_density_bins = STATE_NOTE_DENSITY_BINS
    window_size = STATE_WINDOW_SIZE

    @staticmethod
    def dim():
        return sum(ControlSeq.featur_dimen().values())

    @staticmethod
    def featur_dimen():
        note_density_dim = len(ControlSeq.note_density_bins)
        return collections.OrderedDict([
            ('pitch_histogram', 12),
            ('note_density', note_density_dim)
        ])

    @staticmethod
    def featur_ranges():
        offset = 0
        featur_ranges = collections.OrderedDict()
        for feat_name, feat_dim in ControlSeq.featur_dimen().items():
            featur_ranges[feat_name] = range(offset, offset + feat_dim)
            offset += feat_dim
        return featur_ranges
    
    @staticmethod
    def recover_compressed_array(array):
        featur_dimen = ControlSeq.featur_dimen()
        ndens = np.zeros([array.shape[0], featur_dimen['note_density']])
        ndens[np.arange(array.shape[0]), array[:, 0]] = 1. # [steps, dens_dim]
        phist = array[:, 1:].astype(np.float64) / 255 # [steps, hist_dim]
        return np.concatenate([ndens, phist], 1) # [steps, dens_dim + hist_dim]

    def __init__(self, controls):
        for control in controls:
            self.controls = copy.deepcopy(controls)


import os
import numpy as np

def search_files(root, exts=[]):
    def _has_ext(name):
        if not exts:
            return True
        name = name.lower()
        for ext in exts:
            if name.endswith(ext):
                return True
        return False
    for path, _, files in os.walk(root):
        for name in files:
            if _has_ext(name):
                yield os.path.join(path, name)

def event_indec2midi_file(event_indeces, midi_file_name, velocity_scale=0.8):
    event_seq = Event_Seqce.take_from_array(event_indeces)
    note_seq = event_seq.conv2note_seq()
    for note in note_seq.notes:
        note.velocity = int((note.velocity - 64) * velocity_scale + 64)
    note_seq.convert2midi_file(midi_file_name)
    return len(note_seq.notes)

def grad_norm(parameters, norm_type=2):
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def dict2params(d, f=','):
    return f.join(f'{k}={v}' for k, v in d.items())

def params2dict(p, f=',', e='='):
    d = {}
    for item in p.split(f):
        item = item.split(e)
        if len(item) < 2:
            continue
        k, *v = item
        d[k] = eval('='.join(v))
    return d

class Work_w_Dataset:
    def __init__(self, root, verbose=False):
        assert os.path.isdir(root), root
        paths = search_files(root, ['.data'])
        self.root = root
        self.samples = []
        self.seqlens = []
        if verbose:
            paths = Bar(root).iter(list(paths))
        for path in paths:
            eventseq, controlseq = torch.load(path)
            controlseq = ControlSeq.recover_compressed_array(controlseq)
            assert len(eventseq) == len(controlseq)
            self.samples.append((eventseq, controlseq))
            self.seqlens.append(len(eventseq))
        self.avglen = np.mean(self.seqlens)
    
    def batches(self, batch_size, window_size, stride_size):
        indeces = [(i, range(j, j + window_size))
                   for i, seqlen in enumerate(self.seqlens)
                   for j in range(0, seqlen - window_size, stride_size)]
        while True:
            eventseq_batch = []
            controlseq_batch = []
            n = 0
            for ii in np.random.permutation(len(indeces)):
                i, r = indeces[ii]
                eventseq, controlseq = self.samples[i]
                eventseq = eventseq[r.start:r.stop]
                controlseq = controlseq[r.start:r.stop]
                eventseq_batch.append(eventseq)
                controlseq_batch.append(controlseq)
                n += 1
                if n == batch_size:
                    yield (np.stack(eventseq_batch, axis=1),
                           np.stack(controlseq_batch, axis=1))
                    eventseq_batch.clear()
                    controlseq_batch.clear()
                    n = 0
    
    def __repr__(self):
        return (f'Work_w_Dataset(root="{self.root}", '
                f'samples={len(self.samples)}, '
                f'avglen={self.avglen})')
## Config

import torch
device = torch.device('cpu')

model = {
    'init_dim': 32,
    'event_dim': Event_Seqce.dim(),
    'control_dim': ControlSeq.dim(),
    'hidden_dim': 512,
    'gru_layers': 3,
    'gru_dropout': 0.3,
}

train = {
    'learning_rate': 0.001,
    'batch_size': 64,
    'window_size': 200,
    'stride_size': 10,
    'control_ratio': 1.0,
    'teacher_forcing_ratio': 1.0
}

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import numpy as np
from progress.bar import Bar

class Model_RNN(nn.Module):
    def __init__(self, event_dim, control_dim, init_dim, hidden_dim,
                 gru_layers=3, gru_dropout=0.3):
        super().__init__()
        self.event_dim = event_dim
        self.control_dim = control_dim
        self.init_dim = init_dim
        self.hidden_dim = hidden_dim
        self.gru_layers = gru_layers
        self.concat_dim = event_dim + 1 + control_dim
        self.input_dim = hidden_dim
        self.output_dim = event_dim
        self.primary_event = self.event_dim - 1
        self.inithid_fc = nn.Linear(init_dim, gru_layers * hidden_dim)
        self.inithid_fc_activation = nn.Tanh()
        self.event_embedding = nn.Embedding(event_dim, event_dim)
        self.concat_input_fc = nn.Linear(self.concat_dim, self.input_dim)
        self.concat_input_fc_activation = nn.LeakyReLU(0.1, inplace=True)
        self.gru = nn.GRU(self.input_dim, self.hidden_dim,
                          num_layers=gru_layers, dropout=gru_dropout)
        self.output_fc = nn.Linear(hidden_dim * gru_layers, self.output_dim)
        self.output_fc_activation = nn.Softmax(dim=-1)

        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_normal_(self.event_embedding.weight)
        nn.init.xavier_normal_(self.inithid_fc.weight)
        self.inithid_fc.bias.data.fill_(0.)
        nn.init.xavier_normal_(self.concat_input_fc.weight)
        nn.init.xavier_normal_(self.output_fc.weight)
        self.output_fc.bias.data.fill_(0.)

    def _sample_even(self, output, greedy=True, temperature=1.0):
        if greedy:
            return output.argmax(-1)
        else:
            output = output / temperature
            probs = self.output_fc_activation(output)
            return Categorical(probs).sample()

    def forw(self, event, control=None, hidden=None):
        batch_size = event.shape[1]
        event = self.event_embedding(event)

        if control is None:
            default = torch.ones(1, batch_size, 1).to(device)
            control = torch.zeros(1, batch_size, self.control_dim).to(device)
        else:
            default = torch.zeros(1, batch_size, 1).to(device)

        concat = torch.cat([event, default, control], -1)
        input = self.concat_input_fc(concat)
        input = self.concat_input_fc_activation(input)

        _, hidden = self.gru(input, hidden)
        output = hidden.permute(1, 0, 2).contiguous()
        output = output.view(batch_size, -1).unsqueeze(0)
        output = self.output_fc(output)
        return output, hidden
    
    def simple_event(self, batch_size):
        return torch.LongTensor([[self.primary_event] * batch_size]).to(device)
    
    def initialise2hidden(self, init):
        batch_size = init.shape[0]
        out = self.inithid_fc(init)
        out = self.inithid_fc_activation(out)
        out = out.view(self.gru_layers, batch_size, self.hidden_dim)
        return out
    
    def expand_contr(self, controls, steps):
        if controls.shape[0] > 1:
            return controls[:steps]
        return controls.repeat(steps, 1, 1)
    
    def gen_samples(self, init, steps, events=None, controls=None, greedy=1.0,
                 temperature=1.0, teacher_forcing_ratio=1.0):

        batch_size = init.shape[0]

        use_teacher_forcing = events is not None
        if use_teacher_forcing:
            events = events[:steps-1]

        event = self.simple_event(batch_size)
        use_control = controls is not None
        if use_control:
            controls = self.expand_contr(controls, steps)
        hidden = self.initialise2hidden(init)

        outputs = []
        step_iter = range(steps)

        for step in step_iter:
            control = controls[step].unsqueeze(0) if use_control else None
            output, hidden = self.forw(event, control, hidden)

            use_greedy = np.random.random() < greedy
            event = self._sample_even(output, greedy=use_greedy,
                                       temperature=temperature)

            outputs.append(output)

            if use_teacher_forcing and step < steps - 1:
                if np.random.random() <= teacher_forcing_ratio:
                    event = events[step].unsqueeze(0)
        
        return torch.cat(outputs, 0)

def get_options():
    parser = optparse.OptionParser()

    parser.add_option('-s',
                      dest='sess_path',
                      type='string',
                      default='train.sess')

    parser.add_option('-d',
                      dest='data_path',
                      type='string',
                      default='dataset/processed/')

    parser.add_option('-b',
                      dest='batch_size',
                      type='int',
                      default=train['batch_size'])  

    parser.add_option('-w',
                      dest='window_size',
                      type='int',
                      default=train['window_size'])

    return parser.parse_args()[0]

options = get_options()

#------------------------------------------------------------------------

sess_path = options.sess_path
data_path = options.data_path
saving_interval = 60

learning_rate = train['learning_rate']
batch_size = options.batch_size
window_size = options.window_size
stride_size = train['stride_size']
control_ratio = train['control_ratio']
teacher_forcing_ratio = train['teacher_forcing_ratio']

event_dim = Event_Seqce.dim()
control_dim = ControlSeq.dim()
model_config = model
device = device

def prep_sess():
    global sess_path, model_config, device, learning_rate
    try:
        sess = torch.load(sess_path)
        if 'model_config' in sess and sess['model_config'] != model_config:
            model_config = sess['model_config']
            print('Use session config instead:')
            print(utils.dict2params(model_config))
        model_state = sess['model_state']
        optimizer_state = sess['model_optimizer_state']
        print('Session is loaded from', sess_path)
        sess_loaded = True
    except:
        print('New session')
        sess_loaded = False
    model = Model_RNN(**model_config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    if sess_loaded:
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
    return model, optimizer

def load_dataset():
    global data_path
    dataset = Work_w_Dataset(data_path, verbose=True)
    dataset_size = len(dataset.samples)
    assert dataset_size > 0
    return dataset


print('Loading session')
model, optimizer = prep_sess()
print(model)

print('-' * 70)

print('Loading dataset')
dataset = load_dataset()
print(dataset)

print('-' * 70)

#------------------------------------------------------------------------

def save_model():
    global model, optimizer, model_config, sess_path
    print('Saving to', sess_path)
    torch.save({'model_config': model_config,
                'model_state': model.state_dict(),
                'model_optimizer_state': optimizer.state_dict()}, sess_path)
    print('Done saving')



last_saving_time = time.time()
loss_function = nn.CrossEntropyLoss()

try:
    batch_gen = dataset.batches(batch_size, window_size, stride_size)

    for iteration, (events, controls) in enumerate(batch_gen):

        events = torch.LongTensor(events).to(device)
        assert events.shape[0] == window_size

        if np.random.random() < control_ratio:
            controls = torch.FloatTensor(controls).to(device)
            assert controls.shape[0] == window_size
        else:
            controls = None

        init = torch.randn(batch_size, model.init_dim).to(device)
        outputs = model.gen_samples(init, window_size, events=events[:-1], controls=controls,
                                 teacher_forcing_ratio=teacher_forcing_ratio)
        assert outputs.shape[:2] == events.shape[:2]

        loss = loss_function(outputs.view(-1, event_dim), events.view(-1))
        model.zero_grad()
        loss.backward()

        norm = grad_norm(model.parameters())
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()

        print(f'iter {iteration}, loss: {loss.item()}')

        if time.time() - last_saving_time > saving_interval:
            save_model()
            last_saving_time = time.time()

except KeyboardInterrupt:
    save_model()
