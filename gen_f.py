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

#==================================================================================
# Parameters
#==================================================================================

# Note_Seqce -------------------------------------------------------------------------

STATE_RESOLUTION = 220
STATE_TEMP = 120
STATE_VELOCITY = 52
STATE_PITCH_RANGE = range(21, 109)
STATE_VELOCITY_RANGE = range(21, 109)
# Event_Seqce ------------------------------------------------------------------------

BEAT_LENGTH = 60 / STATE_TEMP
STATE_TIME_SHIFT_BINS = 1.15 ** np.arange(32) / 65
STATE_VELOCITY_STEPS = 32
STATE_NOTE_LENGTH = BEAT_LENGTH * 2
MIN_NOTE_LENGTH = BEAT_LENGTH / 2

# ControlSeq ----------------------------------------------------------------------

STATE_WINDOW_SIZE = BEAT_LENGTH * 4
STATE_NOTE_DENSITY_BINS = np.arange(12) * 3 + 1


#==================================================================================
# Notes
#==================================================================================

class Note_Seqce:
    def __init__(self, notes=[]):
        self.notes = []
        if notes:
            for note in notes:
                notes = filter(lambda note: note.end >= note.start, notes)
            self.add_notes(list(notes))
    
    def copy(self):
        return copy.deepcopy(self)

    def conv_to_mid(self):
        midi = PrettyMIDI(resolution=STATE_RESOLUTION, initial_tempo=STATE_TEMP)
        inst = Instrument(1, False, 'Note_Seqce')
        inst.notes = copy.deepcopy(self.notes)
        midi.instruments.append(inst)
        return midi

    def conv_to_mid_file(self, path, *kargs, **kwargs):
        self.conv_to_mid(*kargs, **kwargs).write(path)


    def add_notes(self, notes):
        self.notes += notes
        self.notes.sort(key=lambda note: note.start)



    
    


#==================================================================================
# Events
#==================================================================================

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
    def from_array(event_indeces):
        time = 0
        events = []
        for event_index in event_indeces:
            for event_type, feat_range in Event_Seqce.feat_ranges().items():
                if feat_range.start <= event_index < feat_range.stop:
                    event_value = event_index - feat_range.start
                    events.append(Event(event_type, time, event_value))
                    if event_type == 'time_shift':
                        time += Event_Seqce.time_shift_bins[event_value]
                    break

        return Event_Seqce(events)

    @staticmethod
    def dim():
        return sum(Event_Seqce.feat_dims().values())

    @staticmethod
    def feat_dims():
        feat_dims = collections.OrderedDict()
        feat_dims['note_on'] = len(Event_Seqce.pitch_range)
        feat_dims['note_off'] = len(Event_Seqce.pitch_range)
        feat_dims['velocity'] = Event_Seqce.velocity_steps
        feat_dims['time_shift'] = len(Event_Seqce.time_shift_bins)
        return feat_dims

    @staticmethod
    def feat_ranges():
        offset = 0
        feat_ranges = collections.OrderedDict()
        for feat_name, feat_dim in Event_Seqce.feat_dims().items():
            feat_ranges[feat_name] = range(offset, offset + feat_dim)
            offset += feat_dim
        return feat_ranges

    @staticmethod
    def get_velocity_bins():
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
    
    def to_note_seq(self):
        time = 0
        notes = []
        
        velocity = STATE_VELOCITY
        velocity_bins = Event_Seqce.get_velocity_bins()

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

    def to_array(self):
        feat_idxs = Event_Seqce.feat_ranges()
        idxs = [feat_idxs[event.type][event.value] for event in self.events]
        dtype = np.uint8 if Event_Seqce.dim() <= 256 else np.uint16
        return np.array(idxs, dtype=dtype)



#==================================================================================
# Controls
#==================================================================================

class Control:

    def __init__(self, pitch_histogram, note_density):
        self.pitch_histogram = pitch_histogram # list
        self.note_density = note_density # int
    
    def __repr__(self):
        return 'Control(pitch_histogram={}, note_density={})'.format(
                self.pitch_histogram, self.note_density)
    
    def to_array(self):
        feat_dims = ControlSeq.feat_dims()
        ndens = np.zeros([feat_dims['note_density']])
        ndens[self.note_density] = 1. # [dens_dim]
        phist = np.array(self.pitch_histogram) # [hist_dim]
        return np.concatenate([ndens, phist], 0) # [dens_dim + hist_dim]


class ControlSeq:

    note_density_bins = STATE_NOTE_DENSITY_BINS
    window_size = STATE_WINDOW_SIZE

    @staticmethod
    def dim():
        return sum(ControlSeq.feat_dims().values())

    @staticmethod
    def feat_dims():
        note_density_dim = len(ControlSeq.note_density_bins)
        return collections.OrderedDict([
            ('pitch_histogram', 12),
            ('note_density', note_density_dim)
        ])

    @staticmethod
    def feat_ranges():
        offset = 0
        feat_ranges = collections.OrderedDict()
        for feat_name, feat_dim in ControlSeq.feat_dims().items():
            feat_ranges[feat_name] = range(offset, offset + feat_dim)
            offset += feat_dim
        return feat_ranges
    
    @staticmethod
    def recover_compressed_array(array):
        feat_dims = ControlSeq.feat_dims()
        ndens = np.zeros([array.shape[0], feat_dims['note_density']])
        ndens[np.arange(array.shape[0]), array[:, 0]] = 1. # [steps, dens_dim]
        phist = array[:, 1:].astype(np.float64) / 255 # [steps, hist_dim]
        return np.concatenate([ndens, phist], 1) # [steps, dens_dim + hist_dim]

    def __init__(self, controls):
        for control in controls:
            self.controls = copy.deepcopy(controls)

#-------------------------------------------------------------------
# ------------------------------------------------------------------  
##UTILS 
#--------------------------------------------------------------------
#--------------------------------------------------------------------
import os
import numpy as np


def find_files_by_extensions(root, exts=[]):
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

def event_indeces_to_midi_file(event_indeces, midi_file_name, velocity_scale=0.8):
    event_seq = Event_Seqce.from_array(event_indeces)
    note_seq = event_seq.to_note_seq()
    for note in note_seq.notes:
        note.velocity = int((note.velocity - 64) * velocity_scale + 64)
    note_seq.conv_to_mid_file(midi_file_name)
    return len(note_seq.notes)




#-------------------------------------------------------------------
# ------------------------------------------------------------------  
## Config
#--------------------------------------------------------------------
#--------------------------------------------------------------------

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

#-------------------------------------------------------------------
# ------------------------------------------------------------------  
## Model
#--------------------------------------------------------------------
#--------------------------------------------------------------------

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

        self._initialize_weights()
    
    def _initialize_weights(self):
        nn.init.xavier_normal_(self.event_embedding.weight)
        nn.init.xavier_normal_(self.inithid_fc.weight)
        self.inithid_fc.bias.data.fill_(0.)
        nn.init.xavier_normal_(self.concat_input_fc.weight)
        nn.init.xavier_normal_(self.output_fc.weight)
        self.output_fc.bias.data.fill_(0.)

    def _sample_event(self, output, greedy=True, temperature=1.0):
        if greedy:
            return output.argmax(-1)
        else:
            output = output / temperature
            probs = self.output_fc_activation(output)
            return Categorical(probs).sample()

    def forward(self, event, control=None, hidden=None):
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
    
    def get_primary_event(self, batch_size):
        return torch.LongTensor([[self.primary_event] * batch_size]).to(device)
    
    def init_to_hidden(self, init):
        batch_size = init.shape[0]
        out = self.inithid_fc(init)
        out = self.inithid_fc_activation(out)
        out = out.view(self.gru_layers, batch_size, self.hidden_dim)
        return out
    
    def expand_controls(self, controls, steps):
        if controls.shape[0] > 1:
            return controls[:steps]
        return controls.repeat(steps, 1, 1)
    
    def generate(self, init, steps, events=None, controls=None, greedy=1.0,
                 temperature=1.0, teacher_forcing_ratio=1.0):

        batch_size = init.shape[0]

        use_teacher_forcing = events is not None
        if use_teacher_forcing:
            events = events[:steps-1]

        event = self.get_primary_event(batch_size)
        use_control = controls is not None
        if use_control:
            controls = self.expand_controls(controls, steps)
        hidden = self.init_to_hidden(init)

        outputs = []
        step_iter = range(steps)
        step_iter = Bar('Some_magic').iter(step_iter)

        for step in step_iter:
            control = controls[step].unsqueeze(0) if use_control else None
            output, hidden = self.forward(event, control, hidden)

            use_greedy = np.random.random() < greedy
            event = self._sample_event(output, greedy=use_greedy,
                                       temperature=temperature)

            outputs.append(event)

            if use_teacher_forcing and step < steps - 1:
                if np.random.random() <= teacher_forcing_ratio:
                    event = events[step].unsqueeze(0)
        
        return torch.cat(outputs, 0)


#==================================================================================
# Generate
#==================================================================================

#========================================================================
# Settings
#========================================================================

def getopt():
    parser = optparse.OptionParser()

    parser.add_option('-c', '--control',
                      dest='control',
                      type='string',
                      default=None)

    parser.add_option('-b', '--batch-size',
                      dest='batch_size',
                      type='int',
                      default=6)

    parser.add_option('-s', '--session',
                      dest='sess_path',
                      type='string',
                      default='save/train.sess',
                      help='session file containing the trained model')

    parser.add_option('-o', '--output-dir',
                      dest='output_dir',
                      type='string',
                      default='output/')

    parser.add_option('-l', '--max-length',
                      dest='max_len',
                      type='int',
                      default=0)

    return parser.parse_args()[0]


opt = getopt()

#------------------------------------------------------------------------

output_dir = opt.output_dir
sess_path = opt.sess_path
batch_size = opt.batch_size
max_len = opt.max_len
control = opt.control 


assert os.path.isfile(sess_path), f'"{sess_path}" is not a file'

if control is not None:
    if os.path.isfile(control) or os.path.isdir(control):
        _, compressed_controls = torch.load(control, map_location='cpu')
        controls = ControlSeq.recover_compressed_array(compressed_controls)
        controls = torch.tensor(controls, dtype=torch.float32)
        controls = controls.unsqueeze(1).repeat(1, batch_size, 1).to(device)

else:
    controls = None
    control = 'NONE'

#------------------------------------------------------------------------
print('=' * 80)

state = torch.load(sess_path, map_location='cpu')
model = Model_RNN(**state['model_config']).to(device)
model.load_state_dict(state['model_state'])
model.eval()
print('=' * 80)

init = torch.randn(batch_size, model.init_dim).to(device)

outputs = model.generate(init, max_len, controls=controls)

outputs = outputs.cpu().numpy().T # [batch, steps]

os.makedirs(output_dir, exist_ok=True)
files = []
for i, output in enumerate(outputs):
    name = f'{i}.mid'
    path = os.path.join(output_dir, name)
    files.append(name)
    n_notes = event_indeces_to_midi_file(output, path)
    print(f'===> {path} ({n_notes} notes)')

print(files)
from midi2audio import FluidSynth
for i, sample in enumerate(files):
    fs = FluidSynth('font.sf2')
    fs.midi_to_audio(sample, os.path.join(output_dir, f'{i}.wav'))

