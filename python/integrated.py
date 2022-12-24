# -*- coding: utf-8 -*-
from fileinput import filename
import os
import sys
# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
os.environ['GOOGLE_APPLICATION_CREDENTIALS']=r"C:/Users/dla12/Documents/Developer/Generative-Conversational-Model-Considering-Age-In-the-Metaverse/python/sesac-371212-227f22f8a69a.json"
# from test import overwrite_chars
import argparse
import logging
from multiprocessing import Process, freeze_support
import numpy as np
import pandas as pd
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.lightning import LightningModule
from torch.utils.data import DataLoader, Dataset
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel
import pyaudio
from six.moves import queue
from google.cloud import speech
from google.cloud import texttospeech
from array import array
import torchaudio
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F

parser = argparse.ArgumentParser(description='Age_dialogue based on KoGPT-2')

parser.add_argument('--chat',
                    action='store_true',
                    default=False,
                    help='response generation on given user input')

parser.add_argument('--model_params',
                    type=str,
                    default='dialogue.ckpt',
                    help='model binary for starting chat')

parser.add_argument('--train',
                    action='store_true',
                    default=False,
                    help='for training')

audio_data = pd.read_csv('C:/Users/dla12/Documents/Developer/Generative-Conversational-Model-Considering-Age-In-the-Metaverse/python/age.csv')

class AgeSoundDataset(Dataset):

    def __init__(self, annotations_file, audio_dir, transformation, target_sample_rate, num_samples, device):
        self.annotaions = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.device = device
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        
    def __len__(self):
        return len(self.annotaions)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)
        # signal -> (num_channels, smaples) -> (2, 16000) -> (1, 16000)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        signal = self.transformation(signal)
        label = str(label)
        label = label.replace('.0','')
        label = int(label)
        label = label/5 - 1
        label = int(label)
        return signal, label

    def _cut_if_necessary(self, signal):
        # signal -> Tensor -> (1, num_samples) -> (1, 50000) -> (1, 22050)
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal

    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            numpy_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, numpy_missing_samples) # (1, 1, 2, 2)
            # [1, 1, 1] -> [0, 1, 1, 1, 0 , 0]
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1: 
            signal = torch.mean(signal, dim = 0, keepdim = True)
        return signal

    def _get_audio_sample_path(self, index):
        path = os.path.join(self.audio_dir,self.annotaions.iloc[index, 2])
        # print(self.annotaions.iloc[index, 2])
        return path

    def _get_audio_sample_label(self, index):
        # print(self.annotaions.iloc[index, 6])
        return self.annotaions.iloc[index, 6]

from torch import nn
from torchsummary import summary

class CNNNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        # 4conv blocks / flatten / linear / softmax
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels = 1,
                out_channels = 16,
                kernel_size = 3,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels = 16,
                out_channels = 32,
                kernel_size = 3,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels = 32,
                out_channels = 64,
                kernel_size = 3,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels = 64,
                out_channels = 128,
                kernel_size = 3,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.flatten = nn.Flatten()
        # conv4's data
        
        self.linear = nn.Linear(128 * 5 * 4, 17)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        logits = self.linear(x)
        preditctions = self.softmax(logits)
        return preditctions


import torch
SAMPLE_RATE = 22050
NUM_SAMPLES = 22050
class_mapping = [
    "5",
    "10",
    "15",
    "20",
    "25",
    "30",
    "35",
    "40",
    "45",
    "50",
    "55",
    "60",
    "65",
    "70",
    "75",
    "80",
    "85"
]

def predict(model, input, target, class_mapping):
    model.eval()
    with torch.no_grad():
        predictions = model(input)
        # Tensor (1, 10) -> [ [0.1, 0.01, ...., 0.6] ]
        predicted_index = predictions[0].argmax(0)
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]
    return predicted, expected


class MicrophoneStream(object):
    """Opens a recording stream as a generator yielding the audio chunks."""
    def __init__(self, rate, chunk):
        self._rate = rate
        self._chunk = chunk

        # Create a thread-safe buffer of audio data
        self._buff = queue.Queue()
        self.closed = True

    def __enter__(self):
        # create pyaudio interface
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            # The API currently only supports 1-channel (mono) audio
            # https://goo.gl/z757pE
            channels=1, rate=self._rate,
            input=True, frames_per_buffer=self._chunk,
            # Run the audio stream asynchronously to fill the buffer object.
            # This is necessary so that the input device's buffer doesn't
            # overflow while the calling thread makes network requests, etc.
            stream_callback=self._fill_buffer,
        )
        
        self.closed = False

        return self

    def __exit__(self, type, value, traceback):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        # Signal the generator to terminate so that the client's
        # streaming_recognize method will not block the process termination.
        self._buff.put(None)
        self._audio_interface.terminate()


    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        """Continuously collect data from the audio stream, into the buffer."""
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        while not self.closed:
            # Use a blocking get() to ensure there's at least one chunk of
            # data, and stop iteration if the chunk is None, indicating the
            # end of the audio stream.
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]
            # Now consume whatever other adata's still buffered.
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break

            yield b''.join(data)


def listen_print_loop(responses):
    """Iterates through server responses and prints them.
    The responses passed is a generator that will block until a response
    is provided by the server.
    Each response may contain multiple results, and each result may contain
    multiple alternatives; for details, see https://goo.gl/tjCPAU.  Here we
    print only the transcription for the top alternative of the top result.
    In this case, responses are provided for interim results as well. If the
    response is an interim one, print a line feed at the end of it, to allow
    the next result to overwrite it, until the response is a final one. For the
    final one, print a newline to preserve the finalized transcription.
    """
    num_chars_printed = 0
    for response in responses:
        if not response.results:
            continue

        # The `results` list is consecutive. For streaming, we only care about
        # the first result being considered, since once it's `is_final`, it
        # moves on to considering the next utterance.
        result = response.results[0]
        if not result.alternatives:
            continue
        
        # Display the transcription of the top alternative.
        transcript = result.alternatives[0].transcript
        
        # Display interim results, but with a carriage return at the end of the
        # line, so subsequent lines will overwrite them.
        #
        # If the previous result was longer than this one, we need to print
        # some extra spaces to overwrite the previous result
        overwrite_chars = ' ' * (num_chars_printed - len(transcript))
             
    
        if not result.is_final:
            sys.stdout.write(transcript + overwrite_chars + '\r')
            sys.stdout.flush()

            num_chars_printed = len(transcript)

            
        else:
            user_input = transcript + overwrite_chars  
            # print(user_input)
            # print(type(user_input))
            # Exit recognition if any of the transcribed phrases could be
            # one of our keywords.
            
            num_chars_printed = 0
            return user_input
            if re.search(r'\b(exit|quit)\b', transcript, re.I):
                print('Exiting..')
                break
            # dialogue_trainer.KoGPT2Chat.chat(KoGPT2Chat.self, sent='0')
            num_chars_printed = 0



logger = logging.getLogger()
logger.setLevel(logging.INFO)

U_TKN = '<usr>'
S_TKN = '<sys>'
BOS = '</s>'
EOS = '</s>'
MASK = '<unused0>'
SENT = '<unused1>'
PAD = '<pad>'

TOKENIZER = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
            bos_token=BOS, eos_token=EOS, unk_token='<unk>',
            pad_token=PAD, mask_token=MASK) 


class CharDataset(Dataset):
    def __init__(self, chats, max_len=32):
        self._data = chats
        self.first = True
        self.q_token = U_TKN
        self.a_token = S_TKN
        self.sent_token = SENT
        self.bos = BOS
        self.eos = EOS
        self.mask = MASK
        self.pad = PAD
        self.max_len = max_len
        self.tokenizer = TOKENIZER 

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        turn = self._data.iloc[idx]
        q = turn['Q']
        a = turn['A']
        age = str(turn['Label'])
        q_toked = self.tokenizer.tokenize(str(self.q_token) + str(q) + \
                                          str(self.sent_token) + str(age))   
        q_len = len(q_toked)
        a_toked = self.tokenizer.tokenize(str(self.a_token) + str(a) + str(self.eos))
        a_len = len(a_toked)
        if q_len + a_len > self.max_len:
            a_len = self.max_len - q_len
            if a_len <= 0:
                q_toked = q_toked[-(int(self.max_len/2)):]
                q_len = len(q_toked)
                a_len = self.max_len - q_len
                assert a_len > 0
            a_toked = a_toked[:a_len]
            a_len = len(a_toked)
            assert a_len == len(a_toked), f'{a_len} ==? {len(a_toked)}'
        # [mask, mask, ...., mask, ..., <bos>,..A.. <eos>, <pad>....]
        labels = [
            self.mask,
        ] * q_len + a_toked[1:]
        if self.first:
            logging.info("contexts : {}".format(q))
            logging.info("toked ctx: {}".format(q_toked))
            logging.info("response : {}".format(a))
            logging.info("toked response : {}".format(a_toked))
            logging.info('labels {}'.format(labels))
            self.first = False
        mask = [0] * q_len + [1] * a_len + [0] * (self.max_len - q_len - a_len)
        self.max_len
        labels_ids = self.tokenizer.convert_tokens_to_ids(labels)
        while len(labels_ids) < self.max_len:
            labels_ids += [self.tokenizer.pad_token_id]
        token_ids = self.tokenizer.convert_tokens_to_ids(q_toked + a_toked)
        while len(token_ids) < self.max_len:
            token_ids += [self.tokenizer.pad_token_id]
        return(token_ids, np.array(mask),
               labels_ids)


class KoGPT2Chat(LightningModule):
    def __init__(self, hparams, **kwargs):
        super(KoGPT2Chat, self).__init__()
        self.hparams = hparams
        self.neg = -1e18
        self.kogpt2 = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')
        self.loss_function = torch.nn.CrossEntropyLoss(reduction='none')

    @staticmethod
    def add_model_specific_args(parent_parser):
        # add model specific args
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--max-len',
                            type=int,
                            default=64,
                            help='max sentence length on input (default: 32)')

        parser.add_argument('--batch-size',
                            type=int,
                            default=28,
                            help='batch size for training (default: 96)')
        parser.add_argument('--lr',
                            type=float,
                            default=5e-5,
                            help='The initial learning rate')
        parser.add_argument('--warmup_ratio',
                            type=float,
                            default=0.1,
                            help='warmup ratio')
        return parser

    def forward(self, inputs):
        # (batch, seq_len, hiddens)
        output = self.kogpt2(inputs, return_dict=True)
        return output.logits

    def training_step(self, batch, batch_idx):
        token_ids, mask, label = batch
        out = self(token_ids)
        mask_3d = mask.unsqueeze(dim=2).repeat_interleave(repeats=out.shape[2], dim=2)
        mask_out = torch.where(mask_3d == 1, out, self.neg * torch.ones_like(out))
        loss = self.loss_function(mask_out.transpose(2, 1), label)
        loss_avg = loss.sum() / mask.sum()
        self.log('train_loss', loss_avg)
        return loss_avg

    def configure_optimizers(self):
        # Prepare optimizer
        param_optimizer = list(self.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.hparams.lr, correct_bias=False)
        # warm up lr
        num_train_steps = len(self.train_dataloader()) * self.hparams.max_epochs
        num_warmup_steps = int(num_train_steps * self.hparams.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)
        lr_scheduler = {'scheduler': scheduler, 'name': 'cosine_schedule_with_warmup',
                        'monitor': 'loss', 'interval': 'step',
                        'frequency': 1}
        return [optimizer], [lr_scheduler]

    def _collate_fn(self, batch):
        data = [item[0] for item in batch]
        mask = [item[1] for item in batch]
        label = [item[2] for item in batch]
        return torch.LongTensor(data), torch.LongTensor(mask), torch.LongTensor(label)

    def train_dataloader(self):
        data = pd.read_csv('merge_shuffled.csv')
        self.train_set = CharDataset(data, max_len=self.hparams.max_len)
        train_dataloader = DataLoader(
            self.train_set, batch_size=self.hparams.batch_size, num_workers=0,
            shuffle=True, collate_fn=self._collate_fn)
        return train_dataloader

    def chat(self, sent='0'):
        tok = TOKENIZER
        sent_tokens = tok.tokenize(sent)
        client = speech.SpeechClient()
        config = speech.RecognitionConfig(encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,sample_rate_hertz=RATE,language_code='ko-KR')
        streaming_config = speech.StreamingRecognitionConfig(config=config, single_utterance = True ,interim_results=True)
        while 1:
            print('user > ')
            with MicrophoneStream(RATE, CHUNK) as stream:
                audio_generator = stream.generator()
            ######################################################
                def myReq(content):
                    
                    # frames.append(content)
                    # wavfile.setnchannels(1)
                    # wavfile.setsampwidth(2)
                    # wavfile.setframerate(22050)
                    # # wav 저장하는 코드
                    # wavfile.writeframes(b''.join(content))#append frames recorded to file
                    return speech.StreamingRecognizeRequest(audio_content=content)
                    
          
                # wav  열고
                
                # wavfile = wave.open(FILE_NAME,'wb') 
                # frames=[]
                # wavfile.setparams((1, 2, 22050, 0, "NONE", "Uncompressed")) 
                requests = (myReq(content)for content in audio_generator)
                # wavfile.close() 
                # requests = (speech.StreamingRecognizeRequest(audio_content=content) for content in audio_generator)
                ###########################
                
                responses = client.streaming_recognize(streaming_config, requests)
                
                chat_input = listen_print_loop(responses)
                # Now, put the transcription responses to use.

                
                with torch.no_grad():
                    ###############################################
                    ANNOTATIONS_FILE = 'C:/Users/dla12/Documents/Developer/Sesac/age/age.csv'
                    AUDIO_DIR = 'C:/Users/dla12/Documents/Developer/Unity/sesac/Assets/Resources/'
                    SAMPLE_RATE = 22050
                    NUM_SAMPLES = 22050

                    # load back the model
                    cnn = CNNNetwork()
                    state_dict = torch.load("C:/Users/dla12/Documents/Developer/Sesac/age/age.pth")
                    cnn.load_state_dict(state_dict)

                    # load age dataset
                    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
                        sample_rate = SAMPLE_RATE,
                        n_fft = 1024,
                        hop_length = 512,
                        n_mels = 64
                    )
                    # ms = mel_spectrogram(signal)

                    asd = AgeSoundDataset(ANNOTATIONS_FILE, AUDIO_DIR, mel_spectrogram,
                                            SAMPLE_RATE, NUM_SAMPLES, "cpu")
                    
                    # get a sample from the age dataset for infence
                    input, target = asd[0][0], asd[0][1] # [batch size, num_channels, frequncy, time]
                    input.unsqueeze_(0)

                    # make an infernece
                    predicted, expected = predict(cnn, input, target, class_mapping)
                    age = int(predicted)

                    if (age <= 19):
                        age_output = ' child'
                    elif (age > 20 & age < 59):
                        age_output = ' adult'
                    else:
                        age_output = ' senior'    

                    print(age)
                    print(f"Predicted: '{age_output}'")
                    ##################################################
                    print(chat_input)
                    with open("C:/Users/dla12/Documents/Developer/Unity/sesac/Assets/Resources/chat_input.txt", "w",encoding="UTF-8") as f:
                        # Write the response to the output file.

                        f.write(chat_input)
                        f.close()
                    p = chat_input + age_output
                    # client_socket.sendall("chat_input.txt".encode("UTF-8"))
                    q = p.strip()
                    a = ''
                    while 1:
                        input_ids = torch.LongTensor(tok.encode(U_TKN + q + SENT + sent + S_TKN + a)).unsqueeze(dim=0)
                        pred = self(input_ids)
                        gen = tok.convert_ids_to_tokens(
                        torch.argmax(
                            pred,
                            dim=-1).squeeze().numpy().tolist())[-1]
                        if gen == EOS:
                            break
                        a += gen.replace('▁', ' ')
                        chat_output = a.strip()
                    print("Chatbot > {}".format(chat_output))
                    with open("C:/Users/dla12/Documents/Developer/Unity/sesac/Assets/Resources/chat_output.txt", "w",encoding="UTF-8") as f:
                        # Write the response to the output file.
                        f.write(chat_output)
                        f.close()
                    # client_socket.sendall("chat_output.txt".encode("UTF-8"))
                    # return q
                    # Instantiates a client
                    client_tts = texttospeech.TextToSpeechClient()

                    # Set the text input to be synthesized
                    synthesis_input = texttospeech.SynthesisInput(text=chat_output)

                    # Build the voice request, select the language code ("en-US") and the ssml
                    # voice gender ("neutral")
                    voice = texttospeech.VoiceSelectionParams(
                        language_code="ko-KR", name="ko-KR-Wavenet-D",ssml_gender=texttospeech.SsmlVoiceGender.MALE
                    )

                    # Select the type of audio file you want returned
                    audio_config = texttospeech.AudioConfig(
                        audio_encoding=texttospeech.AudioEncoding.LINEAR16
                    )

                    # Perform the text-to-speech request on the text input with the selected
                    # voice parameters and audio file type
                    response_tts = client_tts.synthesize_speech(
                        input=synthesis_input, voice=voice, audio_config=audio_config
                    )

                    # The response's audio_content is binary.
                    with open("C:/Users/dla12/Documents/Developer/Unity/sesac/Assets/Resources/output.wav", "wb") as out:
                        # Write the response to the output file.
                        out.write(response_tts.audio_content)
                        f.close()
                    # client_socket.sendall("output.wav".encode("UTF-8"))
                        # print('Audio content written to file "output.mp3"')



parser = KoGPT2Chat.add_model_specific_args(parser)
parser = Trainer.add_argparse_args(parser)
args = parser.parse_args()
logging.info(args)

if __name__ == "__main__":
    ANNOTATIONS_FILE = 'C:/Users/dla12/Documents/Developer/Generative-Conversational-Model-Considering-Age-In-the-Metaverse/python/age.csv'
    AUDIO_DIR = 'C:/Users/dla12/Documents/Developer/Generative-Conversational-Model-Considering-Age-In-the-Metaverse/python/'
    SAMPLE_RATE = 22050
    NUM_SAMPLES = 22050

    # load back the model
    cnn = CNNNetwork()
    state_dict = torch.load("C:/Users/dla12/Documents/Developer/Sesac/age/age.pth")
    cnn.load_state_dict(state_dict)

    # load age dataset
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate = SAMPLE_RATE,
        n_fft = 1024,
        hop_length = 512,
        n_mels = 64
    )
    # ms = mel_spectrogram(signal)

    asd = AgeSoundDataset(ANNOTATIONS_FILE, AUDIO_DIR, mel_spectrogram,
                            SAMPLE_RATE, NUM_SAMPLES, "cpu")
    
    # get a sample from the age dataset for infence
    input, target = asd[0][0], asd[0][1] # [batch size, num_channels, frequncy, time]
    input.unsqueeze_(0)

    # make an infernece
    predicted, expected = predict(cnn, input, target, class_mapping)

    print(f"Predicted: '{predicted}'")
  
