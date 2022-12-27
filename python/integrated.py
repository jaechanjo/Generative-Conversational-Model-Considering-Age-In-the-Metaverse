# -*- coding: utf-8 -*-
from fileinput import filename
import io
import os
import sys
# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
os.environ['GOOGLE_APPLICATION_CREDENTIALS']=r"C:/Users/dla12/Documents/Developer/Generative-Conversational-Model-Considering-Age-In-the-Metaverse/python/sesac-371212-227f22f8a69a.json"
# from test import overwrite_chars
import argparse
import logging
import subprocess
from multiprocessing import Process, freeze_support
import numpy as np
import pandas as pd
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.lightning import LightningModule
from torch.utils.data import DataLoader, Dataset
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel
import pyaudio
from google.cloud import speech
from google.cloud import texttospeech
from array import array
import torchaudio
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
from argparse import ArgumentParser
from googletrans import Translator


from gesticulator.model.model import GesticulatorModel
from gesticulator.interface.gesture_predictor import GesturePredictor
from gesticulator.visualization.motion_visualizer.generate_videos import visualize

global Player_talk
global NPC_talk

parser_chat = argparse.ArgumentParser(description='Age_dialogue based on KoGPT-2')

parser_chat.add_argument('--chat',
                    action='store_true',
                    default='False',
                    help='response generation on given user input')

parser_chat.add_argument('--model_params',
                    type=str,
                    default='C:/Users/dla12/Documents/Developer/Generative-Conversational-Model-Considering-Age-In-the-Metaverse/python/dialogue.ckpt',
                    help='model binary for starting chat')

parser_chat.add_argument('--train',
                    action='store_true',
                    default=False,
                    help='for training')

audio_data = pd.read_csv('C:/Users/dla12/Documents/Developer/Generative-Conversational-Model-Considering-Age-In-the-Metaverse/python/age.csv')
speech_file = 'C:/Users/dla12/Documents/Developer/Generative-Conversational-Model-Considering-Age-In-the-Metaverse/python/audio.wav'


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

def gesture():
    
    # 0. Check feature type based on the model
    feature_type, audio_dim = check_feature_type(args2.model_file)
    # 1. Load the model
    model = GesticulatorModel.load_from_checkpoint(
        args2.model_file, inference_mode=True)
    # print(model)
    # This interface is a wrapper around the model for predicting new gestures conveniently
    gp = GesturePredictor(model, feature_type)
    # args.audio = "C:/Users/sogang/Documents/development/Python/final/output.wav"
    # args.text = "C:/Users/sogang/Documents/development/Python/final/output.txt"
    # 2. Predict the gestures with the loaded model
    print(args2.audio + args2.text)
    motion = gp.predict_gestures(args2.audio, args2.text)
    # 3. Visualize the results
    motion_length_sec = int(motion.shape[1] / 20)
    
    visualize(motion.detach(), "temp.bvh", "temp.npy", "temp.mp4", 
            start_t = 0, end_t = motion_length_sec, 
            data_pipe_dir = 'C:/Users/dla12/Documents/Developer/Generative-Conversational-Model-Considering-Age-In-the-Metaverse/python/gesticulator/utils/data_pipe.sav')
    
    
    

    # Add the audio to the video
    # command = f"ffmpeg -y -i {args2.audio} -i temp.mp4 -c:v libx264 -c:a libvorbis -loglevel quiet -shortest {args2.video_out}"
    # subprocess.call(command.split())

    # print("\nGenerated video:", args.video_out)
    
    # Remove temporary files
    # for ext in ["bvh", "npy", "mp4"]:
    # for ext in ["npy", "mp4"]:
    #     os.remove("temp." + ext)

def check_feature_type(model_file):
    """
    Return the audio feature type and the corresponding dimensionality
    after inferring it from the given model file.
    """
    params = torch.load(model_file, map_location=torch.device('cpu'))

    # audio feature dim. + text feature dim.
    audio_plus_text_dim = params['state_dict']['encode_speech.0.weight'].shape[1]

    # This is a bit hacky, but we can rely on the fact that 
    # BERT has 768-dimensional vectors
    # We add 5 extra features on top of that in both cases.
    text_dim = 768 + 5

    audio_dim = audio_plus_text_dim - text_dim

    if audio_dim == 4:
        feature_type = "Pros"
    elif audio_dim == 64:
        feature_type = "Spectro"
    elif audio_dim == 68:
        feature_type = "Spectro+Pros"
    elif audio_dim == 26:
        feature_type = "MFCC"
    elif audio_dim == 30:
        feature_type = "MFCC+Pros"
    else:
        print("Error: Unknown audio feature type of dimension", audio_dim)
        exit(-1)

    return feature_type, audio_dim


# def truncate_audio(input_path, target_duration_sec):
#     """
#     Load the given audio file and truncate it to 'target_duration_sec' seconds.
#     The truncated file is saved in the same folder as the input.
#     """
#     audio, sr = librosa.load(input_path, duration = int(target_duration_sec))
#     output_path = input_path.replace('.wav', f'_{target_duration_sec}s.wav')

#     librosa.output.write_wav(output_path, audio, sr)

#     return output_path


parser_ges = ArgumentParser()
parser_ges.add_argument('--audio', type=str, default="C:/Users/dla12/Documents/Developer/Generative-Conversational-Model-Considering-Age-In-the-Metaverse/metaverse/Assets/output/output.wav", help="path to the input speech recording")
parser_ges.add_argument('--text', type=str, default="C:/Users/dla12/Documents/Developer/Generative-Conversational-Model-Considering-Age-In-the-Metaverse/metaverse/Assets/output/chat_output_en.txt",
                    help="one of the following: "
                        "1) path to a time-annotated JSON transcription (this is what the model was trained with) "
                        "2) path to a plaintext transcription, or " 
                        "3) the text transcription itself (as a string)")
parser_ges.add_argument('--video_out', '-video', type=str, default="output/generated_motion.mp4",
                    help="the path where the generated video will be saved.")
parser_ges.add_argument('--model_file', '-model', type=str, default="C:/Users/dla12/Documents/Developer/Generative-Conversational-Model-Considering-Age-In-the-Metaverse/python/demo/models/default.ckpt",
                    help="path to a pretrained model checkpoint")
parser_ges.add_argument('--mean_pose_file', '-mean_pose', type=str, default="../gesticulator/utils/mean_pose.npy",
                    help="path to the mean pose in the dataset (saved as a .npy file)")




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


def transcribe_file(speech_file):
    """Transcribe the given audio file."""
    from google.cloud import speech
    import io
    import os
    os.environ['GOOGLE_APPLICATION_CREDENTIALS']=r"C:/Users/dla12/Documents/Developer/Generative-Conversational-Model-Considering-Age-In-the-Metaverse/python/sesac-371212-227f22f8a69a.json"
    client = speech.SpeechClient()

    with io.open(speech_file, "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=44100,
        language_code="ko-KR",
    )

    response = client.recognize(config=config, audio=audio)

    for result in response.results:
        stt_result = result.alternatives[0].transcript
        print(u"Transcript: {}".format(stt_result))
    return stt_result



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
        parser_chat = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser_chat.add_argument('--max-len',
                            type=int,
                            default=64,
                            help='max sentence length on input (default: 32)')

        parser_chat.add_argument('--batch-size',
                            type=int,
                            default=28,
                            help='batch size for training (default: 96)')
        parser_chat.add_argument('--lr',
                            type=float,
                            default=5e-5,
                            help='The initial learning rate')
        parser_chat.add_argument('--warmup_ratio',
                            type=float,
                            default=0.1,
                            help='warmup ratio')
        return parser_chat

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
        
        # while 1:
        print('user > ')
        chat_input = transcribe_file(speech_file)

        with torch.no_grad():
            ###############################################
            ANNOTATIONS_FILE = 'C:/Users/dla12/Documents/Developer/Generative-Conversational-Model-Considering-Age-In-the-Metaverse/python/age.csv'
            AUDIO_DIR = 'C:/Users/dla12/Documents/Developer/Generative-Conversational-Model-Considering-Age-In-the-Metaverse/python/'
            SAMPLE_RATE = 22050
            NUM_SAMPLES = 22050

            # load back the model
            cnn = CNNNetwork()
            state_dict = torch.load("C:/Users/dla12/Documents/Developer/Generative-Conversational-Model-Considering-Age-In-the-Metaverse/python/age.pth")
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

            # print(age)
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
                chat_output = chat_output + '.'
            print("Chatbot > {}".format(chat_output))
            with open("C:/Users/dla12/Documents/Developer/Generative-Conversational-Model-Considering-Age-In-the-Metaverse/metaverse/Assets/output/chat_output.txt", "w",encoding="UTF-8") as f:
                # Write the response to the output file.
                f.write(chat_output)
                f.close()
            client_tts = texttospeech.TextToSpeechClient()

            synthesis_input = texttospeech.SynthesisInput(text=chat_output)

            voice = texttospeech.VoiceSelectionParams(
                language_code="ko-KR", name="ko-KR-Wavenet-D",ssml_gender=texttospeech.SsmlVoiceGender.MALE
            )

            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.LINEAR16
            )

            response_tts = client_tts.synthesize_speech(
                input=synthesis_input, voice=voice, audio_config=audio_config
            )

            with open("C:/Users/dla12/Documents/Developer/Generative-Conversational-Model-Considering-Age-In-the-Metaverse/metaverse/Assets/output/output.wav", "wb") as out:
                out.write(response_tts.audio_content)
                f.close()
            
            chat_output_en = Translator().translate(chat_output).text
            # print(type(chat_output_en))
            # print(chat_output_en)
            with open("C:/Users/dla12/Documents/Developer/Generative-Conversational-Model-Considering-Age-In-the-Metaverse/metaverse/Assets/output/chat_output_en.txt", "w",encoding="UTF-8") as f:
                # Write the response to the output file.
                f.write(chat_output_en)
                f.close()

            chat_input = str(chat_input)
            chat_output = str(chat_output)
            Player_talk = chat_input
            NPC_talk = chat_output

            print('제스처 생성할까요?')
            
            return Player_talk, NPC_talk
            # gesture(parse_args())
                
                




parser_chat = KoGPT2Chat.add_model_specific_args(parser_chat)
parser_chat = Trainer.add_argparse_args(parser_chat)
args = parser_chat.parse_args()
# logging.info(args)

args2 = parser_ges.parse_args()

def unity_run():
    model = KoGPT2Chat.load_from_checkpoint(args.model_params)
    
    
    return model.chat(), gesture()
    

if __name__ == "__main__":
    unity_run()

