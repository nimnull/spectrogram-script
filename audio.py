import pyaudio
import os
import struct
import numpy as np
import matplotlib.pyplot as plt
import sys
import struct
import pyaudio
import wave
import time
import random
import noise
import traceback
import json
from numpy.fft import fft, ifft
from array import array
from tkinter import TclError

with open('note_frequencies.json', 'rb') as f:
    note_frequencies = json.loads(f.read())
    

def noise_linspace(l, o=1):
    r = random.random() + random.randint(0, 1000)
    return list(map(lambda x: (noise.pnoise1(x+r, o)+1.0)/2.0, l))
    
    
class Spectrogram():
    def __init__(self, min_hertz=40, max_hertz=None, audio_format=pyaudio.paInt16, channels=1, rate=44100, chunk_size=8192, fft_scaling_factor=(2**25)):
        # create matplotlib figure and axes
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, figsize=(15, 7))
        
        self.fft_scaling_factor = fft_scaling_factor
        
        self.audio_format = audio_format
        self.audio_channels = channels
        self.audio_rate = rate
        self.audio_chunk_size = chunk_size
        self.audio_rate_per_frame = self.audio_rate/self.audio_chunk_size
        self.min_hertz = min_hertz
        if max_hertz == None:
            self.max_hertz = self.audio_chunk_size
            
        else:
            self.max_hertz = max_hertz
            
        self.recording = False
        self.record_time = None
        self.record_start_time = None

        self.raw_audio_buffer = array('h')
        self.spectrogram_buffer = []
        
        # pyaudio class instance
        self.pyaudio = pyaudio.PyAudio()

        # stream object to get data from microphone
        self.stream = self.pyaudio.open(
            format=self.audio_format,
            channels=self.audio_channels,
            rate=rate,
            input=True,
            output=True,
            frames_per_buffer=self.audio_chunk_size
        )
        
        self.audio_sample_width = self.pyaudio.get_sample_size(self.audio_format)
        self.audio_max_amplitude = int(2.0**(8.0*self.audio_sample_width)/2.0-(8.0**(self.audio_sample_width-1.0)))
        
        # variable for plotting
        x = np.arange(0, 2*self.audio_chunk_size, 2)
        xf = np.linspace(0, self.audio_rate, self.max_hertz)

        # create a line object with random data
        self.line_spectrum, = self.ax1.plot(x, np.random.rand(self.audio_chunk_size), '-', lw=2)

        # create semilogx line for spectrum
        self.line_fft, = self.ax2.semilogx(xf, np.random.rand(self.max_hertz), '-', lw=2)

        # basic formatting for the axes
        self.ax1.set_title('AUDIO WAVEFORM')
        self.ax1.set_xlabel('samples')
        self.ax1.set_ylabel('volume')
        self.ax1.set_ylim(-self.audio_max_amplitude, self.audio_max_amplitude)
        self.ax1.set_xlim(0, self.audio_chunk_size)

        # format spectrum axes
        self.ax2.set_xscale('log')
        #self.ax2.set_xlim(self.min_hertz, self.max_hertz)
        self.ax2.set_yscale('log')

        # show the plot
        plt.show(block=False)
        
        print('stream started')
        
    def get_peak_frequency(self, data):
        # get the most significant frequency
        data = np.abs(data[:int(self.audio_chunk_size/self.audio_rate_per_frame)])
        f = np.argmax(data)
        f = f*self.audio_rate_per_frame
        return f 
    
    def get_frequency(self, data, frequency):
        data = np.abs(data[:int(self.audio_chunk_size/self.audio_rate_per_frame)])
        v = data[int(frequency/self.audio_rate_per_frame)]/self.fft_scaling_factor
        return v
    
    def set_frequency(self, data, frequency, value):
        data[int(frequency/self.audio_rate_per_frame)] = value*self.fft_scaling_factor
        
    def save_audio_to_file(self, data, filename):
        # pack data into valid binary format
        data = struct.pack('<' + ('h'*len(data)), *data)

        # setup wav file and save
        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.audio_channels)
        wf.setsampwidth(self.audio_sample_width)
        wf.setframerate(self.audio_rate)
        wf.writeframes(data)
        wf.close()
        
    def plot_spectrogram(self):
        print(len(self.spectrogram_buffer[0]))
        self.spectrogram_buffer = np.rollaxis(np.array(self.spectrogram_buffer), 1)
        plt.imshow(self.spectrogram_buffer**0.1, origin='lower', cmap='jet', aspect='auto', extent=[0, self.recording_time, 0, self.max_hertz])
        plt.show()
        
    def record_and_graph(self, record_time, fft_callback=None, spectrum_callback=None, save_file=None):
        self.recording = True
        self.recording_start_time = time.time()
        self.recording_time = record_time
        
        frame_count = 0
        
        try:
            # record
            while time.time()-self.recording_start_time < self.recording_time:
                # binary data
                data = self.stream.read(self.audio_chunk_size)       
                
                # unpack binary data
                data_int = struct.unpack(str(self.audio_chunk_size) + 'h', data)
                
                # compute FFT
                fs = fft(data_int)[:int(self.audio_chunk_size/self.audio_rate_per_frame)]
                fs[np.abs(fs) < 5000] = 0
                if fft_callback != None:
                    fft_callback(self, fs)

                # compute inverse FFT
                data_np = ifft(fs, 8192).astype(int)
                if spectrum_callback != None:
                    spectrum_callback(self, data_np)
                 
                # update lines
                self.line_spectrum.set_ydata(data_np)
                self.line_fft.set_ydata(np.abs(fs)/self.fft_scaling_factor)
                
                fmt_data = np.clip(data_np.astype(int), -self.audio_max_amplitude, self.audio_max_amplitude)

                # pack data into array
                data = struct.pack(str(int(self.audio_chunk_size)) + 'h', *list(fmt_data))
                data_bin = array('h', data)    
                self.raw_audio_buffer.extend(data_bin)
                
                freqs = np.abs(fs)/self.fft_scaling_factor
                self.spectrogram_buffer.append(freqs)                
                
                # update figure canvas
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
                frame_count += 1
            
        except (KeyboardInterrupt, TclError):
            pass
                
        except:
            traceback.print_exc()
            
        finally:
            print('stream stopped')
            frame_rate = frame_count/(time.time()-self.recording_start_time)
            print('average frame rate = {:.0f} FPS'.format(frame_rate))
            
            self.recording = False
            self.recording_start_time = None
            
            self.ax1.remove()
            self.ax2.remove()
            
            if save_file != None:
                self.save_audio_to_file(self.raw_audio_buffer, save_file)
        
    
def test(spect, data):
    # move up the scale as time progesses
    try:
        duration = (time.time()-spect.recording_start_time)/spect.recording_time
        freq = list(note_frequencies.values())[int(duration*len(note_frequencies))]
        spect.set_frequency(data, freq, 0.1)
        
        #print("peaking at {}Hz".format(spect.get_peak_frequency(data)))
        
    except IndexError:
        return
    
    
def main():
    s = Spectrogram()
    s.record_and_graph(10, save_file='demo.wav')
    s.plot_spectrogram()

    
if __name__ == '__main__':
    main()