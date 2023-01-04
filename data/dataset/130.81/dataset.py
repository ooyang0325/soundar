#https://github.com/synthizer/synthizer
import sys,time,math
import synthizer
import json
import pyaudio
import wave
import numpy as np
import matplotlib.pyplot as plt
import os

def ran():
    return int.from_bytes(os.urandom(16), byteorder='little') % 400  - 200

dict = {'x': [], 'y': [], 'filename': []}

with synthizer.initialized(log_level=synthizer.LogLevel.DEBUG, logging_backend=synthizer.LoggingBackend.STDERR):
    
    chunk = 4096                       #size of chunk
    sample_format = pyaudio.paInt16    #sample size
    channels = 2                       #number of channels
    fs = 44100                         #sample rates
    t = 1
    dt = 0.1
    temp = 0
    
    if len(sys.argv) != 2:
            print(f"Usage: {sys.argv[0]} <file>")
            sys.exit(1)

    
    # Get our context, which almost everything requires.
    # This starts the audio threads.
    ctx = synthizer.Context()
    # Enable HRTF as the default panning strategy before making a source
    ctx.default_panner_strategy.value = synthizer.PannerStrategy.HRTF
    # A BufferGenerator plays back a buffer:
    generator = synthizer.BufferGenerator(ctx)
    # A buffer holds audio data. We read from the specified file:
    buffer = synthizer.Buffer.from_file(sys.argv[1])
    # Tell the generator to use the buffer.
    generator.buffer.value = buffer
    # A Source3D is a 3D source, as you'd expect.
    source = synthizer.Source3D(ctx)
    # It'll play the BufferGenerator.
    source.add_generator(generator)
    # keep track looping
    generator.looping.value = True
    
    time.sleep(0.5)
    t = 1
    
    for i in range(0, 3000): 
        t = 1 
        x, y = ran(),ran()
        dict['x'].append(x)
        dict['y'].append(y)
        source.position.value = (x/10, y/10, 0)
        print(str(x) + ', ' + str(y) + ', ' + '0\n')
        time.sleep(0.1)
        
        filename = f"output_{i}.wav"            #output sound file name
        p = pyaudio.PyAudio() 
        stream = p.open(format=sample_format, channels=channels, rate=fs, frames_per_buffer=chunk, input=True)
        frames = []
        dict['filename'].append(filename)
        
        while True:
            data = stream.read(chunk)
            frames.append(data)
            if t <= 0:
                break
            t -= dt
            time.sleep(dt)
        
        stream.stop_stream()             
        stream.close()                   
        p.terminate()
        wf = wave.open(filename, 'wb')   
        wf.setnchannels(channels)        
        wf.setsampwidth(p.get_sample_size(sample_format))  
        wf.setframerate(fs)              
        wf.writeframes(b''.join(frames)) 
        wf.close()  
        
with open("loc.json", 'a+') as f:
     json.dump(dict, f, indent = 4)
     f.write('\n')