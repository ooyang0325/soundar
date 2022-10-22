import sys,time,math
import synthizer
import pyaudio
import wave
import numpy as np
import matplotlib.pyplot as plt


chunk = 4096                       #size of chunk
sample_format = pyaudio.paInt16    #sample size
channels = 2                       #number of channels
fs = 44100                         #sample rates           
filename = "output.wav"            #output sound file name
p = pyaudio.PyAudio() 
stream = p.open(format=sample_format, channels=channels, rate=fs, frames_per_buffer=chunk, input=True)
frames = []

theta = 0
delta = math.pi/50
r = 30

if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} <file>")
    sys.exit(1)

with synthizer.initialized(
    log_level=synthizer.LogLevel.DEBUG, logging_backend=synthizer.LoggingBackend.STDERR
):
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
    
    while True:
        theta += delta
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        source.position.value = (x, y, 0)
        print(str(x) + ', ' + str(y) + ', ' + '0\n')
        if theta >= 4*math.pi:
            break
        data = stream.read(chunk)
        frames.append(data)
        time.sleep(0.01)
    
    stream.stop_stream()             
    stream.close()                   
    p.terminate()    
    wf = wave.open(filename, 'wb')   
    wf.setnchannels(channels)        
    wf.setsampwidth(p.get_sample_size(sample_format))  
    wf.setframerate(fs)              
    wf.writeframes(b''.join(frames)) 
    wf.close()  
    

        
       