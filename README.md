# EH5TinyML
In this project i will aquire data by recording the audio of different engines that run normaly at idle and engines that run poorly at idle with a faulty ignition system.

With my Particle Photon 2 connected to a PDM micrphone i will aquire as much audio data as possible, sending the data over TCP to be saved on my laptop.
The data will be collected from different cars at idle. Some will idle normally with no faulty ignition system, others will have a faulty iginition system implemented.
I have been given permission by a local workshop and car gallery (KP Biler / Mikkel O. Pedersen Racing) to use their stock of sale cars and some high performance sports cars for this project. 

I plan to record audio from a car f.eks. CAR_01_NORMAL for up to 5 minutes at idle. Then record the same amount for CAR_01_FAULTY at idle.
I will then segment these audio files down to smaller sizes (maybe 2-4 seconds) to include in a large dataset.

The goal is to train and deploy a lightweight machine learning model on the Particle Photon 2 that can classify idle engine sounds in real time as either “normal” or “faulty”.
