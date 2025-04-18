# Real-time EEG Monitoring with Python

This project aims to create an EEG GUI for real-time plotting and to be the BCI approach for sending the predicted SSVEP command to Unity.
This GUI can stream EEG data via LSL from an EEG device.

![Project Screenshot](Screenshot.png)

Please set up follow instructions below:

1. Download VScode on your PC [VScode](https://code.visualstudio.com/download)
2. Download Python V.3.11 [Python V3.11](https://www.python.org/downloads/release/python-3110/)
3. Download this repository
```
git clone https://github.com/pipogood/BCI-realtime-GUI.git
```
4. Create local environments in VS Code using virtual environments (venv), you can follow these steps: open the Command Palette (Ctrl+Shift+P), search for the Python: Create Environment command, and select it.
[venv setup tutorial](https://code.visualstudio.com/docs/python/environments)

   4.1 Activate .venv in terminal
   ```
   .\.venv\Scripts\activate
   ```

   4.2 If you use conda or install manually please install required library
   ```
   pip install -r requirements.txt
   ```

5. Run biosemi_stream.ipynb to simulate EEG data streaming     
6. Run main.py to start real-time GUI:
```
python main.py
```

**For GUI Algorithm detail, please read refer to Instruction.pdf**   


# Authors
1. Sirapakit Limtragooltongchai (sirapakit.lim@gmail.com)
2. Suparach Intarasopa (suparach.int@gmail.com)
   
Neuroscience Center for Research and Innovation
King Mongkut's University of Technology Thonburi, Thailand
