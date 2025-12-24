import sounddevice as sd

# Print a list of all audio devices that PortAudio/sounddevice can see.
# Each entry shows:
#   - the device index (number you pass as `device=`),
#   - the device name (e.g. "MacBook Pro Microphone", "iPhone"),
#   - how many input and output channels it has.
# Use this list to find which index corresponds to your Mac + iPhone setup.
print(sd.query_devices())

# Print the default input/output device indices that sounddevice will use
# if you do NOT specify a device explicitly. This is mainly for debugging,
# so you can see what "default" means on your system.
print("Default input device:", sd.default.device)
