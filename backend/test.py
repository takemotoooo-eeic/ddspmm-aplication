import requests

wav_path = "test/AuMix_11_Maria_ob_vc.wav"   # 実際のファイル名に置き換えてください
midi_path = "test/Sco_11_Maria_ob_vc.mid"   # 実際のファイル名に置き換えてください

url = "http://localhost:8888/backend-api/ddsp/train"

files = {
    "wav_file": ("your_audio.wav", open(wav_path, "rb"), "audio/wav"),
    "midi_file": ("your_midi.mid", open(midi_path, "rb"), "audio/midi"),
}

response = requests.post(url, files=files)

print("Status code:", response.status_code)
print("Response:", response.json())

url = "http://localhost:8888/backend-api/ddsp/generate"

for i, feature in enumerate(response.json()["features"]):
    response = requests.post(url, json=feature)
    print("Status code:", response.status_code)
    with open(f"test/generated_{i}.wav", "wb") as f:
        f.write(response.content)
