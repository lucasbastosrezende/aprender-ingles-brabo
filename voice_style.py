"""Lightweight anime-inspired voice styling, processed locally with PyAV."""
import io
import wave

import av
import numpy as np


def style_voice(audio, text, semitones=2.0):
    """Raise pitch modestly, keeping tempo independent and PCM output valid."""
    if semitones == 0:
        return audio
    ending = text.rstrip().rstrip('"\u201d\u2019')
    emphasis = 0.6 if ending.endswith("!") else 0.25 if ending.endswith("?") else 0.0
    tempo = 1.055 if ending.endswith("!") else 1.015 if ending.endswith("?") else 1.035
    pitch = 2 ** ((semitones + emphasis) / 12)
    with wave.open(io.BytesIO(audio), "rb") as reader:
        rate = reader.getframerate()
        if reader.getsampwidth() != 2 or reader.getnchannels() != 1:
            raise ValueError("A estilização de voz requer PCM mono de 16 bits.")
        samples = np.frombuffer(reader.readframes(reader.getnframes()), dtype=np.int16)
    frame = av.AudioFrame.from_ndarray(samples.reshape(1, -1), format="s16", layout="mono")
    frame.sample_rate = rate
    frame.pts = 0
    graph = av.filter.Graph()
    source = graph.add("abuffer", f"time_base=1/{rate}:sample_rate={rate}:sample_fmt=s16:channel_layout=mono")
    nodes = [source, graph.add("asetrate", str(round(rate * pitch))),
             graph.add("aresample", str(rate)),
             graph.add("atempo", str(tempo / pitch)),
             graph.add("aformat", "sample_fmts=s16:channel_layouts=mono"),
             graph.add("abuffersink")]
    for left, right in zip(nodes, nodes[1:]):
        left.link_to(right)
    graph.configure()
    source.push(frame)
    source.push(None)
    chunks = []
    while True:
        try:
            chunks.append(nodes[-1].pull().to_ndarray().tobytes())
        except (av.error.BlockingIOError, av.error.EOFError):
            break
    if not chunks:
        raise RuntimeError("A estilização não produziu áudio.")
    result = io.BytesIO()
    with wave.open(result, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(rate)
        writer.writeframes(b"".join(chunks))
    return result.getvalue()
