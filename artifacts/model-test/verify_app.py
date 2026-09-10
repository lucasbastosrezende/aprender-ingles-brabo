import json,time
from pathlib import Path
import requests

root=Path(__file__).resolve().parent
for attempt in range(20):
    try:
        health=requests.get('http://localhost:8000/api/health',timeout=5)
        health.raise_for_status()
        assert health.json()['model']=='qwen3:4b-instruct-2507-q4_K_M'
        break
    except (requests.RequestException,AssertionError):
        time.sleep(1)
else:
    raise RuntimeError('Updated app did not start')
print('health',health.json(),flush=True)
results=[]
for question in [
    'Eu falei "I have 25 years" para dizer minha idade. Está certo?',
    'Qual a diferença entre "I used to play" e "I am used to playing"?',
    'Hoje estou cansado. Não quero aula chata. Gosto de Minecraft.',
    'Como digo "estou com fome"? É "I am with hunger"?',
]:
    start=time.perf_counter()
    response=requests.post('http://localhost:8000/api/explain',json={'question':question},timeout=180)
    response.raise_for_status()
    item={'question':question,'seconds':round(time.perf_counter()-start,2),**response.json()}
    results.append(item)
    print(json.dumps(item,ensure_ascii=False),flush=True)
    (root/'app-results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
response=requests.post('http://localhost:8000/api/speak',json={'text':'Podemos praticar uma conversa curta sobre Minecraft.'},timeout=30)
response.raise_for_status()
assert response.content[:4]==b'RIFF'
(root/'voice-check.wav').write_bytes(response.content)
print('voice_wav_bytes',len(response.content),flush=True)
state=requests.get('http://localhost:11434/api/ps',timeout=5).json()
(root/'runtime.json').write_text(json.dumps(state,indent=2),encoding='utf-8')
print(json.dumps(state),flush=True)
