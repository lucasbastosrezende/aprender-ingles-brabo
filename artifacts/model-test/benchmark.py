import json
import time
import sys
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parent
SYSTEM = '''Você é uma professora de inglês que conversa com Lucas em português brasileiro.
Converse com naturalidade e curiosidade. Responda ao que ele disse antes de ensinar.
Use inglês em exemplos e nos momentos de prática, com apoio em português quando necessário.
Prefira 2 a 4 frases, sem títulos, listas, bordões ou elogios automáticos.
Você pode sugerir um assunto, compartilhar um exemplo ou propor uma situação ligada aos interesses dele.
Faça no máximo uma pergunta por vez, apenas quando ajudar a conversa; nem toda resposta precisa de pergunta.
Corrija com delicadeza e explique um ponto por vez. Não invente informações sobre Lucas nem texto da tela.
Se ele quiser conversar sem captura, converse normalmente. Só peça uma captura melhor quando precisar ler algo ilegível.
Não finja experiências pessoais reais. Respeite quando ele quiser silêncio.'''
if '--app-prompt' in sys.argv:
    sys.argv.remove('--app-prompt')
    sys.path.insert(0, str(ROOT.parent.parent))
    from app import SYSTEM_PROMPT
    SYSTEM = SYSTEM_PROMPT
CASES = [
    'Hoje estou cansado. Não quero uma aula chata. Gosto de jogos, principalmente Minecraft.',
    'Eu falei "I have 25 years". Está certo? Me explica sem complicar.',
    'Qual a diferença entre "I used to play" e "I am used to playing"?',
    'Conversa anterior:\nAluno: Meu jogo favorito é Minecraft.\nProfessora: Podemos praticar inglês construindo uma casa.\nAluno: Pode ser.\nAgora continua a conversa e sugere o próximo passo.',
]
result_path = ROOT / ('results-' + sys.argv[1].replace(':', '-').replace('/', '-') + '.json' if len(sys.argv) > 1 else 'results.json')
results = []
for model in (sys.argv[1:] or ['gemma3:1b', 'qwen3.5:2b-q4_K_M']):
    for index, prompt in enumerate(CASES):
        start = time.perf_counter()
        first = None
        answer = ''
        with requests.post('http://127.0.0.1:11434/api/generate', json={
            'model': model, 'system': SYSTEM, 'prompt': prompt + '\nResponda em português brasileiro, com inglês apenas nos exemplos.', 'think': False,
            'stream': True, 'keep_alive': '5m',
            'options': {'num_ctx': 2048, 'num_predict': 220, 'temperature': 0.7, 'seed': 42},
        }, stream=True, timeout=180) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                item = json.loads(line)
                if item.get('error'):
                    raise RuntimeError(item['error'])
                if item.get('response') and first is None:
                    first = time.perf_counter() - start
                answer += item.get('response', '')
                if item.get('done'):
                    final = item
        result = {'model': model, 'case': index + 1, 'prompt': prompt,
                  'first_token_s': round(first or 0, 2), 'total_s': round(time.perf_counter() - start, 2),
                  'tokens_s': round(final.get('eval_count', 0) / max(final.get('eval_duration', 1) / 1e9, 0.001), 2),
                  'done_reason': final.get('done_reason'), 'answer': answer}
        results.append(result)
        result_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(result, ensure_ascii=False), flush=True)
    requests.post('http://127.0.0.1:11434/api/generate', json={'model': model, 'keep_alive': 0}, timeout=30).raise_for_status()
