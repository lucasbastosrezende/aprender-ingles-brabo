# Inglês Brabo — professor mascote (app nativo, 100% local)

App nativo do Windows — **sem navegador, sem janela, sem aba** — que aparece como um
mascote flutuante sempre por cima das outras janelas. Você liga o microfone e/ou o
compartilhamento de tela direto nos botões de cima dele, e ele explica o inglês que
aparece na tela ou que você perguntar em voz alta. Sem API paga, sem chave e sem
mensalidade: OCR, IA, voz e reconhecimento de fala rodam localmente no seu PC.

## Captura rápida e voz alegre

O reconhecimento mantém o modelo multilíngue `small`, aquecido na inicialização, e
usa CUDA quando disponível. No Windows, `WHISPER_CUDA_PATH` aponta para as bibliotecas
CUDA 12 já instaladas com o Ollama. A inicialização testa uma transcrição real de silêncio
e registra o dispositivo usado; se falhar, usa CPU int8 com quatro threads.
O aquecimento não abre nem grava o microfone.

A captura usa blocos de 32 ms e o detector neural Silero local. A probabilidade passa por
suavização temporal e uma histerese (entrada em 0,60, saída em 0,35), então um estalo ou
uma oscilação isolada não abre o microfone e uma consoante curta não encerra a frase.
Espera 480 ms de silêncio, preserva o início da frase em um buffer curto e remove silêncio
excedente antes de reconhecer.
A interrupção exige 608 ms de fala sustentada e um limiar maior durante a reprodução.
Não há cancelamento de eco acústico: fones ainda ajudam quando o alto-falante vaza no mic.
As filas mantêm somente a pergunta pendente mais recente, descartando resultados antigos
e sessões encerradas. O reconhecimento continua enquanto a professora responde.

O Piper fica carregado e guarda até 16 falas em memória. O mascote reproduz a primeira
frase enquanto prepara a próxima. O estilo inspirado em anime eleva o tom em dois
semitons, com pequenas variações de ritmo e tom em perguntas e exclamações. O ajuste é
feito localmente com PyAV, sem aumentar exageradamente a aleatoriedade do modelo.
`VOICE_PITCH_SEMITONES=0` restaura o timbre original. Isso é estilização de áudio e texto;
não é um modelo de atuação emocional. A preferência pelo timbre precisa ser avaliada ouvindo.

Configurações atuais: `MIC_SILENCE_MS=480`, `MIC_MIN_SPEECH_MS=160`,
`MIC_INTERRUPT_MS=608`, `MIC_VAD_ONSET_PROBABILITY=0.60`,
`MIC_VAD_RELEASE_PROBABILITY=0.35`, `MIC_VAD_ONSET_FRAMES=2`,
`MIC_VAD_RELEASE_FRAMES=3` e `MIC_VAD_SMOOTHING_FRAMES=3`.
Elas substituem os antigos parâmetros `MIC_*_CHUNKS`. Se ainda houver disparos com
ruído, aumente `MIC_VAD_ONSET_FRAMES` para 3; se cortar palavras, reduza
`MIC_VAD_RELEASE_PROBABILITY` para 0.30 ou aumente `MIC_VAD_RELEASE_FRAMES`.
`faster-whisper==1.2.1` fixa a interface do detector Silero usada pela captura.

## Como funciona

- O mascote fica sempre visível e por cima das outras janelas; arraste para reposicionar.
- Botão de tela: na primeira vez, pede pra você arrastar um retângulo na área que quer que
  ele leia (igual uma ferramenta de recorte) — só essa área é capturada dali em diante, a
  cada alguns segundos, e passa por **OCR com Tesseract**. Pra escolher outra área depois,
  use "Selecionar área da tela" no menu (⋮) do mascote.
- Botão de microfone: liga o reconhecimento de fala local (**faster-whisper**, multilíngue —
  entende português e inglês misturados na mesma frase) — pergunte em voz alta e ele responde.
- O texto (da tela e/ou da sua pergunta) é enviado para um modelo rodando no **Ollama**
  (localhost), que escreve a explicação em português.
- A explicação aparece num balão de fala e é lida em voz alta com **Piper TTS** (voz local).
- Nada sai da sua máquina, e nada é capturado com os dois botões desligados.

## Onde tudo foi instalado (disco D)

- `D:\ferramentas\Tesseract-OCR` — OCR (extraído do instalador oficial UB-Mannheim)
- `D:\ferramentas\Ollama` + `D:\ferramentas\Ollama-models` — motor de IA local e modelos baixados
- `D:\ferramentas\Piper` — motor de voz local (voz feminina `pt_BR-dii-medium`)
- `D:\ferramentas\Whisper\models` — reconhecimento de fala local multilíngue (faster-whisper),
  baixado automaticamente do Hugging Face na primeira vez que liga o microfone (modelo
  `small`; o tamanho instalado pode variar conforme os arquivos do pacote)
- `D:\ferramentas\Python312` — Python embeddable + pip + libs do projeto
- `D:\ferramentas\InglesBrabo-dist\InglesBrabo\InglesBrabo.exe` — o app empacotado (gerado pelo PyInstaller)

## Rodar o app

**Opção 1 — executável pronto:** abra `D:\ferramentas\InglesBrabo-dist\InglesBrabo\InglesBrabo.exe`.
Ele sobe o Ollama sozinho se não estiver rodando, e o mascote aparece no canto da tela
(sem janela, sem ícone na barra de tarefas). Clique nos botões de microfone/tela em cima
dele para ligar, arraste o mascote para reposicionar, e use os três pontinhos para sair.

**Opção 2 — direto do código (para editar/depurar):**

```powershell
Copy-Item .env.example .env
D:\ferramentas\Python312\python.exe -m pip install -r requirements.txt
D:\ferramentas\Python312\python.exe "desktop.py"
```

Na primeira execução, revise o `.env` caso as ferramentas estejam instaladas em caminhos
diferentes dos padrões em `D:\ferramentas`.

**Reempacotar o .exe depois de alterar o código:**

```powershell
D:\ferramentas\Python312\python.exe -m PyInstaller --noconfirm --clean `
  --distpath "D:\ferramentas\InglesBrabo-dist" --workpath "D:\ferramentas\InglesBrabo-build" `
  InglesBrabo.spec
```

O PySide6 (a interface do mascote) adiciona bastante DLL do Qt à pasta gerada — o `InglesBrabo-dist`
fica maior em **disco** que a versão antiga baseada em navegador embutido (isso é espaço em
disco, não RAM usada durante a execução). Teste o `.exe` novo depois de reempacotar.

## Desempenho: seu PC tem pouca RAM

Este PC tem ~7 GB de RAM. Com Windows + navegador + editores + Discord abertos, sobra pouca
memória livre para o modelo de IA — quando falta RAM, o Windows troca dados com o disco (swap),
e cada explicação pode levar 40-90 segundos em vez de poucos segundos.

**Para respostas rápidas: feche outros programas pesados (navegador, VSCode, Discord) antes de
usar o Inglês Brabo.**

O modelo padrão é o `qwen3:4b-instruct-2507-q4_K_M` (cerca de 2,5 GB em disco),
com contexto de 2048 tokens e sem etapa de pensamento. Os arquivos ficam em
`D:\ferramentas\Ollama-models`. O prompt permite conversa livre, sugestões relacionadas
aos interesses do aluno e respostas curtas, sem um formato fixo de aula.
Isso permite iniciativa nas respostas; não adiciona um temporizador para puxar assunto no silêncio.

Na comparação local de setembro de 2026, os modelos pequenos também cometeram erros
de gramática; a seleção é um compromisso entre qualidade e velocidade, não garantia de
correção. Os registros da comparação ficam em `artifacts/model-test/`.
O consumo em execução é maior que o tamanho do download. A aceleração NVIDIA depende
das bibliotecas CUDA completas do Ollama; uma instalação incompleta pode usar apenas CPU.

## Privacidade e limites

Não compartilhe banco, mensagens privadas, senhas ou dados pessoais na tela enquanto o
compartilhamento estiver ligado. Nada é capturado — nem tela, nem microfone — com os dois
botões desligados; o microfone só grava enquanto o botão está ativo, não fica gravando em
segundo plano escondido. A voz é feminina (Piper, voz "Dii" em pt-BR, gratuita e local).

## Solução de problemas

- **"Ollama não está rodando"**: normalmente o app sobe sozinho; se falhar, rode manualmente:
  `$env:OLLAMA_MODELS="D:\ferramentas\Ollama-models"; D:\ferramentas\Ollama\ollama.exe serve`
- **Texto do OCR vazio ou errado**: capturar a tela inteira lê taskbar, ícones e outras janelas
  junto e confunde o OCR — refaça a seleção (menu ⋮ → "Selecionar área da tela") cobrindo só o
  texto em inglês, e aumente o zoom da página/app antes.
- **Primeira vez que liga o microfone demora / baixando modelo**: o faster-whisper baixa o
  modelo (`small`) do Hugging Face na primeira execução e guarda em
  `D:\ferramentas\Whisper\models`; nas próximas vezes já carrega do disco.
- **Microfone demora pra responder**: confira no log a linha `[voice] Whisper small: CUDA int8_float16`.
  Se aparecer fallback para CPU, verifique `WHISPER_CUDA_PATH`. O modelo é aquecido ao abrir
  o mascote; falas curtas usam uma hipótese de decodificação e detecção automática do idioma.
- **Interrupção (barge-in)**: falar enquanto o mascote está pensando/explicando interrompe a
  fala dele na hora, desde que o volume do microfone fique alto (acima do ruído ambiente,
  aprendido automaticamente) por alguns instantes seguidos — um barulho único não segura
  esse tempo todo, então ruído pontual ou voz de fundo (TV, outra pessoa) não interrompe à
  toa. Se o mic não estiver te ouvindo, abaixe `MIC_LOUD_MULTIPLIER` no `.env`; se ele se
  interromper sozinho ouvindo a própria voz vindo do alto-falante (ou disparar à toa com
  ruído de fundo), suba (ou use fone de ouvido).
- **Sem voz**: confirme que `D:\ferramentas\Piper\piper\piper.exe` e a voz em `D:\ferramentas\Piper\voices\` existem.
- **Trocar de modelo de IA**: baixe outro com `ollama pull <nome>` e mude `OLLAMA_MODEL` no `.env`.
- **Mascote sumiu da tela** (ex.: depois de desconectar um monitor): apague o arquivo
  `.mascot_position.json` na pasta do projeto para ele voltar ao canto padrão.
