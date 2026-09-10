# Resultado da seleção local

Selecionado: `qwen3:4b-instruct-2507-q4_K_M`, configurado no `.env` e como padrão do código.
Arquivos do modelo: `D:\ferramentas\Ollama-models`, aproximadamente 2,50 GB.
Os três downloads novos têm manifestos e blobs completos no D: (`storage.json`); o Ollama verificou seus hashes durante o download.

## Comparação e limites

Foram experimentados Gemma 3 1B, Qwen 3.5 2B Q4, Qwen3 4B Instruct Q4 e Qwen 3.5 4B Q4.
As primeiras rodadas expuseram erros de gramática e desobediência ao idioma pedido.
O Qwen3 4B Instruct foi selecionado como compromisso entre velocidade e qualidade,
com instruções mais diretas, temperatura 0,3, contexto de 2048 tokens e respostas sem estrutura fixa de aula.
Os prompts mudaram entre rodadas; os arquivos não constituem um ranking científico de modelos.
Mesmo o selecionado ainda suaviza excessivamente algumas correções, por exemplo começando com "Sim"
ao corrigir "I have 25 years". Não há garantia de exatidão gramatical.

## Validação final pelo aplicativo

- `/api/health`: modelo selecionado confirmado; Ollama e Tesseract disponíveis.
- Quatro chamadas reais a `/api/explain`: todas retornaram respostas em português na rodada final.
- Tempo total final: 20,64 s na primeira chamada; 11,02 s, 7,62 s e 7,05 s nas seguintes.
- Rodada anterior: 4,56–6,83 s, mostrando variação conforme o estado da máquina.
- `/api/speak`: WAV gerado, 140764 bytes; não houve avaliação auditiva nem teste de conversa pelo microfone nesta rodada.
- Modelo carregado: aproximadamente 3,22 GB no total, 2,33 GB na GPU (`runtime.json`).
- 12 testes existentes passaram; compilação Python e `git diff --check` passaram.

O suporte CUDA estava incompleto. As DLLs ausentes de CUDA 12 foram recuperadas do ZIP
oficial do Ollama 0.33.3, com SHA256 conferido contra a publicação oficial.
O ZIP e as DLLs foram armazenados no D:. Arquivos existentes e modelos anteriores foram preservados.

O prompt agora permite sugestões e continuidade na conversa. Não foi implementado um novo
temporizador para iniciar conversas durante o silêncio. O comportamento automático de captura
de tela que já existia foi preservado.

Registros: `results*.json`, `final-results.json`, `app-results.json`, `runtime.json`, `storage.json`.
