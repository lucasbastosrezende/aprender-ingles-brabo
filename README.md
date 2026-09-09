# Inglês Brabo — professor de tela

MVP local para estudar inglês a partir do conteúdo que você escolhe compartilhar.

## O que já faz

- Você escolhe uma aba, janela ou monitor: nada é capturado antes dessa autorização.
- O app mostra uma prévia local e envia uma imagem somente quando você pede uma explicação ou ativa a análise periódica.
- O professor explica em português: sentido no contexto, palavras e expressões, estrutura, ordem natural em português e uma prática curta.
- Você pode conversar sobre a mesma captura no painel de perguntas.

## Rodar no Windows (PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
# edite .env e informe OPENAI_API_KEY
uvicorn app:app --reload
```

Abra `http://127.0.0.1:8000` no Chrome ou Edge. Clique em **Compartilhar tela**, escolha apenas o conteúdo de estudo e depois em **Explicar esta tela**.

## Privacidade e limites

Não compartilhe banco, mensagens privadas, senhas ou dados pessoais. A captura é feita pelo navegador e não funciona em segundo plano sem a permissão que ele exibe. A análise “ao vivo” deste MVP é uma foto a cada intervalo escolhido; isso é mais barato, privado e útil para leitura do que transmitir vídeo contínuo. Para voz de baixa latência e um app instalado, a próxima etapa é empacotar a interface em Electron/Tauri e adicionar a API Realtime.

