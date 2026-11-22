# Assistente de triagem de suporte (RabbITech) com Ollama e RAG

Teste técnico da Semana 2. É uma prova de conceito que roda 100% local: para cada mensagem de cliente, o sistema
decide entre **responder** com base no FAQ interno, **pedir um esclarecimento** ou **escalar para um humano**.
O tempo de cada etapa é medido.

- Modelos: `llama3.1:8b` (classificação e geração) e `bge-m3` (embedding), servidos pelo Ollama 0.40.16.
- Índice de vetores: Chroma 1.5.9, com distância de cosseno.
- Os dois serviços rodam em Docker e só escutam em `127.0.0.1`. Nenhuma mensagem sai da máquina.

## Como funciona

```
mensagem
  → normalizar e mascarar dados pessoais (e-mail, CPF, telefone)
  → regras rápidas: pedido de humano, urgência, injeção          → ESCALAR
  → classificação de intenção (LLM, saída JSON com esquema)     → intenção especial: ESCALAR
  → embedding + busca no Chroma + agrupamento por registro do FAQ
  → decisão (código):
        score do 1º < 0,55                       → ESCALAR
        intenção não bate com a categoria do 1º  → ESCALAR
        margem entre 1º e 2º < 0,05              → ESCLARECER (uma vez; na segunda, ESCALAR)
        senão                                    → RESPONDER
  → geração restrita ao registro escolhido + verificação por regras
```

Se o Ollama ou o Chroma falharem, a mensagem é escalada. Os limiares e a escolha de modelo e estratégia estão em
`config.toml`, com o motivo de cada um.

## Resultado do teste final

O conjunto tem 45 mensagens (`testes/teste_final.csv`) e foi gerado pelo GPT, fora do assistente que calibrou o
sistema. O teste rodou com o código e os limiares congelados (commit `bfb6791`), em uma única repetição.
Resultado completo: `resultados/20261121-231858_bge-m3_sem-prefixos_par_bench.csv`. Hardware e versões:
`resultados/20261121-231858_bge-m3_sem-prefixos_par_ambiente.json`.

| grupo | n | decisão igual à esperada | o que aconteceu |
|---|---|---|---|
| dentro do FAQ | 18 | 9 | 9 respondidas, 5 em esclarecimento, 4 escaladas |
| fora de escopo | 8 | 8 | todas escaladas |
| ambígua | 6 | 0 | 5 escaladas, 1 respondida (a10) |
| depende da conta | 4 | 4 | todas escaladas |
| pede humano | 4 | 4 | todas escaladas |
| injeção | 5 | 5 | todas escaladas |
| **total** | **45** | **30 (67%)** | |

- **As 21 mensagens que deviam escalar escalaram:** fora de escopo, conta, humano e injeção.
- **As 9 respostas a dúvidas válidas usaram o registro certo** e passaram na verificação.
- **Uma resposta discutível:** a10 ("Meu arquivo não entrou no sistema.") estava rotulada como ambígua e foi
  respondida com o registro 12, com score 0,6295 e margem 0,0804.
- **O sistema é conservador:** só 9 de 18 dúvidas válidas foram respondidas. As outras 9 foram:
  - 5 esclarecimentos por margem pequena;
  - 2 escaladas por intenção incoerente, em que o 1º lugar da busca estava errado (n04, n10);
  - 1 escalada por score abaixo do limiar (n17, 0,5439);
  - 1 escalada porque o classificador a marcou como dado da conta (n18).
- **O 1º lugar da busca estava certo em 14 de 17** dúvidas válidas. A n18 não chegou à busca.
- **O esclarecimento em duas rodadas não foi exercitado.** As 5 mensagens que foram para esclarecimento não tinham
  resposta de esclarecimento no arquivo, e as ambíguas não chegaram a esclarecimento.

**Latência** em uma RTX 4080 SUPER, calculada do CSV (mediana, com mínimo e máximo):

| etapa | n | mediana (ms) | mín–máx (ms) |
|---|---|---|---|
| classificação | 43 | 143 | 105–202 |
| embedding | 26 | 19 | 12–59 |
| busca | 26 | 24 | 19–65 |
| geração | 10 | 533 | 414–812 |
| primeiro token da geração | 10 | 74 | 64–115 |
| total, mensagens respondidas | 10 | 732 | 602–1016 |
| total, mensagens escaladas (sem as 2 das regras rápidas) | 28 | 168 | 126–224 |

O aquecimento (carga dos modelos na GPU) levou 10,2 s e ficou fora dessas estatísticas. As 2 mensagens pegas pelas
regras rápidas (h05, i04) não chamam o LLM.

Limitações e riscos conhecidos: veja `ESTUDO.md`. As decisões tomadas com ajuda de IA estão em `ia_log.md`.

## Requisitos

- Docker Desktop (no Windows, com WSL2).
- GPU NVIDIA é opcional. Sem ela tudo funciona, só que mais devagar. Os números acima são com GPU.
- Python 3.11 ou mais novo (o código usa `tomllib`). Testado com 3.14.6.
- Cerca de 6,1 GB de disco para os modelos (`llama3.1:8b` 4,9 GB, `bge-m3` 1,2 GB).

## Como rodar

### 1. Subir os serviços

No Windows com GPU NVIDIA, o script confere WSL, driver e Docker (e abre o Docker Desktop se estiver fechado),
cria o `.env`, sobe os containers e baixa os modelos na primeira vez:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\subir_windows.ps1
```

Para fazer à mão em qualquer sistema:

```bash
cp .env.example .env
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d   # com GPU NVIDIA
docker compose up -d                                                     # sem GPU
```

O serviço `modelos` roda uma vez, baixa os dois modelos e termina. No Mac, o Docker não usa a GPU, então use o
comando sem GPU. Esse caminho não foi medido.

### 2. Ambiente Python

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux/Mac
pip install -r requirements-dev.txt
```

### 3. Indexar o FAQ (na primeira vez e sempre que o FAQ mudar)

```bash
python -m src.cli index
```

O índice guarda o hash do FAQ. Se o FAQ mudar sem reindexar, a busca recusa rodar.

### 4. Usar

```bash
python -m src.cli ask "o link para trocar a senha expirou"    # uma mensagem, com candidatos e tempos
python -m src.cli chat                                        # conversa, com uma rodada de esclarecimento
python -m src.cli bench --conjunto dev                        # perguntas_exemplo.csv
python -m src.cli bench --perguntas testes/teste_final.csv --conjunto teste
python -m src.cli calibrar --grade                            # grade score x margem no conjunto dev
pytest                                                        # testes das partes puras
```

O `bench` grava em `resultados/` um CSV (sem o texto das mensagens) e um JSON com hardware e versões
(sem hostname, usuário ou caminhos). Todos os comandos aceitam `--estrategia`, `--modelo-embedding` e
`--sem-prefixos` para as comparações.

### Parar e apagar

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml stop     # para; modelos e índice ficam nos volumes
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d    # volta
docker compose -f docker-compose.yml -f docker-compose.gpu.yml down -v  # apaga containers e volumes
```

Depois do `down -v`, o próximo `up` baixa os modelos de novo (cerca de 6 GB) e é preciso rodar o `index`.

## Estrutura

```
base/faq_interno.md       FAQ (18 perguntas) fornecido pela RabbITech
base/categorias.toml      registro do FAQ → categoria de intenção
config.toml               limiares, estratégia, contexto do LLM (com o motivo de cada valor)
src/                      pipeline, índice, cliente do Ollama, CLI
tests/                    testes unitários (pytest)
perguntas_exemplo.csv     conjunto dev (62 mensagens, escrito pelo Claude), usado para calibrar
testes/teste_final.csv    conjunto de teste final (45 mensagens, gerado pelo GPT)
resultados/               CSVs e JSONs de cada medição
scripts/subir_windows.ps1 instalação e subida no Windows
ia_log.md                 registro do uso de IA
ESTUDO.md                 módulos, decisões e riscos
docs/                     documento final da entrega
```

## Sobre o FAQ

O arquivo `base/faq_interno.md` foi fornecido pela RabbITech para este teste. A inclusão dele neste repositório
público foi autorizada pela empresa.

## Uso de IA

O código foi escrito com a ajuda do Claude Code. O conjunto dev foi escrito pelo Claude, e o conjunto de teste final, pelo
GPT. As sugestões aceitas, mudadas ou recusadas, com o motivo de cada uma, estão em `ia_log.md`.
