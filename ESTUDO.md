# Estudo: módulos, decisões e riscos

Notas para a apresentação. Os números são do teste final
(`resultados/20261121-231858_bge-m3_sem-prefixos_par_bench.csv`, 45 mensagens), salvo quando outra fonte é citada.
Os ids (n04, x07…) são as linhas de `testes/teste_final.csv`.

## Módulos

| arquivo | o que faz |
|---|---|
| `src/config.py` | Lê o `config.toml` (algoritmo) e o `.env` (portas e modelos, o mesmo do compose). Limiar ausente vira `None`, e o pipeline trata como "não calibrado": escala em vez de inventar um valor padrão. |
| `src/faq.py` | Lê o `faq_interno.md` em 18 registros (pergunta + resposta) e liga cada um à sua categoria (`categorias.toml`). Falha se o número de pares não bater com o esperado. |
| `src/armazenamento.py` | O Chroma guarda só os vetores (filhos), e o texto completo (pai) fica em `indice/registros.json`. Agrupa os vetores por registro antes de calcular score e margem. Recusa buscar se o hash do FAQ mudou sem reindexar. |
| `src/ollama_api.py` | Cliente mínimo da API REST do Ollama (`/api/embed`, `/api/chat` com streaming). Converte os tempos da API de nanossegundos para milissegundos e mede o primeiro token no cliente. |
| `src/pipeline.py` | O atendimento: máscara de dados pessoais, regras rápidas, classificação (LLM com esquema JSON), busca, `decidir()`, geração restrita ao registro e verificação. Qualquer falha de serviço escala. |
| `src/tempos.py` | Cronômetro por etapa e estatísticas de latência (mediana, p95…). |
| `src/ambiente.py` | Registra hardware e versões de cada medição, sem hostname, usuário ou caminhos, porque vai para o repositório público. |
| `src/cli.py` | Comandos `index`, `ask`, `chat`, `bench` e `calibrar` (com `--grade`). |

## Decisões e a evidência de cada uma

**1. A decisão de responder é do código, não do LLM.** O LLM classifica e redige, mas quem decide entre responder,
esclarecer e escalar é a `decidir()`, com score, coerência e margem. É uma função pura, com testes em
`tests/test_pipeline.py`. Assim a regra é auditável e reprodutível.

**2. Score mínimo de 0,55.** Foi escolhido no conjunto dev (motivo em `config.toml` e `ia_log.md`) e congelado
antes do teste.
- **Evidência no teste: x07** (integração com Salesforce, fora de escopo) teve score 0,5293, margem 0,0665 e
  coerência verdadeira. Com 0,50 ela seria **respondida errado** com o registro 13. Com 0,55, escalou.
- **Custo: n17** (válida, 1º lugar certo, score 0,5439) foi escalada. **a08** (ambígua, score 0,5458, margem
  0,0025) escalou em vez de virar esclarecimento.
- Essa comparação explica a escolha, mas não serve para mudar o limiar: o teste não pode virar conjunto de calibração.

**3. Coerência entre intenção e registro antes do esclarecimento** (mudança do commit `bfb6791`).
- **Evidência: n04 e n10** (válidas) tinham o 1º lugar errado (registro 13 nas duas). A intenção não batia com a
  categoria do registro 13, então escalaram em vez de responder errado.
- **c05** (bloqueio de conta) só escalou por esse motivo, porque o classificador errou e disse `suporte_status`.
- **Custo: a12** (ambígua, score 0,5517, margem 0,0039) foi incoerente e escalou. Na ordem antiga, viraria
  esclarecimento.

**4. Margem mínima de 0,05 → esclarecer.** Quando dois registros empatam, o sistema pergunta em vez de escolher.
- **Evidência: n01** tinha o 1º lugar errado (registro 2 no lugar do 1), com margem 0,0118. O esclarecimento evitou
  uma resposta errada.
- Os registros 1, 2 e 3 (senha e acesso) estão próximos entre si: n01, n02 e n03 foram todas para esclarecimento.

**5. Classificação de intenção como segunda barreira.**
- A regra de injeção pegou só 1 de 5 (i04). As outras 4 (i05 a i08) foram pegas pelo classificador como
  `fora_de_escopo`.
- A regra de pedido de humano pegou 1 de 4 (h05), e o classificador pegou as outras 3.
- **Custo: n18** (fatura antiga, que o FAQ cobre) foi marcada como `dados_da_conta` e escalou sem passar pela busca.

**6. Pai e filho no índice.** O vetor só aponta para o registro, e o texto que vai para o LLM sai do
`registros.json` pelo id. Isso permite trocar a estratégia de vetor (`par`, `pergunta`, `separado`) sem mudar o texto
usado na resposta, e cada modelo e estratégia tem sua coleção. A comparação entre estratégias e modelos está em
`resultados/` e no `ia_log.md`.

**7. Tudo local.** Ollama e Chroma em Docker, com portas só em `127.0.0.1`, telemetria do cliente Chroma desligada
e dados pessoais mascarados antes do LLM. Nenhuma mensagem sai da máquina.

## Resultado em uma frase

O sistema errou sempre para o lado seguro: as 21 mensagens que deviam escalar escalaram, e as 9 respostas a dúvidas
válidas usaram o registro certo. O preço é responder só 9 de 18 dúvidas válidas. A única resposta discutível é a a10.

## Riscos conhecidos e mitigação

| risco | visto no teste | mitigação atual | próximo passo |
|---|---|---|---|
| Resposta tirada do registro errado | não houve entre as respondidas; n04, n10 e n01 tinham o 1º lugar errado | coerência e margem | mais exemplos rotulados nos registros 1, 2, 3 e 13 |
| A verificação por regras não detecta registro errado | as 10 respostas passaram (`ok`) | só detecta número fora do registro e frase interna; ver `ia_log.md` | verificação de conteúdo (ex.: segunda checagem pelo LLM), a medir |
| Classificador escala dúvida válida | n18 | nenhuma (erro seguro) | revisar a descrição da categoria `dados_da_conta` no prompt |
| Classificador acerta a decisão pelo motivo errado | x02 e x04 como `dados_da_conta`, x06 e a09 como `pedido_humano`, c05 como `suporte_status` | a decisão final foi certa | mostrar o motivo na métrica, e não só a decisão |
| Regras rápidas cobrem pouco | injeção 1 de 5, humano 1 de 4 | o classificador cobriu o resto | a regra é só um atalho; não depender dela |
| Ambíguas quase nunca viram esclarecimento | 0 de 6 | escalam (seguro) | medir com mais casos ambíguos |
| Segunda rodada de esclarecimento sem medição | não exercitada | a lógica tem teste unitário | preencher `resposta_esclarecimento` em todas as linhas que podem esclarecer |
| Resposta gerada afirma dados da conta do cliente | no dev (c04, a02; ver `ia_log.md`) | registrado como risco conhecido | — |
| Calibração com n pequeno | 18 válidas, 8 fora de escopo e 6 ambíguas no teste | limiares fixados antes do teste | mais mensagens reais antes de produção |
| Aquecimento lento | 10,2 s na primeira mensagem | `keep_alive = "30m"` mantém o modelo carregado | — |

## Latência (RTX 4080 SUPER, medianas do CSV)

- Classificação: 143 ms. Embedding: 19 ms. Busca: 24 ms.
- Geração: 533 ms, com o primeiro token em 74 ms.
- Total: 732 ms nas respondidas e 168 ms nas escaladas (sem as 2 pegas pelas regras rápidas, que não chamam o LLM).
- O maior prompt de classificação teve 491 tokens, e o de geração, 281. Os dois cabem com folga no contexto
  configurado (`num_ctx = 4096`).

## Perguntas prováveis

- **Por que não deixar o LLM decidir sozinho?** Porque a decisão por limiar pode ser auditada e medida, e o LLM
  não respondeu `NAO_SEI` em nenhum caso no dev (`ia_log.md`).
- **Por que 0,55 e não 0,50?** Pela x07: com 0,50 ela seria uma resposta errada. O custo é a n17.
- **O teste é independente?** As mensagens foram geradas pelo GPT, fora do Claude, que ajudou a calibrar. O código
  e os limiares ficaram congelados antes da execução, e o Claude só leu o arquivo depois do resultado.
- **O que falta para produção?** Mais dados rotulados, medir a segunda rodada de esclarecimento, uma verificação de
  conteúdo da resposta e o tempo de cada etapa em hardware sem GPU.
