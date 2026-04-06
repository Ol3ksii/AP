# Deteção e Classificação de Autoria em Texto: Humanos vs IA
# UC: Aprendizagem Profunda (Mestrado em Inteligência Artificial) | Grupo 7 | 2025/2026**

---

## Constituição do Grupo
* Carlos Tiago da Costa Ribeiro - A65433
* Oleksii Tantsura - A102131
* Vicente de Carvalho Castro - PG60395
* Martim de Oliveira e Melo Ferreira - PG60391

---

## Descrição do Projeto
O objetivo deste projeto é o desenvolvimento de modelos de *Machine Learning* e *Deep Learning* capazes de classificar textos curtos (100-120 caracteres) em 5 classes distintas: 
1. Human (Texto escrito por humanos)
2. OpenAI (Modelos GPT)
3. Google (Modelos Gemma, Gemini)
4. Meta (Modelos Llama)
5. Anthropic/Mistral (Modelos Claude e Mistral-7B)

O projeto seguiu uma evolução progressiva, partindo de implementações manuais em **NumPy** até ao uso de **Transformers (BERT)** em **PyTorch**.

---

## Estrutura do Repositório

O repositório está organizado de acordo com as fases de submissão e o pipeline de dados:

.
├── Dataset/                     # Gestão e construção dos dados
│   ├── data_collection.ipynb    # Recolha inicial de múltiplas fontes (HC3, M4, etc.)
│   └── data_collection_fix.ipynb # Limpeza, filtragem (100-120 chars) e balanceamento
│
├── Submissão_1/                 # Modelos Iniciais
│   ├── subm1-g7-MIA-A.ipynb     # Implementação de raiz em NumPy (DNN e Reg. Logística)
│   └── subm1-g7-MIA-B.ipynb     # Primeira abordagem em PyTorch (Embeddings + LSTM)
│
├── Submissão_2/                 # Refinamento e Complexidade
│   ├── subm2-g7-MIA-A.ipynb     # Otimização NumPy com TF-IDF e Ensemble
│   └── subm2-g7-MIA-B.ipynb     # Bidirectional LSTM com Max/Mean Pooling
│
├── Submissão_3/                 # Modelos Finais (Estado da Arte)
│   ├── C1.ipynb                 # Fine-tuning de Transformers (BERT-base)
│   └── C2.ipynb                 # Otimização de hiperparâmetros e pesos de classe
│
├── Relatorio_Final_G7.pdf       # Relatório técnico completo (estilo LNCS)
└── README.md                    # Identificação e guia do repositório

---

## Tecnologias Utilizadas

Linguagem: Python 3.10+
Bibliotecas Base: NumPy, Pandas, Scikit-learn
Deep Learning: PyTorch
NLP: HuggingFace Transformers (BERT), Tokenizers
Ambiente: Jupyter Notebooks

---

## Como Executar

1) Os notebooks estão preparados para ser executados sequencialmente.
2) Certifique-se de que possui as dependências instaladas via pip install torch transformers datasets pandas numpy.
3) Os dados de treino e validação devem estar na mesma diretoria dos notebooks (conforme especificado no código).
4) Nota: Para os notebooks da Submissão 3 (C1/C2), recomenda-se a utilização de uma GPU para o processo de fine-tuning.

---

## Principais Resultados

Os nossos modelos finais (C1 e C2) demonstraram uma capacidade elevada de generalização, superando as baselines de NumPy em textos de baixa densidade de caracteres através da captura de relações semânticas profundas. 
Os detalhes das métricas e rankings encontram-se no Relatório Final.
