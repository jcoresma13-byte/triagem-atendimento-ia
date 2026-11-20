# Triagem Inteligente de Atendimento com IA

Projeto desenvolvido para aplicar conceitos fundamentais de **Inteligência Artificial** e **Processamento de Linguagem Natural (PLN)** na organização e priorização de mensagens de atendimento.

## Objetivo
Demonstrar como modelos de aprendizado de máquina (*Machine Learning*) podem transformar dados brutos (mensagens de texto) em informação estruturada para apoiar decisões rápidas, reduzindo o tempo de espera e melhorando a experiência das pessoas atendidas.

## Como funciona (Entrada → Processamento → Saída)
- **Entrada:** Mensagens em texto livre enviadas por usuários ou clientes.
- **Processamento:** Vetorização das palavras com `CountVectorizer` e classificação supervisionada utilizando o algoritmo `Multinomial Naive Bayes` (`scikit-learn`).
- **Saída:** Categorização automática da solicitação em *Suporte Técnico*, *Comercial*, *Feedback Positivo* ou *Urgência*.

## Tecnologias utilizadas
- Python 3
- Scikit-learn

## Como executar
1. Instale a biblioteca necessária:
   pip install scikit-learn
2. Execute o script:
   python classificador_ia.py

## Competências aplicadas
- Lógica de programação e estruturação de fluxo de dados;
- Fundamentos de Machine Learning supervisionado;
- Empatia aplicada à tecnologia: uso da automação para priorizar urgências e humanizar o atendimento.
