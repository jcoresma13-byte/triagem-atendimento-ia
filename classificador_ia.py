"""
Projeto: Triagem Inteligente de Mensagens com IA (PLN Básico)
Objetivo: Classificar solicitações recebidas (Suporte Técnico, Dúvida Comercial,
Elogio/Feedback ou Urgência) para agilizar o atendimento humanizado.
"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# 1. DADOS DE TREINAMENTO (Exemplos rotulados de mensagens reais)
mensagens_treino = [
    "O sistema travou e não consigo acessar minha conta",
    "Erro na tela ao tentar emitir o relatório financeiro",
    "A internet caiu e o aplicativo não abre",
    "Gostaria de saber o valor do plano mensal e formas de pagamento",
    "Vocês fazem orçamento para empresas pequenas?",
    "Quero contratar o serviço para minha equipe",
    "Parabéns pelo atendimento rápido e atencioso",
    "Gostei muito da nova atualização, ficou bem mais fácil de usar",
    "Excelente trabalho da equipe de suporte",
    "Preciso resolver isso agora, meu prazo vence em 10 minutos",
    "Urgente: o servidor parou e os clientes estão sem acesso",
    "Emergência no sistema principal, precisamos de ajuda imediata",
]

categorias_treino = [
    "Suporte Técnico",
    "Suporte Técnico",
    "Suporte Técnico",
    "Comercial",
    "Comercial",
    "Comercial",
    "Feedback Positivo",
    "Feedback Positivo",
    "Feedback Positivo",
    "Urgência",
    "Urgência",
    "Urgência",
]

# 2. PROCESSAMENTO (Vetorização de texto + Treinamento do modelo Naive Bayes)
vetorizador = CountVectorizer()
X_treino = vetorizador.fit_transform(mensagens_treino)

modelo_ia = MultinomialNB()
modelo_ia.fit(X_treino, categorias_treino)


def classificar_mensagem(texto: str) -> str:
    """Recebe uma nova mensagem e retorna a categoria prevista pela IA."""
    texto_vetorizado = vetorizador.transform([texto])
    previsao = modelo_ia.predict(texto_vetorizado)[0]
    return previsao


# 3. SAÍDA (Teste prático com novas mensagens)
if __name__ == "__main__":
    novas_mensagens = [
        "Não consigo fazer login, aparece erro de senha inválida",
        "Qual o preço para contratar 5 licenças?",
        "Muito obrigado pela atenção, vocês resolveram tudo!",
        "Urgente! O sistema saiu do ar bem na hora da apresentação",
    ]

    print("=== TRIAGEM INTELIGENTE DE ATENDIMENTO ===\n")
    for msg in novas_mensagens:
        categoria = classificar_mensagem(msg)
        print(f"Mensagem: {msg}")
        print(f"-> Categoria prevista pela IA: {categoria}\n")
