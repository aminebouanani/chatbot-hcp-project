document.addEventListener('DOMContentLoaded', () => {
    // --- Éléments du DOM ---
    const chatbotContainer = document.getElementById('chatbot-container');
    const openChatBtn = document.getElementById('open-chat-btn');
    const closeBtn = document.getElementById('close-btn');
    const fullscreenBtn = document.getElementById('fullscreen-btn');
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const chatBody = document.getElementById('chat-body');

    // --- URL de l'API Backend ---
    // Assurez-vous que votre serveur Flask fonctionne sur cette adresse et ce port
    const API_URL = 'http://127.0.0.1:5000/ask';

    // --- Gestion de l'interface ---

    openChatBtn.addEventListener('click', () => {
        chatbotContainer.classList.add('visible');
        openChatBtn.style.display = 'none';
        // Ajoute un message de bienvenue seulement la première fois qu'on ouvre
        if (chatBody.children.length === 0) {
            addBotMessage("السلام، كيفاش نقدر نعاونك اليوم؟");
        }
        userInput.focus();
    });

    closeBtn.addEventListener('click', () => {
        chatbotContainer.classList.remove('visible');
        openChatBtn.style.display = 'flex';
    });

    fullscreenBtn.addEventListener('click', () => {
        chatbotContainer.classList.toggle('fullscreen');
    });

    // --- Logique de la discussion ---

    chatForm.addEventListener('submit', (e) => {
        e.preventDefault();
        const userMessage = userInput.value.trim();
        if (userMessage) {
            addUserMessage(userMessage);
            sendQuestionToBot(userMessage);
            userInput.value = '';
            userInput.focus();
        }
    });

    // Fonction pour ajouter un message UTILISATEUR à l'interface
    function addUserMessage(message) {
        const messageElement = document.createElement('div');
        messageElement.className = 'chat-message user-message';
        messageElement.textContent = message;
        chatBody.appendChild(messageElement);
        scrollToBottom();
    }

    // Fonction pour ajouter un message BOT à l'interface
    function addBotMessage(message) {
        const messageElement = document.createElement('div');
        messageElement.className = 'chat-message bot-message';
        // Pour un rendu correct des sauts de ligne venant du modèle
        messageElement.innerHTML = message.replace(/\n/g, '<br>');
        chatBody.appendChild(messageElement);
        scrollToBottom();
    }
    
    // Affiche l'indicateur de chargement "le bot est en train d'écrire"
    function showLoadingIndicator() {
        const loadingElement = document.createElement('div');
        loadingElement.className = 'chat-message bot-message loading';
        loadingElement.innerHTML = '<span></span><span></span><span></span>';
        loadingElement.id = 'loading-indicator';
        chatBody.appendChild(loadingElement);
        scrollToBottom();
    }

    // Retire l'indicateur de chargement
    function hideLoadingIndicator() {
        const loadingIndicator = document.getElementById('loading-indicator');
        if (loadingIndicator) {
            loadingIndicator.remove();
        }
    }

    // Fait défiler automatiquement la fenêtre de chat vers le bas
    function scrollToBottom() {
        chatBody.scrollTop = chatBody.scrollHeight;
    }

    // Envoie la question au backend et gère la réponse
    async function sendQuestionToBot(question) {
        showLoadingIndicator();
        try {
            const response = await fetch(API_URL, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ question: question }),
            });

            hideLoadingIndicator(); // Retirer le chargement dès qu'on reçoit une réponse

            if (!response.ok) {
                // Gérer les erreurs HTTP (ex: 500 Internal Server Error)
                throw new Error(`Erreur du serveur: ${response.status}`);
            }

            const data = await response.json();
            addBotMessage(data.answer);

        } catch (error) {
            console.error("Erreur lors de la communication avec l'API:", error);
            hideLoadingIndicator(); // S'assurer que le chargement est retiré même en cas d'erreur
            addBotMessage("عذراً، كاين شي مشكل تقني حالياً. حاول مرة أخرى من فضلك.");
        }
    }
});