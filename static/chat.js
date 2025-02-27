document.addEventListener('DOMContentLoaded', function() {
    // Chat UI elements
    const chatMessages = document.getElementById('chatMessages');
    const userMessageInput = document.getElementById('userMessage');
    const sendMessageBtn = document.getElementById('sendMessage');
    const sidebarToggle = document.getElementById('sidebarToggle');
    const chatSidebar = document.querySelector('.chat-sidebar');
    const suggestionChips = document.querySelectorAll('.suggestion-chip');

    // Check for mobile and add mobile sidebar toggle
    if (window.innerWidth <= 768) {
        const chatHeader = document.querySelector('.current-chat-info');
        const mobileSidebarToggle = document.createElement('button');
        mobileSidebarToggle.className = 'mobile-sidebar-toggle';
        mobileSidebarToggle.innerHTML = '<i class="fas fa-bars"></i>';
        chatHeader.prepend(mobileSidebarToggle);

        mobileSidebarToggle.addEventListener('click', function() {
            chatSidebar.classList.toggle('active');
        });
    }

    // Toggle sidebar
    if (sidebarToggle) {
        sidebarToggle.addEventListener('click', function() {
            chatSidebar.classList.toggle('collapsed');

            if (chatSidebar.classList.contains('collapsed')) {
                chatSidebar.style.width = '70px';
                this.innerHTML = '<i class="fas fa-chevron-right"></i>';
            } else {
                chatSidebar.style.width = '300px';
                this.innerHTML = '<i class="fas fa-chevron-left"></i>';
            }
        });
    }

    // Send message function
    function sendMessage(text) {
        if (!text.trim()) return;

        // Create and add user message to chat
        const currentTime = new Date();
        const hours = currentTime.getHours();
        const minutes = currentTime.getMinutes();
        const timeStr = `${hours}:${minutes < 10 ? '0' + minutes : minutes} ${hours >= 12 ? 'PM' : 'AM'}`;

        const userMessageHTML = `
            <div class="message user-message">
                <div class="message-content">
                    <div class="message-text">
                        <p>${text}</p>
                    </div>
                    <div class="message-time">${timeStr}</div>
                </div>
                <div class="message-avatar">
                    <i class="fas fa-user"></i>
                </div>
            </div>
        `;

        chatMessages.insertAdjacentHTML('beforeend', userMessageHTML);
        userMessageInput.value = '';

        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;

        // Simulate bot typing indicator
        const typingIndicator = document.createElement('div');
        typingIndicator.className = 'typing-indicator';
        typingIndicator.innerHTML = `
            <div class="message bot-message">
                <div class="message-avatar">
                    <i class="fas fa-robot"></i>
                </div>
                <div class="message-content">
                    <div class="typing-dots">
                        <span></span>
                        <span></span>
                        <span></span>
                    </div>
                </div>
            </div>
        `;

        chatMessages.appendChild(typingIndicator);
        chatMessages.scrollTop = chatMessages.scrollHeight;

        // Generate bot response after delay
        setTimeout(() => {
            // Remove typing indicator
            chatMessages.removeChild(typingIndicator);

            // Predefined responses based on user message
            let botResponse = "";
            const lowerText = text.toLowerCase();

            if (lowerText.includes('doctor') && lowerText.includes('appointment')) {
                botResponse = `
                    <p>I can help you manage your doctor appointment. Here are some options:</p>
                    <ul>
                        <li>I can set up reminders for your upcoming appointments</li>
                        <li>I can help you find transportation services to and from your appointment</li>
                        <li>I can find doctors in your network if you need a new healthcare provider</li>
                    </ul>
                    <p>Would you like me to help with any of these options?</p>
                `;
            } else if (lowerText.includes('medication') || lowerText.includes('medicine') || lowerText.includes('pills')) {
                botResponse = `
                    <p>Medication management is important. I can help you with:</p>
                    <ul>
                        <li>Setting up reminders for taking your medications</li>
                        <li>Information about potential drug interactions</li>
                        <li>Finding affordable prescription options</li>
                        <li>Tracking your medication schedule</li>
                    </ul>
                    <p>What specific help do you need with your medications?</p>
                `;
            } else if (lowerText.includes('lonely') || lowerText.includes('alone') || lowerText.includes('bored')) {
                botResponse = `
                    <p>I understand feeling lonely can be difficult. Here are some suggestions that might help:</p>
                    <ul>
                        <li>Local senior community centers in your area that host regular events</li>
                        <li>Virtual social groups and classes specifically for seniors</li>
                        <li>Volunteer opportunities that match your interests</li>
                        <li>Senior companion programs available through local organizations</li>
                    </ul>
                    <p>Would you like me to provide more specific information about any of these options?</p>
                `;
            } else if (lowerText.includes('medicare') || lowerText.includes('insurance') || lowerText.includes('coverage')) {
                botResponse = `
                    <p>Medicare and insurance questions can be complex. I can provide information about:</p>
                    <ul>
                        <li>Medicare enrollment periods and deadlines</li>
                        <li>The differences between Medicare Parts A, B, C, and D</li>
                        <li>Supplemental insurance options</li>
                        <li>Coverage for specific treatments or medications</li>
                    </ul>
                    <p>What specific Medicare or insurance information would you like?</p>
                `;
            } else {
                botResponse = `
                    <p>Thank you for your message. I'd be happy to help you with that. Could you provide a bit more information so I can give you the most relevant assistance?</p>
                    <p>I'm here to help with healthcare needs, daily living assistance, social activities, transportation, and many other elder care services.</p>
                `;
            }

            // Add bot response
            const botMessageHTML = `
                <div class="message bot-message">
                    <div class="message-avatar">
                        <i class="fas fa-robot"></i>
                    </div>
                    <div class="message-content">
                        <div class="message-sender">iElder Assistant</div>
                        <div class="message-text">
                            ${botResponse}
                        </div>
                        <div class="message-time">${timeStr}</div>
                    </div>
                </div>
            `;

            chatMessages.insertAdjacentHTML('beforeend', botMessageHTML);

            // Add suggestion chips based on the conversation
            const suggestionsHTML = `
                <div class="message-suggestions">
                    <button class="suggestion-chip">Tell me more about services</button>
                    <button class="suggestion-chip">I need help with something else</button>
                    <button class="suggestion-chip">Connect me to a human</button>
                </div>
            `;

            chatMessages.insertAdjacentHTML('beforeend', suggestionsHTML);

            // Add click handlers to new suggestion chips
            document.querySelectorAll('.suggestion-chip').forEach(chip => {
                chip.addEventListener('click', function() {
                    sendMessage(this.textContent);
                });
            });

            // Scroll to bottom
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }, 1500);
    }

    // Send message when button is clicked
    if (sendMessageBtn) {
        sendMessageBtn.addEventListener('click', function() {
            sendMessage(userMessageInput.value);
        });
    }

    // Send message when Enter key is pressed
    if (userMessageInput) {
        userMessageInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendMessage(userMessageInput.value);
            }
        });
    }

    // Suggestion chips click handler
    suggestionChips.forEach(chip => {
        chip.addEventListener('click', function() {
            sendMessage(this.textContent);
        });
    });

    // Chat history items click handler
    const historyItems = document.querySelectorAll('.history-item');
    historyItems.forEach(item => {
        item.addEventListener('click', function() {
            // Clear current chat except for the welcome message
            while (chatMessages.children.length > 2) {
                chatMessages.removeChild(chatMessages.lastChild);
            }

            // Add a system message about loading previous chat
            const systemMessage = document.createElement('div');
            systemMessage.className = 'message-date';
            systemMessage.textContent = 'Loading previous conversation...';
            chatMessages.appendChild(systemMessage);

            // Get conversation id from data-id attribute
            const conversationId = this.getAttribute('data-id');

            // Simulate loading previous conversation
            setTimeout(() => {
                chatMessages.removeChild(systemMessage);

                // Sample conversation data based on the selected history item
                if (conversationId === '1') {
                    sendMessage("I need help with my medication schedule");
                } else if (conversationId === '2') {
                    sendMessage("I need transportation to my doctor's appointment");
                } else if (conversationId === '3') {
                    sendMessage("I have questions about my Medicare coverage");
                }
            }, 1000);
        });
    });

    // New chat button
    const newChatBtn = document.querySelector('.menu-item[data-action="new-chat"]');
    if (newChatBtn) {
        newChatBtn.addEventListener('click', function() {
            // Clear current chat except for the welcome message
            while (chatMessages.children.length > 2) {
                chatMessages.removeChild(chatMessages.lastChild);
            }

            // Set focus to input field
            userMessageInput.focus();
        });
    }

    // Close sidebar when clicking outside on mobile
    document.addEventListener('click', function(e) {
        if (window.innerWidth <= 768 &&
            !e.target.closest('.chat-sidebar') &&
            !e.target.closest('.mobile-sidebar-toggle') &&
            chatSidebar.classList.contains('active')) {
            chatSidebar.classList.remove('active');
        }
    });

    // Initial focus on input field
    if (userMessageInput) {
        setTimeout(() => {
            userMessageInput.focus();
        }, 500);
    }
});