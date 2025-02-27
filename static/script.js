document.addEventListener('DOMContentLoaded', function() {
    // Mobile Navigation
    const navToggle = document.querySelector('.nav-toggle');
    const nav = document.querySelector('nav');

    navToggle.addEventListener('click', function() {
        nav.classList.toggle('active');
    });

    // Handle dropdowns in mobile view
    const dropdownLinks = document.querySelectorAll('nav ul li a:not(:only-child)');

    dropdownLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            if (window.innerWidth <= 768) {
                e.preventDefault();
                this.parentElement.classList.toggle('dropdown-active');
            }
        });
    });

    // Close menu when clicking outside
    document.addEventListener('click', function(e) {
        const isNavToggle = e.target.closest('.nav-toggle');
        const isNav = e.target.closest('nav');

        if (!isNavToggle && !isNav && nav.classList.contains('active')) {
            nav.classList.remove('active');
        }
    });

    // Smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            if (this.getAttribute('href').length > 1) {
                e.preventDefault();

                // Close mobile menu if open
                if (nav.classList.contains('active')) {
                    nav.classList.remove('active');
                }

                const targetId = this.getAttribute('href');
                const targetElement = document.querySelector(targetId);

                if (targetElement) {
                    window.scrollTo({
                        top: targetElement.offsetTop - 70,
                        behavior: 'smooth'
                    });
                }
            }
        });
    });

    // Population Chart (using Chart.js)
    const populationChartEl = document.getElementById('populationChart');
    if (populationChartEl) {
        populationChartEl.innerHTML = '<canvas></canvas>';
        const ctx = populationChartEl.querySelector('canvas').getContext('2d');

        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['2000', '2010', '2020', '2030 (Projected)', '2040 (Projected)'],
                datasets: [{
                    label: 'US Senior Population (Millions)',
                    data: [35, 41, 56, 73, 82],
                    backgroundColor: [
                        'rgba(91, 143, 185, 0.6)',
                        'rgba(91, 143, 185, 0.6)',
                        'rgba(91, 143, 185, 0.8)',
                        'rgba(91, 143, 185, 0.4)',
                        'rgba(91, 143, 185, 0.4)'
                    ],
                    borderColor: [
                        'rgba(91, 143, 185, 1)',
                        'rgba(91, 143, 185, 1)',
                        'rgba(91, 143, 185, 1)',
                        'rgba(91, 143, 185, 1)',
                        'rgba(91, 143, 185, 1)'
                    ],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Population (Millions)'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Year'
                        }
                    }
                }
            }
        });
    }

    // Chatbot functionality
    const chatMessages = document.getElementById('chatMessages');
    const userMessageInput = document.getElementById('userMessage');
    const sendMessageBtn = document.getElementById('sendMessage');

    if (chatMessages && userMessageInput && sendMessageBtn) {
        sendMessageBtn.addEventListener('click', sendMessage);
        userMessageInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });

        function sendMessage() {
            const message = userMessageInput.value.trim();
            if (message !== '') {
                // Add user message
                addMessage(message, 'user');
                userMessageInput.value = '';

                // Simulate bot response after a delay
                setTimeout(() => {
                    const botResponses = [
                        "I'd be happy to help with that. Can you provide more details?",
                        "Thank you for your question. Let me find that information for you.",
                        "I understand. Here are some resources that might help you.",
                        "That's a common concern for many seniors. Here's what you can do.",
                        "I can connect you with a care specialist who can assist with this specific need."
                    ];

                    const randomResponse = botResponses[Math.floor(Math.random() * botResponses.length)];
                    addMessage(randomResponse, 'bot');

                    // Scroll to the bottom of the chat
                    chatMessages.scrollTop = chatMessages.scrollHeight;
                }, 1000);
            }
        }

        function addMessage(text, sender) {
            const messageDiv = document.createElement('div');
            messageDiv.classList.add('message');
            messageDiv.classList.add(sender + '-message');
            messageDiv.textContent = text;
            chatMessages.appendChild(messageDiv);

            // Scroll to the bottom of the chat
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
    }

    // Service Filter Functionality
    const filterBtns = document.querySelectorAll('.filter-btn');
    const serviceItems = document.querySelectorAll('.service-item');

    if (filterBtns.length > 0 && serviceItems.length > 0) {
        filterBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                // Remove active class from all buttons
                filterBtns.forEach(b => b.classList.remove('active'));

                // Add active class to clicked button
                btn.classList.add('active');

                const filter = btn.getAttribute('data-filter');

                serviceItems.forEach(item => {
                    if (filter === 'all' || item.getAttribute('data-type') === filter) {
                        item.style.display = 'flex';
                    } else {
                        item.style.display = 'none';
                    }
                });
            });
        });
    }

    // Contact form handling
    const contactForm = document.querySelector('.contact-form');
    if (contactForm) {
        contactForm.addEventListener('submit', function(e) {
            e.preventDefault();
            const formData = new FormData(this);
            const formObject = {};

            formData.forEach((value, key) => {
                formObject[key] = value;
            });

            // Here you would typically send the data to your backend
            console.log('Form submitted with data:', formObject);

            // Show success message
            const successMessage = document.createElement('div');
            successMessage.className = 'success-message';
            successMessage.textContent = 'Thank you for contacting us! We will get back to you within 24 hours.';
            successMessage.style.color = '#5cb85c';
            successMessage.style.fontWeight = 'bold';
            successMessage.style.padding = '20px';
            successMessage.style.backgroundColor = '#f8fff8';
            successMessage.style.border = '1px solid #5cb85c';
            successMessage.style.borderRadius = '6px';
            successMessage.style.marginTop = '20px';

            // Replace form with success message
            contactForm.innerHTML = '';
            contactForm.appendChild(successMessage);
        });
    }

    // Check if we're on a specific route based on URL
    const currentPath = window.location.pathname;
    const sections = document.querySelectorAll('section');

    if (currentPath === '/about' || currentPath === '/services' ||
        currentPath === '/contact') {

        // Hide all sections first
        sections.forEach(section => {
            section.style.display = 'none';
        });

        // Show only the relevant section based on path
        let sectionId = currentPath.substring(1); // Remove the leading slash
        const activeSection = document.getElementById(sectionId);

        if (activeSection) {
            activeSection.style.display = 'flex';

            // Scroll to this section
            window.scrollTo({
                top: 0,
                behavior: 'smooth'
            });
        }
    }
});