document.addEventListener('DOMContentLoaded', function() {
    // DOM elements
    const generateBtn = document.getElementById('generate-btn');
    const nameDisplay = document.getElementById('name-display');
    
    // Animation variables
    let isAnimating = false;
    
    // Add click event to generate button
    generateBtn.addEventListener('click', async function() {
        // Prevent multiple clicks during animation
        if (isAnimating) return;
        
        // Start animation
        isAnimating = true;
        
        // Add loading animation
        nameDisplay.innerHTML = '<p class="name-text">Finding a name...</p>';
        nameDisplay.classList.add('shake-animation');
        
        try {
            // Fetch random name from API
            const response = await fetch(`${window.location.origin}/api/generate-name`);
            
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            
            const data = await response.json();
            
            // Add some delay for animation effect (300ms)
            setTimeout(() => {
                // Remove animation class
                nameDisplay.classList.remove('shake-animation');
                
                // Create elements for displaying the name with comic effect
                const nameContainer = document.createElement('div');
                nameContainer.classList.add('generated-name');
                
                // Add the name with a comic speech bubble style
                nameContainer.innerHTML = `
                    <p class="name-text">${data.name}</p>
                    <div class="sparkle-effect"></div>
                `;
                
                // Clear previous content and add new name
                nameDisplay.innerHTML = '';
                nameDisplay.appendChild(nameContainer);
                
                // Add appear animation
                nameContainer.classList.add('pop-in');
                
                // Reset animation flag after animation completes
                setTimeout(() => {
                    isAnimating = false;
                }, 500);
                
            }, 300);
            
        } catch (error) {
            console.error('Error fetching name:', error);
            nameDisplay.innerHTML = '<p class="name-text error">Oops! Something went wrong. Try again!</p>';
            nameDisplay.classList.remove('shake-animation');
            isAnimating = false;
        }
    });
    
    // Add bounce effect on button hover
    generateBtn.addEventListener('mouseenter', function() {
        if (!isAnimating) {
            this.classList.add('bounce');
        }
    });
    
    generateBtn.addEventListener('mouseleave', function() {
        this.classList.remove('bounce');
    });
    
    // Initial button pulse animation
    setTimeout(() => {
        generateBtn.classList.add('pulse');
        setTimeout(() => {
            generateBtn.classList.remove('pulse');
        }, 1000);
    }, 1500);
});
