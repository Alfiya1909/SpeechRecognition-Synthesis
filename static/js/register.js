document.addEventListener("DOMContentLoaded", function () {
    var form = document.getElementById("register");
    var password = document.getElementById("password");
    var confirmPassword = document.getElementById("confrim_password");
    var errorMessage = document.getElementById("error_message");
    var emailInput = document.getElementById("email");

    form.addEventListener("submit", function (event) {
        var emailPattern = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/;
        if (!emailPattern.test(emailInput.value)) {
            alert("Please enter a valid email address (e.g., user@example.com)");
            event.preventDefault();
        }
    });

    function validatePassword() {
        if (password.value !== confirmPassword.value) {
            errorMessage.style.display = "block"; // Show error message
            confirmPassword.setCustomValidity("Passwords do not match!");
        } else {
            errorMessage.style.display = "none"; // Hide error message
            confirmPassword.setCustomValidity(""); // Reset validation
        }
    }

    // Validate passwords on input
    password.addEventListener("input", validatePassword);
    confirmPassword.addEventListener("input", validatePassword);
    
});
