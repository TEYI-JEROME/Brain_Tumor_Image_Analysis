document.addEventListener('DOMContentLoaded', function () {
    const navLinks = document.querySelectorAll('.nav-link');
    const pages = document.querySelectorAll('.page');
    const imageUpload = document.getElementById('image-upload');
    const imagePreview = document.getElementById('image-preview');
    const addImageBtn = document.querySelector('.add-image');

    // Definition of dynamic messages for each class
    const resultMessages = {
        glioma: {
            meaning: "The analysis suggests the possible presence of a glioma, a tumor that develops from glial cells, which support neurons in the brain or spinal cord. Gliomas can vary in severity, ranging from slow-growing to more aggressive forms.",
            condition: "Common symptoms include:\n- Persistent headaches, often worse in the morning.\n- Seizures or unexplained epileptic episodes.\n- Difficulty with movement or balance.\n- Cognitive issues, such as memory problems or trouble concentrating.",
            advice: "It’s important to consult a neurologist for a thorough evaluation, such as an MRI or CT scan, to better understand your situation. Early diagnosis can lead to tailored treatment options and improve your well-being."
        },
        meningioma: {
            meaning: "The analysis indicates a possible meningioma, a tumor that forms in the meninges, the protective layers surrounding the brain and spinal cord. Most meningiomas are benign and slow-growing but may cause symptoms if they press on nearby structures.",
            condition: "Common symptoms include:\n- Headaches that worsen over time.\n- Vision problems, such as blurred vision or loss of peripheral vision.\n- Seizures.\n- Weakness or numbness in the arms or legs.",
            advice: "A suspected meningioma warrants evaluation by a healthcare professional, such as a neurologist or neurosurgeon. Imaging tests like an MRI can help clarify the diagnosis and guide follow-up or treatment if needed. Schedule an appointment to discuss your results with confidence."
        },
        notumor: {
            meaning: "The analysis indicates no brain tumor was detected, suggesting your brain appears normal based on this preliminary evaluation.",
            condition: "No specific tumor-related symptoms are expected. If you experience symptoms like headaches or dizziness, they could be due to non-tumor causes, such as stress or migraines.",
            advice: "While this analysis is reassuring, if you have unusual symptoms, such as persistent headaches or vision changes, consult a doctor for a comprehensive evaluation. Further tests can provide peace of mind."
        },
        pituitary: {
            meaning: "The analysis suggests a possible pituitary tumor, often a pituitary adenoma, which develops in the pituitary gland at the base of the brain, regulating hormones. These tumors are typically benign but may affect hormone levels or nearby structures.",
            condition: "Common symptoms include:\n- Headaches.\n- Vision problems, such as loss of peripheral vision or double vision.\n- Hormonal imbalances, leading to irregular menstrual cycles, unexplained breast milk production, or unusual fatigue.\n- Loss of libido or infertility.",
            advice: "If a pituitary tumor is suspected, an endocrinologist or neurologist can perform tests, such as an MRI or hormone level checks, to better understand your condition. Early management can help address symptoms and improve your quality of life."
        }
    };

    // Function to toggle pages
    function showPage(pageId) {
        pages.forEach(page => {
            page.classList.add('hidden');
            if (page.id === pageId) {
                page.classList.remove('hidden');
            }
        });
        navLinks.forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('data-section') === pageId) {
                link.classList.add('active');
            }
        });
    }

    // Handle sidebar navigation
    navLinks.forEach(link => {
        link.addEventListener('click', function (e) {
            e.preventDefault();
            const pageId = this.getAttribute('data-section');
            showPage(pageId);
        });
    });

    // Show home page by default
    showPage('home');

    // Handle image preview
    imageUpload.addEventListener('change', function () {
        const file = this.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function (e) {
                imagePreview.innerHTML = `<img src="${e.target.result}" alt="Uploaded Image">`;
            };
            reader.readAsDataURL(file);
        } else {
            imagePreview.innerHTML = '';
        }
    });

    // Handle add-image button (resets or re-triggers image upload)
    addImageBtn.addEventListener('click', function () {
        imageUpload.value = ''; // Clear the input
        imagePreview.innerHTML = ''; // Clear the preview
        imageUpload.click(); // Trigger file input
    });

    // Handle form submission for prediction
    document.getElementById('upload-form').addEventListener('submit', async function (e) {
        e.preventDefault();

        // Validation: Check if an image is selected
        if (!imageUpload.files[0]) {
            alert('Please select an image to upload.');
            return;
        }

        const form = new FormData();
        const modelSelect = document.getElementById('model-select');
        form.append('image', imageUpload.files[0]);
        form.append('model_type', modelSelect.value);

        const loading = document.getElementById('loading');
        const results = document.getElementById('results');
        const predictionEl = document.getElementById('prediction');
        const probabilitiesEl = document.getElementById('probabilities');
        const meaningEl = document.getElementById('meaning');
        const conditionEl = document.getElementById('condition');
        const adviceEl = document.getElementById('advice');

        loading.classList.remove('hidden');
        results.classList.add('hidden');

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: form
            });
            const data = await response.json();

            if (data.error) {
                alert('Error: ' + data.error);
                loading.classList.add('hidden');
                return;
            }

            // Display the prediction
            predictionEl.textContent = data.prediction;
            probabilitiesEl.innerHTML = '';
            for (const [className, prob] of Object.entries(data.probabilities)) {
                const li = document.createElement('li');
                li.textContent = `${className}: ${prob}`;
                probabilitiesEl.appendChild(li);
            }

            // Display dynamic comments
            const messages = resultMessages[data.prediction] || {
                meaning: "Unknown prediction result.",
                condition: "No specific information available.",
                advice: "Please consult a healthcare professional for further evaluation."
            };
            meaningEl.textContent = messages.meaning;
            conditionEl.textContent = messages.condition;
            adviceEl.textContent = messages.advice;

            loading.classList.add('hidden');
            results.classList.remove('hidden');
        } catch (error) {
            alert('An error occurred: ' + error.message);
            loading.classList.add('hidden');
        }
    });
});