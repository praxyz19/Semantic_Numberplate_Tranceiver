const form = document.querySelector("#uploadForm");
const imageInput = document.querySelector("#imageInput");
const fileNameDisplay = document.querySelector("#fileNameDisplay");
const submitBtn = document.querySelector("#submitBtn");

const inputPreview = document.querySelector("#inputPreview");
const inputPlaceholder = document.querySelector("#inputPlaceholder");
const platePreview = document.querySelector("#platePreview");
const receivedPreview = document.querySelector("#receivedPreview");
const reconPlate = document.querySelector("#reconPlate");
const reconScene = document.querySelector("#reconScene");

const snr = document.querySelector("#snr");
const snrValue = document.querySelector("#snrValue");
const noise = document.querySelector("#noise");
const noiseValue = document.querySelector("#noiseValue");

const dashboard = document.querySelector("#dashboard");
const metricBandwidth = document.querySelector("#metricBandwidth");
const metricSemantic = document.querySelector("#metricSemantic");
const metricCosine = document.querySelector("#metricCosine");
const metricAccuracy = document.querySelector("#metricAccuracy");

const inputMetaSize = document.querySelector("#inputMetaSize");
const semanticPayload = document.querySelector("#semanticPayload");
const compressionRatio = document.querySelector("#compressionRatio");
const receivedMeta = document.querySelector("#receivedMeta");
const charAccuracyDisplay = document.querySelector("#characterAccuracy");
const reconPsnr = document.querySelector("#reconPsnr");
const reconSsim = document.querySelector("#reconSsim");

snr.addEventListener("input", () => {
    snrValue.textContent = `${snr.value} dB`;
});

noise.addEventListener("input", () => {
    noiseValue.textContent = Number(noise.value).toFixed(2);
});

imageInput.addEventListener("change", () => {
    const file = imageInput.files[0];
    if (!file) return;
    
    fileNameDisplay.textContent = file.name;
    inputPreview.src = URL.createObjectURL(file);
    inputPlaceholder.style.display = "none";
    inputPreview.style.display = "block";
});

form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const file = imageInput.files[0];
    if (!file) return;

    const body = new FormData();
    body.append("image", file);
    body.append("snr_db", snr.value);
    body.append("channel_noise", noise.value);

    submitBtn.disabled = true;
    submitBtn.innerHTML = `<svg class="spin" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12a9 9 0 1 1-6.219-8.56"/></svg> Processing...`;

    try {
        const response = await fetch("/api/process", { method: "POST", body });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "Processing failed");

        // Update Images
        inputPreview.src = data.input_image;
        platePreview.src = data.extracted_plate;
        receivedPreview.src = data.received_semantic_map;
        reconPlate.src = data.reconstructed_plate;
        reconScene.src = data.reconstructed_scene;

        // Update Footers
        inputMetaSize.textContent = `Size: ${formatBytes(data.metrics.input_bytes)}`;
        semanticPayload.textContent = `Seq: [${data.metrics.semantic_text}] | Size: ${formatBytes(data.metrics.transmitted_bytes)}`;
        compressionRatio.textContent = `Saving: ${data.metrics.bandwidth_saving_percent}%`;
        receivedMeta.textContent = `Map Sim: ${data.metrics.map_cosine_similarity_percent}%`;
        charAccuracyDisplay.textContent = `Accuracy: ${data.metrics.character_accuracy_percent}%`;
        reconPsnr.textContent = `PSNR: ${formatMetric(data.metrics.psnr_db, "dB")}`;
        reconSsim.textContent = `SSIM: ${formatMetric(data.metrics.ssim)}`;

        // Update Dashboard
        dashboard.style.display = "block";
        metricBandwidth.textContent = `${data.metrics.bandwidth_saving_percent}%`;
        metricSemantic.textContent = `${data.metrics.semantic_similarity_percent}%`;
        metricCosine.textContent = `${data.metrics.image_cosine_similarity_percent}%`;
        metricAccuracy.textContent = `${data.metrics.character_accuracy_percent}%`;

    } catch (error) {
        alert(error.message);
    } finally {
        submitBtn.disabled = false;
        submitBtn.innerHTML = `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m5 12 7-7 7 7"/><path d="M12 19V5"/></svg> Run Semantic Transmission`;
    }
});

function formatMetric(value, suffix = "") {
    if (value === null || value === undefined) return "N/A";
    const trimmed = Number(value).toFixed(2);
    return suffix ? `${trimmed} ${suffix}` : trimmed;
}

function formatBytes(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

// Add CSS for spinning icon
const style = document.createElement('style');
style.textContent = `
    @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
    .spin { animation: spin 1s linear infinite; }
`;
document.head.appendChild(style);
