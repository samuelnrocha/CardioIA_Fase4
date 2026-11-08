let currentBlob = null;
let currentFilename = "exame_radiologico.png";

const fileInput = document.getElementById("fileInput");
const previewImg = document.getElementById("previewImg");
const dropzone = document.getElementById("dropzone");

dropzone.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", (e) => {
  if (e.target.files && e.target.files[0]) {
    const file = e.target.files[0];
    currentBlob = file;
    currentFilename = file.name;
    previewImg.src = URL.createObjectURL(file);
    previewImg.style.display = "block";
  }
});

async function loadSample(type) {
  const url = type === "cardio"
    ? "/static/samples/cardiomegalia_sample.png"
    : "/static/samples/normal_sample.png";
  const resp = await fetch(url);
  currentBlob = await resp.blob();
  currentFilename = type === "cardio" ? "cardiomegalia_sample.png" : "normal_sample.png";
  previewImg.src = url;
  previewImg.style.display = "block";

  if (type === "cardio") {
    document.getElementById("bpmInput").value = 134;
    document.getElementById("tempInput").value = 38.3;
    document.getElementById("fallInput").value = "true";
  } else {
    document.getElementById("bpmInput").value = 72;
    document.getElementById("tempInput").value = 36.5;
    document.getElementById("fallInput").value = "false";
  }
  await runAnalysis();
}

async function runAnalysis() {
  if (!currentBlob) {
    alert("Selecione uma radiografia ou clique em um dos botões de Exemplo Rápido.");
    return;
  }

  const formData = new FormData();
  formData.append("file", currentBlob, currentFilename);
  formData.append("bpm", document.getElementById("bpmInput").value);
  formData.append("temperatura", document.getElementById("tempInput").value);
  formData.append("queda_detectada", document.getElementById("fallInput").value);
  formData.append("limiar", document.getElementById("thrInput").value);

  const resp = await fetch("/api/v1/multimodal", {
    method: "POST",
    body: formData,
  });
  const data = await resp.json();
  renderResult(data);
}

function renderResult(data) {
  const cnn = data.analise_imagem_fase4;
  const isCardio = cnn.classe_id === 1;

  const badge = document.getElementById("diagBadge");
  badge.textContent = cnn.status_clinico;
  badge.className = "status-pill " + (isCardio ? "pill-critical" : "pill-normal");

  document.getElementById("diagTitle").textContent = cnn.diagnostico;
  document.getElementById("diagTitle").style.color = isCardio ? "#FF2E73" : "#10B981";

  const fill = document.getElementById("probFill");
  fill.style.width = `${cnn.confianca_pct}%`;
  fill.style.background = isCardio ? "#ED145B" : "#10B981";

  document.getElementById("confVal").textContent = `${cnn.confianca_pct}%`;
  document.getElementById("ictVal").textContent = cnn.ict_estimado;
  document.getElementById("timeVal").textContent = `${cnn.tempo_inferencia_ms} ms`;

  document.getElementById("multiScore").textContent = `${data.score_multimodal} / 100`;
  document.getElementById("multiClass").textContent = data.classificacao_risco_global;
  document.getElementById("multiConduta").textContent = data.conduta_integrada;

  const ul = document.getElementById("recList");
  ul.innerHTML = "";
  cnn.recomendacoes.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item;
    ul.appendChild(li);
  });
}
