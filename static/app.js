const $ = (selector) => document.querySelector(selector);
const video = $("#preview");
const answer = $("#answer");
const question = $("#question");
const status = $("#status");
const share = $("#share");
const stop = $("#stop");
const live = $("#live");
const interval = $("#interval");
let stream;
let timer;
const history = [];

function setStatus(text) { status.textContent = text; }
function setCaptureState(active) {
  share.disabled = active;
  stop.disabled = !active;
  live.disabled = !active;
  interval.disabled = !active;
  $("#empty-preview").hidden = active;
}

async function startCapture() {
  try {
    stream = await navigator.mediaDevices.getDisplayMedia({ video: { frameRate: 3 }, audio: false });
    video.srcObject = stream;
    stream.getVideoTracks()[0].addEventListener("ended", endCapture);
    setCaptureState(true);
    setStatus("Tela conectada");
  } catch (error) {
    setStatus("Compartilhamento cancelado");
  }
}
function endCapture() {
  clearInterval(timer); timer = undefined;
  if (stream) stream.getTracks().forEach((track) => track.stop());
  stream = undefined; video.srcObject = null; live.checked = false;
  setCaptureState(false); setStatus("Tela desconectada");
}
function captureFrame() {
  if (!stream || !video.videoWidth) return null;
  const canvas = document.createElement("canvas");
  const ratio = Math.min(1, 1280 / video.videoWidth);
  canvas.width = Math.round(video.videoWidth * ratio);
  canvas.height = Math.round(video.videoHeight * ratio);
  canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL("image/jpeg", 0.82);
}
function render(text) {
  answer.innerHTML = text
    .replace(/&/g, "&amp;").replace(/</g, "&lt;")
    .replace(/^## (.*)$/gm, "<h3>$1</h3>")
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    .replace(/\n/g, "<br>");
}
async function ask(customQuestion) {
  const prompt = (customQuestion || question.value || "Explique o inglês que aparece nesta tela.").trim();
  const image = captureFrame();
  if (!image && !history.length) { setStatus("Compartilhe uma tela primeiro"); return; }
  $("#explain").disabled = true; setStatus("Professor analisando...");
  try {
    const response = await fetch("/api/explain", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ image_data_url: image, question: prompt, history }) });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Erro desconhecido");
    render(data.answer); history.push({ role: "user", text: prompt }, { role: "assistant", text: data.answer });
    if (history.length > 12) history.splice(0, 2);
    question.value = ""; setStatus("Explicação pronta");
  } catch (error) { render(`**Não consegui analisar agora.** ${error.message}`); setStatus("Verifique a configuração"); }
  finally { $("#explain").disabled = false; }
}
share.addEventListener("click", startCapture); stop.addEventListener("click", endCapture);
live.addEventListener("change", () => {
  clearInterval(timer);
  if (live.checked) { timer = setInterval(() => ask("Explique o novo inglês visível nesta tela. Só responda se houver uma frase nova e legível."), Number(interval.value) * 1000); ask("Explique o inglês visível nesta tela."); }
});
interval.addEventListener("change", () => { if (live.checked) { live.checked = false; live.dispatchEvent(new Event("change")); live.checked = true; live.dispatchEvent(new Event("change")); } });
$("#question-form").addEventListener("submit", (event) => { event.preventDefault(); ask(); });
question.addEventListener("keydown", (event) => { if (event.ctrlKey && event.key === "Enter") { event.preventDefault(); ask(); } });

