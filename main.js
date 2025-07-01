let session = null;
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");


ctx.fillStyle = "black";
ctx.fillRect(0, 0, canvas.width, canvas.height);

ctx.lineWidth = 20;
ctx.lineCap = "round";
ctx.strokeStyle = "white";

let isDrawing = false;

canvas.addEventListener("mousedown", () => {
  isDrawing = true;
});

canvas.addEventListener("mouseup", () => {
  isDrawing = false;
  ctx.beginPath();
});

canvas.addEventListener("mousemove", draw);

function draw(e) {
  if (!isDrawing) return;
  const rect = canvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;

  ctx.lineTo(x, y);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(x, y);
}

function clearCanvas() {
  ctx.fillStyle = "black";  *
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.beginPath();
  document.getElementById("result").innerText = "Prediction: ...";
}


// Load ONNX model
async function loadModel() {
  if (!session) {
    session = await ort.InferenceSession.create("./mnist_cnn.onnx");
  }
  return session;
}
function preprocess(canvas) {
  const tempCanvas = document.createElement("canvas");
  tempCanvas.width = 28;
  tempCanvas.height = 28;
  const tempCtx = tempCanvas.getContext("2d");

  tempCtx.drawImage(canvas, 0, 0, 28, 28);
  const imageData = tempCtx.getImageData(0, 0, 28, 28).data;

  const input = new Float32Array(1 * 1 * 28 * 28);

  for (let i = 0; i < 28 * 28; i++) {
    const r = imageData[i * 4];
    input[i] = r / 255.0;
  }

  return new ort.Tensor("float32", input, [1, 1, 28, 28]);
}


async function predict() {
  const session = await loadModel();
  const inputTensor = preprocess(canvas);
  const feeds = { input: inputTensor };

  console.log(feeds)
  const results = await session.run(feeds);
  const output = results.output.data;
  console.log(output)

  const maxIndex = output.indexOf(Math.max(...output));
  document.getElementById("result").innerText =
    `Prediction: ${maxIndex}`;
}
