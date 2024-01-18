const webcam = new Webcam(document.getElementById("wc"));
let isPredicting = true;
const MODEL_URL = "./men_women_model.json";

async function predict() {
  //Load tfjs converted model
  const model = await tf.loadLayersModel(MODEL_URL);
  while (isPredicting) {
    const predictedClass = tf.tidy(() => {
      //Get webcam captured image for predicting
      const img = webcam.capture();
      const predictions = model.predict(img);

      return predictions.dataSync()[0];
    });
    const classId = await predictedClass;
    console.log(classId);
    var predictionText = "";

    //Binary classification
    if (classId >= 0.5) {
      predictionText = "Man";
    } else {
      predictionText = "Woman";
    }

    // Update the DOM with the prediction label and confidence
    if (predictionText == "Man") {
      document.getElementById("prediction").innerHTML = `
        <span style="font-size: 48px; color: blue;">${predictionText}</span><br>
        <span style="font-size: 24px;">${Math.round(classId * 100)}%</span>`;
    } else {
      document.getElementById("prediction").innerHTML = `
        <span style="font-size: 48px; color: pink;">${predictionText}</span><br>
        <span style="font-size: 24px;">${Math.round(
          (1 - classId) * 100
        )}%</span>`;
    }

    await tf.nextFrame();
  }
}

async function init() {
  await webcam.setup();
  const webcamElement = document.getElementById("wc");
  webcamElement.classList.add("mirrored");
  predict();
}

init();
