const webcam = new Webcam(document.getElementById("wc"));
let isPredicting = true;
const MODEL_URL = "https://raw.githubusercontent.com/Kerollosl/Man_Woman_Binary_CNN_and_HTML_Webcam/main/men_women_model.json";

async function predict() {
  //Load tfjs converted model
  const model = await tf.loadLayersModel(MODEL_URL);
  while (isPredicting) {
    const predictedClass = tf.tidy(() => {
      //Get webcam captured image for predicting
      let img = webcam.capture();

      // Resize the image to 64x64
      img = tf.image.resizeBilinear(img, [64, 64]);
      
      // Make sure the image has a batch dimension
      //img = img.expandDims(0);

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
        <span style="font-size: 72px; color: blue;">${predictionText}</span><br>
        <span style="font-size: 36px;">${Math.round(classId * 100)}%</span>`;
    } else {
      document.getElementById("prediction").innerHTML = `
        <span style="font-size: 72px; color: pink;">${predictionText}</span><br>
        <span style="font-size: 36px;">${Math.round(
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
