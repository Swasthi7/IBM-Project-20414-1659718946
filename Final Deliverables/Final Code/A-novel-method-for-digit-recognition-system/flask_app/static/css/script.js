var canvas,ctx;

var base_url = window.location.origin;
let model;
(async function(){
    console.log("model loading...");
    model = await tf.loadLayersModel("mnistCNN.h5")
    console.log("model loaded..");
})();


function preprocessCanvas(image) {

    let tensor = tf.browser.fromPixels(image).resizeNearestNeighbor([28, 28]).mean(2).expandDims(2).expandDims().toFloat();
    console.log(tensor.shape);
    return tensor.div(255.0);
}

document.getElementById('predict_button').addEventListener("click",async function(){
    var imageData = canvas.toDataURL();
    let tensor = preprocessCanvas(canvas);
    console.log(tensor)
    let predictions = await model.predict(tensor).data();
    console.log(predictions)
    let results = Array.from(predictions);
    displayLabel(results);
    console.log(results);
});


//output
function displayLabel(data) {
    var max = data[0];
    var maxIndex = 0;
    for (var i = 1; i < data.length; i++) {
      if (data[i] > max) {
        maxIndex = i;
        max = data[i];
      }
    }
document.getElementById('result').innerHTML = maxIndex;
document.getElementById('confidence').innerHTML = "Confidence: "+(max*100).toFixed(2) + "%";
}
