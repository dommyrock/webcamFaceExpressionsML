const video = document.getElementById("video");
//We could copy our model across number of devices(gpu+s) and we shard the data
//Sharding data ---> means we will be processing input in parallel across our devices(so we can scale aprox linear with number of devices we hae...)

//Run ALL in parallel(load all models than start video)
Promise.all([
  faceapi.nets.tinyFaceDetector.loadFromUri("/models"),
  //registers diferent parts of the face
  faceapi.nets.faceLandmark68Net.loadFromUri("/models"),
  //allow api to recognise where face is (box around face)
  faceapi.nets.faceRecognitionNet.loadFromUri("/models"),
  //recognises face expression
  faceapi.nets.faceExpressionNet.loadFromUri("/models")
]).then(startVideo);

function startVideo() {
  navigator.getUserMedia({ video: {} }, stream => (video.srcObject = stream), err => console.error(err));
}
//event listener(called every 100ms)
video.addEventListener("play", () => {
  const canvas = faceapi.createCanvasFromMedia(video);
  document.body.append(canvas);
  //get displaysize so canvas can be perfectly sized over the video
  const displaySize = { width: video.width, height: video.height };
  faceapi.matchDimensions(canvas, displaySize);
  setInterval(async () => {
    //detect all faces
    const detections = await faceapi
      .detectAllFaces(video, new faceapi.TinyFaceDetectorOptions())
      .withFaceLandmarks() //when we draw face on the scren with dots
      .withFaceExpressions(); //determine face expression
    console.log(detections);

    //to make sure boxes are properly sized on top of video
    const resizedDetections = faceapi.resizeResults(detections, displaySize);
    canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height); //clear canvas before we draw again
    faceapi.draw.drawDetections(canvas, resizedDetections);
    faceapi.draw.drawFaceLandmarks(canvas, resizedDetections);
    faceapi.draw.drawFaceExpressions(canvas, resizedDetections);
  }, 100);
});
