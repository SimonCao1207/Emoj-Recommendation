const express = require('express')
const app = express()
const port = 8080
const bodyParser = require('body-parser');

const cors = require("cors");
const axios = require('axios')

const allowedOrigins = ["http://172.20.41.58:3000/", "http://172.20.41.58:8080/"];

app.use(express.json());

app.use(cors({
    origin: '*'
}));

app.use((req, res, next) => {
    res.setHeader('Access-Control-Allow-Origin', '*');
    next();
})

app.post('/predict', (req, res) => {
  console.log(req.body.text);
  axios.post('http://localhost:8000/predict', {
    text: req.body.text
  }) 
  .then(pred => {
    res.send(pred.data);
  })
})

app.listen(port, () => {
  console.log(`Example app listening at http://localhost:${port}`)
})